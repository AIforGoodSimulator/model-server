import time
import logging
import numpy as np
from mesa import Model
from numba import njit
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace

from ai4good.models.abm.mesa.agent import Person
from ai4good.models.abm.initialise_parameters import Parameters
from ai4good.models.abm.mesa.utils import read_age_gender
from ai4good.models.abm.mesa.common import Route
from ai4good.models.abm.mesa.common import DiseaseStage
from ai4good.models.abm.mesa.helper import CampHelper, PersonHelper


class Camp(Model, CampHelper):
    """
    Modelling Moria camp
    # TODO: can we add thread locks/semaphore to share resources and then simulate all agents in parallel?
    """

    # Side length of square shaped camp
    # In tucker model, CAMP_SIZE=1.0. However, any value can be set here since all distance calculations are done
    # relative to `CAMP_SIZE`
    CAMP_SIZE = 100.0

    # If a susceptible and an infectious individual interact, then the infection is transmitted with probability pa
    # Fang and colleagues (2020)
    # TODO: this is baseline value, parameterize it
    Pa = 0.1

    def __init__(self, params: Parameters):
        super().__init__()
        self.params = params

        if self.params.camp.upper() != 'MORIA':
            raise NotImplementedError("Only Moria camp is implemented for abm at the moment")

        # all agents are susceptible before simulation starts
        self.agents_disease_states = np.array([DiseaseStage.SUSCEPTIBLE for _ in range(self.people_count)])

        # get age and gender of the agents
        self.agents_age = read_age_gender(self.people_count)[:, 0]
        self.agents_gender = read_age_gender(self.people_count)[:, 1]

        # In our baseline model, we assume that females and individuals under the age of 10 use home ranges with radius
        # 0.02 (`smaller_movement_radius`), and that males over the age of 10 use home ranges with radius 0.1
        # (`larger_movement_radius`)
        self.agents_home_ranges = np.array([
            self.params.smaller_movement_radius * Camp.CAMP_SIZE if (self.agents_gender[i] == 0 or
                                                                     self.agents_age[i] < 10)
            else self.params.larger_movement_radius * Camp.CAMP_SIZE
            for i in range(self.people_count)
        ])

        # randomly position agents throughout the camp
        self.agents_pos = Camp.CAMP_SIZE * np.random.random((self.people_count, 2))

        # initially all agents are in their respective households
        self.agents_route = np.array([Route.HOUSEHOLD] * self.people_count)

        # [id, x, y] of each household
        self.households = self.get_households(self.params.number_of_isoboxes, self.params.number_of_tents)

        self.agents_households = self.assign_households_to_agents()  # household ids of the agents (does not change)
        self.agents_ethnic_groups = np.array([])  # TODO: can we remove/de-prioritize it for abm?

        # There are 144 toilets evenly distributed throughout the camp. Toilets are placed at the centres of the
        # squares that form a 12 x 12 grid covering the camp (baseline)
        self.toilets = self._position_blocks(self.params.toilets_blocks[0])
        self.toilets_queue = {}  # dict containing toilet_id: [ list of agents unique ids ] for toilet occupancy

        # The camp has one food line (baseline)
        # each person going to the food line to collect food will enter this queue
        self.foodlines = self._position_blocks(self.params.foodline_blocks[0])  # initial food lines
        self.foodline_queue = {}  # dict containing foodline_id: [ list of agents unique ids ] who are standing in line

        # mesa scheduler
        self.schedule = RandomActivation(self)
        # mesa space
        self.space = ContinuousSpace(x_max=Camp.CAMP_SIZE, y_max=Camp.CAMP_SIZE, torus=False)

        # randomly select one person and mark as "Exposed"
        self.agents_disease_states[np.random.randint(0, high=self.people_count)] = DiseaseStage.EXPOSED

        # add agents to the model
        for i in range(self.people_count):
            p = Person(i, self)
            self.schedule.add(p)
            self.space.place_agent(p, self.agents_pos[i, :])

    def step(self):
        # simulate 1 day in camp

        # check if simulation can stop
        if self.stop_simulation(self.agents_disease_states):
            return

        # step all agents
        self.schedule.step()

        # clear some model day-wise variables
        self.foodline_queue = {}  # clear food line queue at end of the day
        self.toilets_queue = {}  # clear toilet queues at end of the day

    def simulate(self):
        t1 = time.time()  # start of the model execution
        for _ in range(self.params.number_of_steps):
            self.step()
        t2 = time.time()  # end of the model execution
        logging.info("Completed x{} steps in {} seconds".format(self.params.number_of_steps, t2 - t1))

    @staticmethod
    @njit
    def stop_simulation(disease_states) -> int:
        # We ran each simulation until all individuals in the population were either susceptible or recovered, at which
        # point the epidemic had ended
        n = disease_states.shape[0]
        for i in range(n):
            if disease_states[i] not in [DiseaseStage.SUSCEPTIBLE, DiseaseStage.RECOVERED]:
                # DO NOT stop the simulation if any person is NOT (susceptible or recovered)
                return 0

        # if all agents are either susceptible or recovered, time to stop the simulation
        return 1

    def apply_interventions(self, vt=None, lockdown=None, sector=None, isolation=None) -> None:
        """
        Add interventions

        Parameters
        ----------
            vt: Transmission Reduction factor
                Behaviours including wearing face masks, frequent hand washing, and maintaining a safe distance from
                others can reduce the risk of COVID-19 transmission (refs). To simulate transmission reduction
                interventions, we scaled the per interaction transmission probability pa by a factor of vt
                i.e. (new Pa) = (old Pa) * vt

            lockdown: A dict {rl: float, wl: float} containing lockdown parameters (refer tucker model)
                In lockdown, home range values of all individuals are set to `r1`. `wl` is the proportion of the people
                who violate the lockdown home range value.
                Personal assumption: In ABM, we will treat `wl` as the probability that a person will violate lockdown
                at any given moment

            sector: Divide the camp into an `sector` x `sector` grid of squares (or sectors), each with its
                own food line

            isolation: A dict {b: float, n: int} containing isolation parameters (refer tucker model)
                Each individual with symptoms (i.e., symptomatic, mild case, or severe case) is detected with
                probability `b` on each day. If an individual with symptoms is detected, that individual and its
                household are removed from the camp.
                Individuals are returned to the camp `n` days (baseline=7) after they have recovered, or if they do not
                become infected, `n` days after the last infected person in their household has recovered

        """

        # In Moria, there is approximately one tap per 42 people, so frequent hand washing (e.g., greater than 10x per
        # day, as in Jefferson et al. 2009) may be impossible. Due to the high population density in Moria
        # (~20,000 people km-2), maintaining safe distances among people may also be difficult or impossible.
        # However, people in Moria have been provided with face masks. We simulated a population in which all
        # individuals wear face masks outside their homes by setting vt = 0.32 (Jefferson et al. 2009)
        if vt is not None:
            Camp.Pa = Camp.Pa * vt

        # Some countries have attempted to limit the spread of COVID-19 by requiring people to stay in or close to
        # their homes (ref). This intervention has been called "lockdown". We simulated a lockdown in which most
        # individuals are restricted to a home range with radius rl around their households. We assumed that a
        # proportion wl of the population will violate the lockdown. Thus, for each individual in the population,
        # we set their home range to rl with probability (1- wl), and to 0.1 otherwise. By manipulating rl and wl we
        # simulated lockdowns that are more or less restrictive and/or strictly enforced.
        if lockdown is not None:
            rl = lockdown['rl']  # new home range (for all agents)
            wl = lockdown['wl']  # proportion of people who violate lockdown
            # TODO
            pass

        # The camp in our baseline model has a single food line, where transmission can potentially occur between two
        # individuals from any parts of the camp. This facilitates the rapid spread of COVID-19 infection. A plausible
        # intervention would be to divide the camp into sectors with separate food lines, and require individuals to
        # use the food line closest to their households. To simulate this intervention, we divide the camp into an
        # n x n grid of squares, each with its own food line
        if sector is not None:
            # create food lines based on `sector` parameter
            # `foodline` will contain new positions of the foodlines
            self.foodlines = self._position_blocks(sector)

            # assign agents to foodline of their sector (nearest ones to their household)
            for a in self.schedule.agents:
                # update agent's food line id
                # find food line nearest to household of the agent
                a.foodline_id = PersonHelper.find_nearest(a.__getattribute__("household_center"), self.foodlines)

            # reset food line queues
            # TODO: double check
            self.foodline_queue = {}

        # Managers of some populations, including Moria, have planned interventions in which people with COVID-19
        # infections and their households will be removed from populations and kept in isolation until the infected
        # people have recovered. To simulate such remove-and-isolate interventions, we conduct simulations in which in
        # each individual with symptoms (i.e., symptomatic, mild case, or severe case) is detected with probability b
        # on each day. If an individual with symptoms is detected, that individual and its household are removed from
        # the camp. Individuals removed from the camp can infect or become infected by others in their household
        # following equation (2), but cannot infect or become infected by individuals in other households by any
        # transmission route. We assume that individuals are returned to the camp 7 days after they have recovered, or
        # if they do not become infected, 7 days after the last infected person in their household has recovered. By
        # setting different values of b, we can simulate remove-and-isolate interventions with
        # different detection efficiencies.
        if isolation is not None:
            b = isolation['b']  # probability that camp manager detects agent with symptoms
            n = isolation['n']  # number of days after recovery when agent can go back to camp
            # TODO
            pass

    @property
    def people_count(self):
        return self.params.total_population

    def get_filter_array(self):
        # pass compatible `people` array to be passed to `_filter_agents` function
        # need agent's details columns: route, household_id, disease state
        return np.dstack([
            self.agents_route,
            self.agents_households,
            self.agents_disease_states
        ], axis=0)

    def assign_households_to_agents(self):
        # assign households to agents based on capacity
        # Iso-boxes are prefabricated housing units with a mean occupancy of 10 individuals
        # Tents have a mean occupancy of 4 individuals.

        # check if provided population can be fit into given number of households
        camp_capacity = self.params.number_of_people_in_one_isobox * self.params.number_of_isoboxes + \
                        self.params.number_of_people_in_one_tent * self.params.number_of_tents
        assert camp_capacity < self.people_count, \
            "Number of people ({}) exceeds camp capacity ({})".format(self.people_count, camp_capacity)

        out = np.zeros(shape=(camp_capacity,), dtype=np.int32)  # array containing household id for each agent
        o = 0  # counter for `out`

        # shuffle iso-boxes
        iso_boxes = np.arange(self.params.number_of_isoboxes)
        np.random.shuffle(iso_boxes)

        # assign people to iso-boxes in order
        for i in iso_boxes:
            out[o: o + self.params.number_of_people_in_one_isobox] = i
            o = o + self.params.number_of_people_in_one_isobox

        # shuffle tents
        tents = np.arange(self.params.number_of_tents)
        np.random.shuffle(tents)

        # assign people to tents in order
        for t in tents:
            out[o: o + self.params.number_of_people_in_one_tent] = t
            o = o + self.params.number_of_people_in_one_tent

        return out[:self.people_count]

    def get_households(self, num_iso_boxes, num_tents):
        """
        Parameters
        ----------
            num_iso_boxes: Number of iso boxes in the camp
            num_tents: Number of tents in the camp

        Returns
        -------
            out: An 2D array (?, 3) containing id and x,y co-ordinates of the households

        """

        # get positions and ids of iso-boxes
        iso_boxes_pos = self.get_iso_boxes(num_iso_boxes, self.params.area_covered_by_isoboxes)
        iso_boxes_ids = np.arange(0, num_iso_boxes)

        # get positions and ids of tents
        tents_pos = self.get_tents(num_tents, self.params.area_covered_by_isoboxes)
        tents_ids = np.arange(num_iso_boxes, num_iso_boxes + num_tents)

        iso_boxes = np.concatenate([iso_boxes_ids, iso_boxes_pos], axis=0)  # join ids and co-ordinates of iso-boxes
        tents = np.concatenate([tents_ids, tents_pos], axis=0)  # join ids and co-ordinates of tents

        # merge iso-boxes and tents
        households = np.concatenate([iso_boxes, tents], axis=1)
        np.random.shuffle(households)

        return households  # return household data

    @staticmethod
    def get_iso_boxes(num: int, iso_area_ratio: float) -> np.array:
        """
        Get positions of the iso-boxes in the camp.
        Iso-boxes are assigned to random locations in a central square that covers one half of the area of the camp

        Parameters
        ----------
            num: Number of iso-boxes in the camp
            iso_area_ratio: The portion of the camp area (0->1) that is occupied by iso-boxes

        Returns
        -------
            out: (?, 2) array containing co-ordinates of the iso-boxes in the camp
        """

        # Iso-boxes are assigned to random locations in a central square that covers one half of the area of the camp

        # the side length of central square
        center_sq_side = Camp.CAMP_SIZE * iso_area_ratio**0.5

        # minimum and maximum co-ordinates for central square
        p_min = (Camp.CAMP_SIZE - center_sq_side) / 2.0
        p_max = (Camp.CAMP_SIZE + center_sq_side) / 2.0

        pos = (p_max - p_min) * np.random.random(size=(num, 2)) + p_min  # choose random positions from central square

        return pos  # return iso boxes co-ordinates

    @staticmethod
    def get_tents(num: int, iso_area_ratio: float) -> np.array:
        """
        Get positions of the tents in the camp.
        Tents are assigned to random locations in the camp outside of the central square

        Parameters
        ----------
            num: Number of tents in the camp
            iso_area_ratio: The portion of the camp area (0->1) that is occupied by iso-boxes

        Returns
        -------
            out: (?, 2) array containing co-ordinates of the tents in the camp
        """

        # The area outside the central square can be divided into 4 parts (bottom, right, top, left)
        # Below is the positions of tents distributed in all these 4 parts

        # the side length of central square
        center_sq_side = Camp.CAMP_SIZE * iso_area_ratio ** 0.5

        min1 = 0.0  # minimum co-ordinate for the region outside central square
        max1 = (Camp.CAMP_SIZE - center_sq_side) / 2.0  # co-ordinate of first edge of central square
        min2 = (Camp.CAMP_SIZE + center_sq_side) / 2.0  # co-ordinate of second edge of central square
        max2 = Camp.CAMP_SIZE  # co-ordinate of camp end

        assert num >= 4, "For calculations, we need minimum 4 tents"

        # assign few tents at the region below central square
        bottom_num = int(num / 4)
        bottom = np.dstack((
            (max2 - min1) * np.random.random((bottom_num,)) + min1,  # X co-ordinate
            (max1 - min1) * np.random.random((bottom_num,)) + min1  # Y co-ordinate
        ))

        # assign few tents at the region on right of central square
        right_num = int(num / 4)
        right = np.dstack((
            (max2 - min2) * np.random.random((right_num,)) + min2,  # X co-ordinate
            (min2 - max1) * np.random.random((right_num,)) + max1  # Y co-ordinate
        ))

        # assign few tents at the region above central square
        top_num = int(num / 4)
        top = np.dstack((
            (max2 - min1) * np.random.random((top_num,)) + min1,  # X co-ordinate
            (max2 - min2) * np.random.random((top_num,)) + min2  # Y co-ordinate
        ))

        # assign few tents at the region on left of central square
        left_num = num - (bottom_num + right_num + top_num)
        left = np.dstack((
            (max1 - min1) * np.random.random((left_num,)) + min1,  # X co-ordinate
            (min2 - max1) * np.random.random((left_num,)) + max1  # Y co-ordinate
        ))

        # merge all positions
        pos = np.concatenate([bottom, right, top, left], axis=1).squeeze()
        np.random.shuffle(pos)

        return pos  # return tents co-ordinates
