import logging
import numpy as np
from tqdm import tqdm
from mesa import Model
from numba import njit
from scipy.cluster.vq import kmeans
from mesa.space import ContinuousSpace
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

from ai4good.models.abm.mesa_impl.common import *
from ai4good.models.abm.mesa_impl.agent import Person
from ai4good.models.abm.initialise_parameters import Parameters
from ai4good.models.abm.mesa_impl.helper import CampHelper, PersonHelper
from ai4good.models.abm.mesa_impl.utils import read_age_gender, get_incubation_period, log


class Camp(Model, CampHelper):
    """
    Modelling Moria camp
    # TODO: can we add thread locks/semaphore to share resources and then simulate all agents in parallel?
    """

    # ethnic groups and their proportions in the camp
    # this is referred from previous abm.py file
    # TODO: tucker model refers to 8 ethnic groups. But abm.py contained only 7. verify this.
    # TODO: In abm.py, people count per ethnic group was mentioned. We have transformed it into proportions to work for
    # TODO: any population size
    ethnic_groups = [  # [ethnic group name, proportion of people in ethnic group]
        ['afghan', 7919 / 10135],
        ['cameroon', 149 / 10135],
        ['congo', 706 / 10135],
        ['iran', 107 / 10135],
        ['iraq', 83 / 10135],
        ['somalia', 442 / 10135],
        ['syria', 729 / 10135]
    ]

    @log(name="camp initialization")
    def __init__(self, params: Parameters):
        super(Camp, self).__init__()

        self.params = params

        # If a susceptible and an infectious individual interact, then the infection is transmitted with probability pa
        # Fang and colleagues (2020)
        # TODO: this is baseline value, parameterize it
        # TODO: should this be picked from some distribution for each agent or having fixed value is fine?
        self.Pa = 0.1

        # Probability that camp managers will detect people with symptoms
        # Initially, this value will be 0.0, but by adding intervention via `apply_interventions` (isolation) we can
        # change this value
        self.P_detect = 0.0
        self.P_n = 0

        if self.params.camp.upper() != 'MORIA':
            raise NotImplementedError("Only Moria camp is implemented for abm at the moment")

        # day counter for each agent to track the state of the disease (this is NOT simulation day counter)
        self.agents_day_counter = np.zeros((self.people_count,), dtype=np.int32)

        # all agents are susceptible before simulation starts
        self.agents_disease_states = np.array([SUSCEPTIBLE for _ in range(self.people_count)])

        # incubation period : number of days from exposure until symptoms appear
        self.agents_incubation_periods = get_incubation_period(self.people_count)

        # get age and gender of the agents
        self.agents_age = read_age_gender(self.people_count)[:, 0]
        self.agents_gender = read_age_gender(self.people_count)[:, 1]

        # In our baseline model, we assume that females and individuals under the age of 10 use home ranges with radius
        # 0.02 (`smaller_movement_radius`), and that males over the age of 10 use home ranges with radius 0.1
        # (`larger_movement_radius`)
        self.agents_home_ranges = np.array([
            self.params.smaller_movement_radius * CAMP_SIZE if (self.agents_gender[i] == 0 or self.agents_age[i] < 10)
            else self.params.larger_movement_radius * CAMP_SIZE
            for i in range(self.people_count)
        ])

        # randomly position agents throughout the camp
        # TODO: should this be changed to household center co-ordinates instead? as we are using initial route=HOUSEHOLD
        self.agents_pos = CAMP_SIZE * np.random.random((self.people_count, 2))

        # initially all agents are in their respective households
        self.agents_route = np.array([HOUSEHOLD] * self.people_count)

        self.agents_ethnic_groups = self._assign_ethnicity_to_agents()  # ethnic group ids of the agents

        # [id, capacity, x, y] of each household
        # TODO: not on priority as will require changes in various code blocks
        # TODO: Right now each household is just a point in space. It should have some area for an ABM
        self.households = self._get_households()

        self.agents_households = self._assign_households_to_agents()  # household ids of the agents (does not change)

        # There are 144 toilets evenly distributed throughout the camp. Toilets are placed at the centres of the
        # squares that form a 12 x 12 grid covering the camp (baseline)
        self.toilets = self._position_blocks(self.params.toilets_blocks[0])
        self.toilets_queue = {}  # dict containing toilet_id: [ list of agents unique ids ] for toilet occupancy

        # The camp has one food line (baseline)
        # each person going to the food line to collect food will enter this queue
        self.foodlines = self._position_blocks(self.params.foodline_blocks[0])  # initial food lines
        self.foodline_queue = {}  # dict containing foodline_id: [ list of agents unique ids ] who are standing in line

        # mesa_impl scheduler
        self.schedule = SimultaneousActivation(self)
        # mesa_impl space
        self.space = ContinuousSpace(x_max=CAMP_SIZE, y_max=CAMP_SIZE, torus=False)

        # randomly select one person and mark as "Exposed"
        self.agents_disease_states[np.random.randint(0, high=self.people_count)] = EXPOSED

        # add agents to the model
        for i in range(self.people_count):
            # create agent
            p = Person(i, self)
            # add agent to simulation scheduler
            self.schedule.add(p)
            # place agent in the camp
            self.space.place_agent(p, self.agents_pos[i, :])

        # data collector
        # TODO
        # most data is maintained by model, so `agent_reporters` may not be needed
        self.data_collector = DataCollector(
            model_reporters={},
            agent_reporters={}
        )

    def step(self):
        # simulate 1 day in camp

        logging.info("Running step: {}".format(self.schedule.steps))

        # step all agents
        self.schedule.step()

        # clear some model day-wise variables
        self.foodline_queue = {}  # clear food line queue at end of the day
        self.toilets_queue = {}  # clear toilet queues at end of the day

    @log(name="simulation")
    def simulate(self) -> None:
        # simulate for number of days

        logging.info("Starting simulation for x{} days".format(self.params.number_of_steps))
        for t in tqdm(range(self.params.number_of_steps)):
            # check if simulation can stop
            if self.stop_simulation(self.agents_disease_states):
                return

            # simulate 1 day
            self.step()

            # collect data after end of every day
            self.data_collector.collect(self)

    @staticmethod
    @njit
    def stop_simulation(disease_states) -> int:
        # We ran each simulation until all individuals in the population were either susceptible or recovered, at which
        # point the epidemic had ended
        n = disease_states.shape[0]
        for i in range(n):
            if disease_states[i] not in [SUSCEPTIBLE, RECOVERED]:
                # DO NOT stop the simulation if any person is NOT (susceptible or recovered)
                return 0

        # if all agents are either susceptible or recovered, time to stop the simulation
        return 1

    def apply_interventions(self, vt=None, lockdown=None, sector=None, isolation=None) -> None:
        """
        Add interventions. This method should be called once when intervention is applied. It need not be called at
        each step.

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
            # scale transmission probability
            self.Pa = self.Pa * vt
            # propagate new value to all agents
            for a in self.schedule.agents:
                a.Pa = self.Pa

        # Some countries have attempted to limit the spread of COVID-19 by requiring people to stay in or close to
        # their homes (ref). This intervention has been called "lockdown". We simulated a lockdown in which most
        # individuals are restricted to a home range with radius rl around their households. We assumed that a
        # proportion wl of the population will violate the lockdown. Thus, for each individual in the population,
        # we set their home range to rl with probability (1- wl), and to 0.1 otherwise. By manipulating rl and wl we
        # simulated lockdowns that are more or less restrictive and/or strictly enforced.
        if lockdown is not None:
            rl = lockdown['rl']  # new home range (for all agents)
            wl = lockdown['wl']  # proportion of people who violate lockdown

            # actual probabilities that agents will violate lockdown
            will_violate = np.random.random(size=(self.people_count,))

            for i, a in enumerate(self.schedule.agents):  # iterate over all the agents

                # set home range to rl with probability (1- wl), and to 0.1 if lockdown is violated
                # TODO: parameterize this 0.1 value?
                new_home_range = CAMP_SIZE * (0.1 if will_violate[i] < wl else rl)

                # update home range data
                a.home_range = new_home_range
                self.agents_home_ranges[i] = new_home_range

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
                a.foodline_id = PersonHelper.find_nearest(a.household_center, self.foodlines)

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
        # setting different values of b, we can simulate remove-and-isolate interventions with different detection
        # efficiencies.
        if isolation is not None:
            self.P_detect = isolation['b']  # probability that camp manager detects agent with symptoms
            self.P_n = isolation['n']  # number of days after recovery when agent can go back to camp

            assert 0.0 <= self.P_detect <= 1.0, "Probability of detecting symptoms must be within [0,1]"
            assert self.P_n > 0, "Invalid value for isolation parameter: n"

    def isolate_household(self, household_id: int) -> None:
        # isolate household with given id
        # first, find the agents in the household
        hh_agents_idx = self.agents_households == household_id
        # set all agents in the household as quarantined
        self.agents_route[hh_agents_idx] = QUARANTINED
        # update each agent
        agents = self.schedule.agents
        for i in np.argwhere(hh_agents_idx):
            agents[i].isolate()

    def check_remove_from_isolation(self, household_id: int) -> None:
        # Remove agents in household with given id from isolation (quarantine) IF all people in household have recovered
        # TODO: How will agent recover? Is there special care given in isolation which can expedite agent's state to
        # TODO: recovered or people always recover after going through all stages
        # TODO: i.e. symptomatic->mild/severe->recovered?

        # TODO: Right now to remove agents from quarantine, we check if agent has shown no symptoms for some days.
        # TODO: Should the logic be instead: check if agent's state is SUSCEPTIBLE OR RECOVERED? This way, asymptomatic
        # TODO: and exposed agents will not be sent back to the camp

        # number of people in `household_id` household
        num_ppl_hh = np.count_nonzero(self.agents_households == household_id)

        # if all people in the household don't show symptoms for `P_n` days, then free them all
        if np.count_nonzero(np.logical_and(
            self.agents_households == household_id,  # check for agents in same household
            self.agents_disease_states != SYMPTOMATIC,  # agent must not be symptomatic
            self.agents_disease_states != MILD,  # agent must not be mildly infected
            self.agents_disease_states != SEVERE,  # agent must not be severely infected
            self.agents_day_counter >= self.P_n  # agent must've been in no-symptoms state for at least `P_n` days TODO
        )) == num_ppl_hh:
            # first, find the agents in the household
            hh_agents_idx = self.agents_households == household_id
            # set all agents in the household as quarantined
            self.agents_route[hh_agents_idx] = QUARANTINED
            # update each agent
            agents = self.schedule.agents
            for i in np.argwhere(hh_agents_idx):
                agents[i].goto_household(1.0)  # if quarantine is over, then go back to household
                agents[i].day_counter = 0  # restart day counter

    @property
    def people_count(self):
        # number of people in the camp
        return self.params.total_population

    def get_filter_array(self):
        # returns compatible `people` array which is then passed to `_filter_agents` function
        # Agent columns needed: route, household_id, disease state, ethnic group
        return np.dstack([
            self.agents_route,
            self.agents_households,
            self.agents_disease_states,
            self.agents_ethnic_groups
        ]).squeeze()

    def _assign_households_to_agents(self):
        # assign households to agents based on capacity
        # Iso-boxes are prefabricated housing units with a mean occupancy of 10 individuals
        # Tents have a mean occupancy of 4 individuals.

        # In Moria, the homes of people with the same ethnic or national background are spatially clustered, and people
        # interact more frequently with others from the same background as themselves. To simulate ethnicities or
        # nationalities in our camp, we assigned each household to one of eight “backgrounds” in proportion to the
        # self-reported national origins of people in the Moria medical records. For each of the eight simulated
        # backgrounds, we randomly selected one household to be the seed for the cluster. We assigned the x nearest
        # unassigned households to that background, where x is the number of households in the background. Thus, the
        # first background occupies an area that is roughly circular, but other backgrounds may occupy crescents or
        # less regular shapes.
        # NOTE: In our implementation, we assign agents to ethnicities instead of assigning households to ethnicities

        # check if provided population can be fit into given number of households
        camp_capacity = self.params.number_of_people_in_one_isobox * self.params.number_of_isoboxes + \
                        self.params.number_of_people_in_one_tent * self.params.number_of_tents
        assert camp_capacity >= self.people_count, \
            "Number of people ({}) exceeds camp capacity ({})".format(self.people_count, camp_capacity)

        # array containing household id for each agent. initialize all with -1
        out = np.zeros((self.people_count,), dtype=np.int32) - 1
        o = 0  # counter for `out`

        # get leftover capacity for each of the households
        household_left_capacities = self.households[:, 1].copy()

        # create clusters based on number of ethnic groups
        # use kmeans algorithm to cluster households
        # `cluster_pts` contains co-ordinates where clusters are centered. This may not be exactly a household position
        cluster_pts, _ = kmeans(self.households[:, 2:], len(self.ethnic_groups))

        # iterate for all ethnic groups available
        for i, eth in enumerate(self.ethnic_groups):
            # number of people in same ethnic group (not any one assigned to a household initially)
            num_eth_ppl = np.count_nonzero(self.agents_ethnic_groups == i)
            # cluster center co-ordinates
            cluster_center = cluster_pts[i, :]

            # while there are people to allocate to a household
            while num_eth_ppl > 0:
                # get nearest household to cluster center which has some capacity
                hh_idx, _ = PersonHelper.find_nearest(
                    cluster_center,
                    self.households[:, 2:],
                    household_left_capacities > 0  # return only households which have some capacity left
                )
                # check if such household exist
                if hh_idx == -1:
                    raise RuntimeError("Can't find any household for agents")

                # get the capacity of the selected household
                hh_cap = household_left_capacities[hh_idx]

                # get number of people who can fit into this household
                ppl_to_allocate = int(min(num_eth_ppl, hh_cap))

                # assign agents to household
                out[o: o + ppl_to_allocate] = hh_idx
                o = o + ppl_to_allocate

                # update household capacity
                household_left_capacities[hh_idx] -= ppl_to_allocate
                # update number of unassigned agents in the ethnic group
                num_eth_ppl -= ppl_to_allocate

        # return household ids of agents
        return out

    def _get_households(self) -> np.array:
        """
        Returns
        -------
            out: An 2D array (?, 4) containing id, capacity and x,y co-ordinates of the households
        """

        num_iso_boxes = self.params.number_of_isoboxes
        num_tents = self.params.number_of_tents

        # get positions, ids and capacities of iso-boxes
        iso_boxes_pos = self._get_iso_boxes(num_iso_boxes, self.params.area_covered_by_isoboxes)
        iso_boxes_ids = np.arange(0, num_iso_boxes)[:, None]  # expand from shape (?,) to (?,1)
        iso_capacities = np.array([self.params.number_of_people_in_one_isobox] * num_iso_boxes)[:, None]

        # get positions, ids and capacities of tents
        tents_pos = self._get_tents(num_tents, self.params.area_covered_by_isoboxes)
        tents_ids = np.arange(num_iso_boxes, num_iso_boxes + num_tents)[:, None]  # expand from shape (?,) to (?,1)
        tents_capacities = np.array([self.params.number_of_people_in_one_tent] * num_tents)[:, None]

        # join ids, capacities and co-ordinates of iso-boxes and tents
        iso_boxes = np.concatenate([iso_boxes_ids, iso_capacities, iso_boxes_pos], axis=1)
        tents = np.concatenate([tents_ids, tents_capacities, tents_pos], axis=1)

        # merge iso-boxes and tents
        households = np.concatenate([iso_boxes, tents], axis=0)
        np.random.shuffle(households)

        return households  # return household data

    @staticmethod
    def _get_iso_boxes(num: int, iso_area_ratio: float) -> np.array:
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
        center_sq_side = CAMP_SIZE * iso_area_ratio**0.5

        # minimum and maximum co-ordinates for central square
        p_min = (CAMP_SIZE - center_sq_side) / 2.0
        p_max = (CAMP_SIZE + center_sq_side) / 2.0

        pos = (p_max - p_min) * np.random.random(size=(num, 2)) + p_min  # choose random positions from central square

        return pos  # return iso boxes co-ordinates

    @staticmethod
    def _get_tents(num: int, iso_area_ratio: float) -> np.array:
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
        center_sq_side = CAMP_SIZE * iso_area_ratio ** 0.5

        min1 = 0.0  # minimum co-ordinate for the region outside central square
        max1 = (CAMP_SIZE - center_sq_side) / 2.0  # co-ordinate of first edge of central square
        min2 = (CAMP_SIZE + center_sq_side) / 2.0  # co-ordinate of second edge of central square
        max2 = CAMP_SIZE  # co-ordinate of camp end

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

    def _assign_ethnicity_to_agents(self):
        # assign ethnicity to agents of the camp based on `ethnic_groups` array

        # number of ethnic groups
        num_eth = len(self.ethnic_groups)

        assert self.people_count >= num_eth, "Minimum {} people required for calculations".format(num_eth)

        # array containing ethnic group ids
        out = np.zeros((self.people_count,), dtype=np.int32)
        o = 0  # counter for `out`

        for i, grp in enumerate(self.ethnic_groups):
            # calculate number of people in ethnic group from percentage
            grp_ppl_count = int(grp[1] * self.people_count)
            # assign calculated number of people to ethnic group `grp`
            out[o: o + grp_ppl_count] = i
            # increment counter
            o = o + grp_ppl_count

        # note that by default any small number of agents left from above loop (due to rounding off `grp_ppl_count` will
        # be assigned to group 0)

        # shuffle and return
        np.random.shuffle(out)
        return out
