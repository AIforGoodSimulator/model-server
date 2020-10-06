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
from ai4good.models.abm.mesa.helper import CampHelper


class Camp(Model, CampHelper):
    """
    Modelling Moria camp
    """

    # Side length of square shaped camp
    CAMP_SIZE = 100.0

    # If a susceptible and an infectious individual interact, then the infection is transmitted with probability pa
    # Fang and colleagues (2020)
    # TODO: missing in params?
    Pa = 0.1

    def __init__(self, params: Parameters):
        super().__init__()
        self.params = params

        assert self.params.camp

        self.agents_disease_states = np.array([DiseaseStage.SUSCEPTIBLE for _ in range(self.people_count)])
        self.agents_age = read_age_gender(self.people_count)[:, 0]
        self.agents_gender = read_age_gender(self.people_count)[:, 1]

        # TODO: these are baseline values, parameterize it
        self.agents_home_ranges = np.array([
            0.02 * Camp.CAMP_SIZE if (self.agents_gender[i] == 0 or self.agents_age[i] < 10) else 0.1 * Camp.CAMP_SIZE
            for i in range(self.people_count)
        ])

        self.agents_pos = Camp.CAMP_SIZE * np.random.random((self.people_count, 2))
        self.agents_route = np.array([Route.HOUSEHOLD] * self.people_count)
        self.households = self.get_households(self.params.number_of_isoboxes, self.params.number_of_tents)
        self.agents_households = self.assign_households_to_agents()  # TODO: household ids of the agents
        self.agents_ethnic_groups = np.array([])  # TODO: can we remove/de-prioritize it for abm?

        # There are 144 toilets evenly distributed throughout the camp. Toilets are placed at the centres of the
        # squares that form a 12 x 12 grid covering the camp
        self.toilets = self._position_blocks(Camp.CAMP_SIZE, 12)
        self.toilets_queue = {}  # dict containing toilet_id: [ list of agents unique ids ] for toilet occupancy

        # The camp has one food line. Since the position of food line is not modelled, we just maintain food line queue
        # each person going to the food line to collect food will enter this queue
        self.foodline_queue = []  # start with no people in food line

        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(x_max=Camp.CAMP_SIZE, y_max=Camp.CAMP_SIZE, torus=False)

        # randomly select one person and mark as "Exposed"
        self.agents_disease_states[np.random.randint(0, high=self.people_count)] = DiseaseStage.EXPOSED

        for i in range(self.people_count):
            p = Person(i, self)
            self.schedule.add(p)
            self.space.place_agent(p, self.agents_pos[i, :])

    def step(self, times=10):
        # simulate model `times` number of steps

        for _ in range(times):

            # check if simulation can stop
            if self.stop_simulation(self.agents_disease_states):
                return

            # step all agents
            self.schedule.step()

            # clear some model day-wise variables
            self.foodline_queue = []  # clear food line queue at end of the day
            self.toilets_queue = {}  # clear toilet queues at end of the day

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

    def infection_spread_movement(self):
        # probability of infection spread during movement

        # get position of households
        hh_pos = self.households[:, 1:]

        # create an array of people's attributes
        # 0: household id, 1: home range, 2: ethnic group id, 3: disease state
        people = np.concatenate([
            self.agents_households,
            self.agents_home_ranges,
            self.agents_ethnic_groups,
            self.agents_disease_states
        ], axis=0)

        return self._prob_m(hh_pos, people)

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
        # TODO
        return np.array([])

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

        iso_boxes_pos = self.get_iso_boxes(num_iso_boxes)
        iso_boxes_ids = np.arange(0, num_iso_boxes)

        tents_pos = self.get_tents(num_tents)
        tents_ids = np.arange(num_iso_boxes, num_iso_boxes + num_tents)

        iso_boxes = np.concatenate([iso_boxes_ids, iso_boxes_pos], axis=0)
        tents = np.concatenate([tents_ids, tents_pos], axis=0)

        households = np.concatenate([iso_boxes, tents], axis=1)
        np.random.shuffle(households)
        return households

    @staticmethod
    def get_iso_boxes(num):
        """
        Get positions of the iso-boxes in the camp.

        Parameters
        ----------
        num: Number of iso-boxes in the camp

        """

        # Iso-boxes are assigned to random locations in a central square that covers one half of the area of the camp
        p_min = Camp.CAMP_SIZE * ((2**0.5)-1)/(2*(2**0.5))
        p_max = Camp.CAMP_SIZE * ((2**0.5)+1)/(2*(2**0.5))

        pos = (p_max - p_min) * np.random.random(size=(num, 2)) + p_min
        return pos

    @staticmethod
    def get_tents(num):
        """
        Get positions of the tents in the camp.

        Parameters
        ----------
        num: Number of tents in the camp

        """

        # Tents are assigned to random locations in the camp outside of the central square
        # The area outside the central square can be divided into 4 parts (bottom, right, top, left)
        # Below is the positions of tents distributed in all these 4 parts

        min1 = 0.0
        max1 = Camp.CAMP_SIZE * ((2**0.5)-1)/(2*(2**0.5))
        min2 = Camp.CAMP_SIZE * ((2**0.5)+1)/(2*(2**0.5))
        max2 = Camp.CAMP_SIZE

        assert num >= 4

        bottom_num = int(num / 4)
        bottom = np.dstack((
            (max2 - min1) * np.random.random((bottom_num,)) + min1,  # X co-ordinate
            (max1 - min1) * np.random.random((bottom_num,)) + min1  # Y co-ordinate
        ))

        right_num = int(num / 4)
        right = np.dstack((
            (max2 - min2) * np.random.random((right_num,)) + min2,  # X co-ordinate
            (min2 - max1) * np.random.random((right_num,)) + max1  # Y co-ordinate
        ))

        top_num = int(num / 4)
        top = np.dstack((
            (max2 - min1) * np.random.random((top_num,)) + min1,  # X co-ordinate
            (max2 - min2) * np.random.random((top_num,)) + min2  # Y co-ordinate
        ))

        left_num = num - (bottom_num + right_num + top_num)
        left = np.dstack((
            (max1 - min1) * np.random.random((left_num,)) + min1,  # X co-ordinate
            (min2 - max1) * np.random.random((left_num,)) + max1  # Y co-ordinate
        ))

        pos = np.concatenate([bottom, right, top, left], axis=1).squeeze()
        np.random.shuffle(pos)

        return pos