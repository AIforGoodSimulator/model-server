import pandas as pd
from tqdm import tqdm
from scipy.cluster.vq import kmeans

from ai4good.models.abm.np_impl.model import *
from ai4good.models.abm.initialise_parameters import Parameters
from ai4good.models.abm.mesa_impl.utils import read_age_gender, get_incubation_period

CAMP_SIZE = 100.0


class Moria(Camp):
    
    # ethnic groups and their proportions in the camp
    # TODO: tucker model refers to 8 ethnic groups. But abm.py contained only 7. verify this.
    # In abm.py, people count per ethnic group was mentioned. We have transformed it into proportions to work for any 
    # population size
    ethnic_groups = [  # [ethnic group name, proportion of people in ethnic group]
        ['afghan', 7919 / 10135],
        ['cameroon', 149 / 10135],
        ['congo', 706 / 10135],
        ['iran', 107 / 10135],
        ['iraq', 83 / 10135],
        ['somalia', 442 / 10135],
        ['syria', 729 / 10135]
    ]
    
    def __init__(self, params: Parameters):
        super().__init__(params, CAMP_SIZE)

        self.P_detect = 0.0  # probability that camp manager detects agent with symptoms
        self.P_n = 0  # number of days after recovery when agent can go back to camp

        # This number is used to specify the amount of activities happening in the camp. More the activities, more the
        # interactions of agents in the camp
        # TODO: parameterize it
        self.num_activities = 10

        self.t = 0

        # get households in the moria camp
        self.households: np.array = self._get_households()
        # initialize agents array
        agents = np.empty((self.num_people, A_FEATURES))

        # get age and gender of the agents
        agents[:, A_AGE] = read_age_gender(self.num_people)[:, 0]
        agents[:, A_GENDER] = read_age_gender(self.num_people)[:, 1]

        # initialize all agents as susceptible initially
        agents[:, A_DISEASE] = np.array([INF_SUSCEPTIBLE] * self.num_people)

        # get incubation period
        agents[:, A_INCUBATION_PERIOD] = get_incubation_period(self.num_people)
        agents[:, A_DAY_COUNTER] = 0

        # get home ranges of each agent
        agents[:, A_HOME_RANGE] = np.array([
            self.params.smaller_movement_radius * CAMP_SIZE if (agents[i, A_GENDER] == FEMALE or agents[i, A_AGE] < 10)
            else self.params.larger_movement_radius * CAMP_SIZE
            for i in range(self.num_people)
        ])

        # get ethnicities of the agents
        agents[:, A_ETHNICITY] = self._assign_ethnicity_to_agents()
        # assign households to the agents
        agents_households: np.array = self._assign_households_to_agents(self.households, agents[:, A_ETHNICITY])
        agents[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]] = self.households[agents_households, 2:]

        # initially, everyone's inside their households
        agents[:, A_ACTIVITY] = ACTIVITY_HOUSEHOLD

        # calculate which agents are asymptomatic
        is_asymptomatic = (agents[:, A_AGE] < 16.0) | \
                          (np.random.random((self.num_people,)) <= self.params.permanently_asymptomatic_cases)
        agents[:, A_IS_ASYMPTOMATIC] = is_asymptomatic

        # finally, randomly select one person (not asymptomatic) and mark as "Exposed"
        not_asymptomatic = np.argwhere(agents[:, A_IS_ASYMPTOMATIC] == 0).squeeze()
        agents[np.random.choice(not_asymptomatic), A_DISEASE] = INF_EXPOSED

        self.set_agents(agents)

        # initialize toilet and food line queues
        self._init_queues("toilet", self.params.toilets_blocks[0])
        self._init_queues("food_line", self.params.foodline_blocks[0])

        logging.info("Agents: {}".format(agents.shape))

        self.data_collector = pd.DataFrame({
            'DAY': [],
            'SUSCEPTIBLE': [], 'EXPOSED': [], 'PRESYMPTOMATIC': [], 'SYMPTOMATIC': [], 'MILD': [], 'SEVERE': [],
            'ASYMPTOMATIC1': [], 'ASYMPTOMATIC2': [], 'RECOVERED': [],
            'HOSPITALIZED': []
        })
        self.data_collector.to_csv("abm_progress.csv", index=False)

    def simulate(self):
        # Run simulation on Moria camp
        for _ in tqdm(range(self.params.number_of_steps)):

            # check if simulation has concluded
            if Moria.stop_simulation(self.agents[:, A_DISEASE]) == 1:
                break

            # simulate one day
            self.day()

            # save the progress
            self.save_progress()

    def day(self):
        # Run 1 day of simulation in the moria camp

        s_food = int(self.num_activities / (3.0+1.0))  # food line is on 3 times a day, hence split day in 4 equal parts

        # In each day, agents will perform number of activities. This number is denoted by `num_activities`.
        for s in range(self.num_activities):

            prob_food_line = 0.0
            if s % s_food == 0:
                prob_food_line = self.params.pct_food_visit if s % s_food == 0 else 0.0

            # get the activities asymptomatic agents will do in this time step
            activities = Moria.get_activities(self.agents, prob_food_line)

            in_queue = (self.agents[:, A_ACTIVITY] == ACTIVITY_TOILET) | \
                       (self.agents[:, A_ACTIVITY] == ACTIVITY_FOOD_LINE)

            # perform simulations

            # people wandering in the camp
            self.simulate_wander(activities == ACTIVITY_WANDERING)

            # going to toilet
            self.simulate_queues(np.argwhere((activities == ACTIVITY_TOILET) & ~in_queue).reshape((-1,)), "toilet")

            # going to food line
            self.simulate_queues(np.argwhere((activities == ACTIVITY_FOOD_LINE) & ~in_queue).reshape((-1,)),
                                 "food_line")

            # going to households
            self.simulate_households((activities == ACTIVITY_HOUSEHOLD) | (activities == ACTIVITY_QUARANTINED))

            # updating toilet and food line queues
            self.update_queues()

        # increment timer
        self.t += 1

        # increase day counter to track number of days in a disease state
        # not_fine = (self.agents[:, A_DISEASE] != INF_SUSCEPTIBLE) & (self.agents[:, A_DISEASE] != INF_RECOVERED)
        self.agents[:, A_DAY_COUNTER] += 1

        # remove all agents from the queue at the end of the day
        for t in self.toilet_queue:
            self.toilet_queue[t] = []
        for f in self.food_line_queue:
            self.food_line_queue[f] = []
        # Change activity route of agents as well
        in_queue = (self.agents[:, A_ACTIVITY] == ACTIVITY_TOILET) | \
                   (self.agents[:, A_ACTIVITY] == ACTIVITY_FOOD_LINE)
        self.agents[in_queue, A_ACTIVITY] = ACTIVITY_HOUSEHOLD

        # Disease progress at the end of the day
        self.agents = Camp.disease_progression(self.agents)

        if self.P_detect > SMALL_ERROR:
            # Camp managers can detect agents with symptoms with some probability and isolate them
            self.detect_and_isolate()
            # If all agents of isolated household are not showing symptoms for some days, then send them back to camp
            self.check_and_deisolate()

        # logs: number of agents in each disease state
        logging.debug("{}. SUS={}, EXP={}, PRE={}, SYM={}, MIL={}, SEV={}, AS1={}, AS2={}, REC={}".format(
            self.t,
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_SUSCEPTIBLE),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_EXPOSED),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_PRESYMPTOMATIC),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_SYMPTOMATIC),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_MILD),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_SEVERE),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_ASYMPTOMATIC1),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_ASYMPTOMATIC2),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_RECOVERED)
        ))

        # logs: number of agents in each activity zone
        logging.debug("{}. HSH={}, TLT={}, FDL={}, WDR={}, QRT={}, HSP={}".format(
            self.t,
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_HOUSEHOLD),
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_TOILET),
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_FOOD_LINE),
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_WANDERING),
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_QUARANTINED),
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_HOSPITALIZED)
        ))

    @staticmethod
    @nb.njit
    def get_activities(agents: np.array, prob_food_line) -> np.array:
        # Return the activities all agents will do at any point in time.
        # This method gets called multiple times in a day depending on the around of activities agents are doing in camp

        n = agents.shape[0]  # number of agents
        out = np.zeros((n,), dtype=np.int32) - 1  # empty activities array

        # Iterate all agents
        for i in range(n):

            # Check if agent is showing symptoms
            showing_symptoms = agents[i, A_DISEASE] in (INF_SYMPTOMATIC, INF_MILD, INF_SEVERE)

            # If agent is quarantined or hospitalized, then don't do anything
            if agents[i, A_ACTIVITY] == ACTIVITY_QUARANTINED or agents[i, A_ACTIVITY] == ACTIVITY_HOSPITALIZED:
                out[i] = agents[i, A_ACTIVITY]

            # If agent is not showing symptoms, go to toilet with some probability
            # An agent already in the toilet will remain there till the `update_queues` method dequeues it
            elif agents[i, A_ACTIVITY] == ACTIVITY_TOILET or \
                    (not showing_symptoms and random.random() <= 0.3):
                out[i] = ACTIVITY_TOILET
            # Same logic in food line
            elif agents[i, A_ACTIVITY] == ACTIVITY_FOOD_LINE or \
                    (not showing_symptoms and random.random() <= prob_food_line):
                out[i] = ACTIVITY_FOOD_LINE

            # if agent not showing symptoms, he/she will wander with 50% chance
            elif random.random() <= 0.5 and not showing_symptoms:
                out[i] = ACTIVITY_WANDERING
            # other 50% chance is that they will go to / remain in the household
            else:
                out[i] = ACTIVITY_HOUSEHOLD

        # return agents activities for the time step
        return out

    @staticmethod
    @nb.njit
    def stop_simulation(disease_states) -> int:
        # We ran each simulation until all individuals in the population were either susceptible or recovered, at which
        # point the epidemic had ended
        n = disease_states.shape[0]
        for i in range(n):
            if disease_states[i] not in [INF_SUSCEPTIBLE, INF_RECOVERED]:
                # DO NOT stop the simulation if any person is NOT (susceptible or recovered)
                return 0

        # if all agents are either susceptible or recovered, time to stop the simulation
        return 1

    def intervention_transmission_reduction(self, vt: float):
        # In Moria, there is approximately one tap per 42 people, so frequent hand washing (e.g., greater than 10x per
        # day, as in Jefferson et al. 2009) may be impossible. Due to the high population density in Moria
        # (~20,000 people km-2), maintaining safe distances among people may also be difficult or impossible.
        # However, people in Moria have been provided with face masks. We simulated a population in which all
        # individuals wear face masks outside their homes by setting vt = 0.32 (Jefferson et al. 2009)

        # scale transmission probability
        self.prob_spread = self.prob_spread * vt

    def intervention_sectoring(self, sector_size):
        # The camp in our baseline model has a single food line, where transmission can potentially occur between two
        # individuals from any parts of the camp. This facilitates the rapid spread of COVID-19 infection. A plausible
        # intervention would be to divide the camp into sectors with separate food lines, and require individuals to
        # use the food line closest to their households. To simulate this intervention, we divide the camp into an
        # n x n grid of squares, each with its own food line

        # empty current food lines
        in_food_line = self.agents[:, A_ACTIVITY] == ACTIVITY_FOOD_LINE
        self.agents[in_food_line, A_ACTIVITY] = ACTIVITY_HOUSEHOLD  # send people back to their households

        # initialize food lines based on `sector` parameter
        self._init_queues("food_line", sector_size)

    def intervention_lockdown(self, rl, wl):
        # Some countries have attempted to limit the spread of COVID-19 by requiring people to stay in or close to
        # their homes (ref). This intervention has been called "lockdown". We simulated a lockdown in which most
        # individuals are restricted to a home range with radius rl around their households. We assumed that a
        # proportion wl of the population will violate the lockdown. Thus, for each individual in the population,
        # we set their home range to rl with probability (1- wl), and to 0.1 otherwise. By manipulating rl and wl we
        # simulated lockdowns that are more or less restrictive and/or strictly enforced.

        # Parameters:
        #   rl: new home range (for all agents)
        #   wl: proportion of people who violate lockdown

        # check if agents will violate lockdown
        will_violate = np.random.random(size=(self.num_people,)) < wl

        # assign new home ranges
        self.agents[will_violate, A_HOME_RANGE] = 0.1 * CAMP_SIZE
        self.agents[~will_violate, A_HOME_RANGE] = rl * CAMP_SIZE

    def intervention_isolation(self, b, n=7):
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

        assert 0.0 <= b <= 1.0, "Probability of detecting symptoms must be within [0,1]"
        assert n > 0, "Invalid value for isolation parameter: n"

        self.P_detect = b
        self.P_n = n

    def detect_and_isolate(self):
        # Check if agent needs to be quarantined
        # An agent in the camp who is showing symptoms can be quarantined with some probability
        # The detected agent will be removed along with its household

        # create filter
        detected_agent_filter = (self.agents[:, A_ACTIVITY] != ACTIVITY_QUARANTINED) & \
                                (OptimizedOps.showing_symptoms(self.agents[:, A_DISEASE])) & \
                                (np.random.random((self.num_people,)) <= self.P_detect)

        # get households shared by the agents. TODO: can be cached?
        # get household position of all agents
        hh = self.agents[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]]
        # get boolean matrix indicating households sharing by the agents
        # `sharing_hh[i, j]` = 1  when agent `i` and `j` share household, else 0
        sharing_hh = OptimizedOps.distance_matrix(hh) <= SMALL_ERROR

        # Get all agents to be isolated
        # Take all the households and filter out the ones where a symptomatic agent is detected by the camp manager.
        # Each element `sharing_hh[i, j]` will be 1 if agent `i` is detected by the camp manager and agent `j` is
        # sharing the household with `i` hence will be isolated too
        sharing_hh = sharing_hh * detected_agent_filter.reshape((-1, 1))

        isolate_agent_filter = np.sum(sharing_hh, axis=0) > 0.0
        self.agents[isolate_agent_filter, A_ACTIVITY] = ACTIVITY_QUARANTINED

    def check_and_deisolate(self):
        """
        Check agents who are in isolation and return them back to the camp if no agent in their household is showing
        any symptoms for the past `P_n` days.
        From tucker model:
            "We assume that individuals are returned to the camp 7 days after they have recovered, or if they do not
            become infected, 7 days after the last infected person in their household has recovered"
        In our implementation, this "7 days" is parameterized in `P_n`
        """

        # Filter agents who are in quarantine and not showing symptoms0 at least past `P_n` days
        to_deisolate = (self.agents[:, A_ACTIVITY] == ACTIVITY_QUARANTINED) & \
                       ~OptimizedOps.showing_symptoms(self.agents[:, A_DISEASE]) & \
                       (self.agents[:, A_DAY_COUNTER] >= self.P_n)

        # Check for each pair of agents i and j, if both i and j can be removed from isolation and sent back to camp
        # `to_deisolate_ij[i, j]` = 1 when agents `i` and `j` are both Ok to be removed from isolation
        to_deisolate_ij = to_deisolate.reshape((-1, 1)) & to_deisolate.reshape((1, -1))

        # get household position of all agents
        hh = self.agents[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]]
        # get boolean matrix indicating households sharing by the agents
        # `sharing_hh[i, j]` = 1  when agents `i` and `j` share household, else 0
        sharing_hh = OptimizedOps.distance_matrix(hh) <= SMALL_ERROR

        # number of agents in same household for each agent
        num_hh_mates = np.sum(sharing_hh, axis=1)
        # number of household mates that are Ok to sent back to camp
        num_hh_mates_to_deisolate = np.sum(to_deisolate_ij & sharing_hh, axis=1)

        # update agents activity route who are removed from isolation
        self.agents[num_hh_mates == num_hh_mates_to_deisolate, A_ACTIVITY] = ACTIVITY_HOUSEHOLD

    def save_progress(self):
        self.data_collector = pd.read_csv("abm_progress.csv")
        row = [self.t,
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_SUSCEPTIBLE),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_EXPOSED),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_PRESYMPTOMATIC),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_SYMPTOMATIC),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_MILD),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_SEVERE),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_ASYMPTOMATIC1),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_ASYMPTOMATIC2),
               np.count_nonzero(self.agents[:, A_DISEASE] == INF_RECOVERED),
               np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_HOSPITALIZED)
               ]
        self.data_collector.loc[self.data_collector.shape[0]] = row
        self.data_collector.to_csv("abm_progress.csv", index=False)

    def _assign_households_to_agents(self, households: np.array, agents_ethnic_groups: np.array) -> np.array:
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
        assert camp_capacity >= self.num_people, \
            "Number of people ({}) exceeds camp capacity ({})".format(self.num_people, camp_capacity)

        # array containing household id for each agent. initialize all with -1
        out = np.zeros((self.num_people,), dtype=np.int32) - 1
        o = 0  # counter for `out`

        # get leftover capacity for each of the households
        household_left_capacities = households[:, 1].copy()

        # create clusters based on number of ethnic groups
        # use kmeans algorithm to cluster households
        # `cluster_pts` contains co-ordinates where clusters are centered. This may not be exactly a household position
        cluster_pts, _ = kmeans(households[:, 2:], len(self.ethnic_groups))

        # iterate for all ethnic groups available
        for i, eth in enumerate(self.ethnic_groups):
            # number of people in same ethnic group (not any one assigned to a household initially)
            num_eth_ppl = np.count_nonzero(agents_ethnic_groups == i)
            # cluster center co-ordinates
            cluster_center = cluster_pts[i, :]

            # while there are people to allocate to a household
            while num_eth_ppl > 0:
                # get nearest household to cluster center which has some capacity
                hh_idx, _ = OptimizedOps.find_nearest(
                    cluster_center,
                    households[:, 2:],
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

        assert self.num_people >= num_eth, "Minimum {} people required for calculations".format(num_eth)

        # array containing ethnic group ids
        out = np.zeros((self.num_people,), dtype=np.int32)
        o = 0  # counter for `out`

        for i, grp in enumerate(self.ethnic_groups):
            # calculate number of people in ethnic group from percentage
            grp_ppl_count = int(grp[1] * self.num_people)
            # assign calculated number of people to ethnic group `grp`
            out[o: o + grp_ppl_count] = i
            # increment counter
            o = o + grp_ppl_count

        # note that by default any small number of agents left from above loop (due to rounding off `grp_ppl_count` will
        # be assigned to group 0)

        # shuffle and return
        np.random.shuffle(out)
        return out

    def _init_queues(self, queue_name, grid_size):
        # initialize queues

        if queue_name == "toilet":
            # add toilets to the camp
            self.toilets: np.array = OptimizedOps.position_blocks(grid_size, CAMP_SIZE)

            # initialize queues of each toilet in the camp
            for i in range(grid_size*grid_size):
                self.toilet_queue[i] = []

            # assign each agent with the toilet nearest to his/her household
            for i in range(self.num_people):
                # toilet nearest to agent's household
                t_id, _ = OptimizedOps.find_nearest(self.agents[i, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]], self.toilets)
                self.agents[i, A_TOILET] = t_id

        if queue_name == "food_line":
            # add food lines to the camp
            self.food_lines: np.array = OptimizedOps.position_blocks(grid_size, CAMP_SIZE)

            # initialize queues of each food line in the camp
            for i in range(grid_size * grid_size):
                self.food_line_queue[i] = []

            # assign each agent with the food line nearest to his/her household
            for i in range(self.num_people):
                # food line nearest to agent's household
                f_id, _ = OptimizedOps.find_nearest(self.agents[i, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]], self.food_lines)
                self.agents[i, A_FOOD_LINE] = f_id
