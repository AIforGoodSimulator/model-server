import datetime
import pandas as pd
from tqdm import tqdm
from scipy.cluster.vq import kmeans

from ai4good.models.abm.np_impl.model import *
from ai4good.models.abm.np_impl.parameters import Parameters
from ai4good.models.abm.np_impl.utils import get_incubation_period

CAMP_SIZE = 100.0


class Moria(Camp):
    """
    Implementation of Moria camp for ABM.
    Methods defined:
    ----------------
    1. simulate()               : To start simulation. Internally calls day() to simulate each day
    2. day()                    : Simulate each day in Moria camp
    3. get_activities()         : Get the activities to be performed by each agent on each day
    4. stop_simulation()        : Get flag to check if simulation can be stopped
    5. intervention_transmission_reduction()
                                : To apply intervention to reduce transmission probability
    6. intervention_sectoring() : To apply intervention to add sectors in camp where each sector has its own food line
    7. intervention_lockdown()  : To apply intervention to update home ranges of agents
    8. intervention_isolation() : To apply intervention to isolate agents with symptoms by camp managers
    9. intervention_social_distancing()
                                : Not implemented yet! This will apply social distancing between agents
    10. detect_and_isolate()    : To simulate isolation intervention on each day (camp->isolation). Also referred as
                                  quarantine.
    11. check_and_deisolate()   : To simulate deisolation (isolation->camp)
    12. save_progress()         : To save the progress of the simulation in a .csv file
    13. _assign_households_to_agents()
                                : Assign households to each agent by clustering agents with same ethnicity
    14. _get_households()       : Get households information in the moria camp
    15. _get_iso_boxes()        : Get iso-boxes information in the moria camp
    16. _get_tents()            : Get tents information in the moria camp
    17. _assign_ethnicity_to_agents()
                                : Based on data defined in `ethnic_groups`, assign ethnicity to the agents of the camp
    18. _init_queue()           : Initialize queues (toilets or food line) in the camp
    """
    
    def __init__(self, params: Parameters, profile: str):
        super().__init__(params, CAMP_SIZE)

        self.P_detect = 0.0  # probability that camp manager detects agent with symptoms
        self.P_n = 0  # number of days after recovery when agent can go back to camp

        # This number is used to specify the amount of activities happening in the camp. More the activities, more the
        # interactions of agents in the camp
        # DONE: parameterize it => we are given **daily** probability so: toilet + fl + wandering + hh = 4
        self.num_activities = 10

        assert self.params.num_food_visit <= self.num_activities
        assert self.params.num_toilet_visit <= self.num_activities

        # number of days passed in simulation
        self.t = 0

        # get households in the moria camp
        self.households: np.array = self._get_households()
        # initialize agents array
        agents = np.empty((self.num_people, A_FEATURES))

        agents[:, A_AGE] = self.params.age_and_gender[:, 0]
        agents[:, A_GENDER] = self.params.age_and_gender[:, 1]

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
        agents[:, [A_X, A_Y]] = agents[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]].copy()

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
        self._init_queue("toilet", self.params.toilets_blocks[0])
        self._init_queue("food_line", self.params.foodline_blocks[0])

        logging.info("Shape of agents array: {}".format(agents.shape))

        # Name of the file to store progress
        self.tot_progress_file_name = "abm_moria_tot_{}_{}.csv".format(
            profile,
            datetime.datetime.strftime(datetime.datetime.now(), "%d%m%Y_%H%M")
        )
        self.age_progress_file_name = "abm_moria_age_{}_{}.csv".format(
            profile,
            datetime.datetime.strftime(datetime.datetime.now(), "%d%m%Y_%H%M")
        )
        # Initialize progress dataset
        self.data_collector_tot = pd.DataFrame({
            'DAY': [],
            'SUSCEPTIBLE': [], 'EXPOSED': [], 'PRESYMPTOMATIC': [], 'SYMPTOMATIC': [], 'MILD': [], 'SEVERE': [],
            'ASYMPTOMATIC1': [], 'ASYMPTOMATIC2': [], 'RECOVERED': [], 'DECEASED': [],
            'HOSPITALIZED': []
        })
        self.data_collector_age = pd.DataFrame({
            'DAY': [],
            'INF_AGE0-9': [], 'INF_AGE10-19': [], 'INF_AGE20-29': [], 'INF_AGE30-39': [], 'INF_AGE40-49': [],
            'INF_AGE50-59': [], 'INF_AGE60-69': [], 'INF_AGE70+': []
        })

        # Set initial intervention params (if any)
        # 1. Apply transmission reduction by scaling down the probability of infection spread
        self.intervention_transmission_reduction(self.params.transmission_reduction)
        # 2. Apply isolation parameters
        self.intervention_isolation(self.params.probability_spotting_symptoms_per_day, self.params.clear_day)
        # 3. Apply sectoring
        # NOTE: This is already done since initial food line queue initialization is done with `foodline_blocks` param
        # self.intervention_sectoring(self.params.foodline_blocks)
        # 4. Apply lockdown
        # self.intervention_lockdown(rl=?, vl=self.params.prop_violating_lockdown)

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

        # Save initialized progress to file
        self.data_collector_tot.to_csv(self.tot_progress_file_name, index=False)
        self.data_collector_age.to_csv(self.age_progress_file_name, index=False)

    def day(self):
        # Run 1 day of simulation in the moria camp

        # Get the instance of the day when food line will form
        s_food = int(self.num_activities / 3.0)  # food line is on 3 times a day in Moria

        # In each day, agents will perform number of activities. This number is denoted by `num_activities`.
        for s in range(self.num_activities):

            # Check if food line is opened at this time of the day. If it is, then set the probability value of agents
            # visiting the food line
            prob_food_line = 0.0
            if s != 0 and s % s_food == 0:
                # If for e.g. `num_activities` = 10, and food line opens 3 times a day, then value of `s_food` will be 3
                # That is, on time 4, 7, 10, the food line will be open
                prob_food_line = self.params.pct_food_visit if s % s_food == 0 else 0.0

            # Get the activities agents will do in this time step. Each element of `activities` is either one of the
            # ACTIVITY_* constants or -1 (when agent is inactive/deceased)
            activities = Moria.get_activities(self.agents, prob_food_line,
                                              self.params.num_toilet_visit/self.num_activities)

            # Get the agents who are currently in toilet or food line queues (i.e. from previous step `s-1` of the day)
            in_queue = (self.agents[:, A_ACTIVITY] == ACTIVITY_TOILET) | \
                       (self.agents[:, A_ACTIVITY] == ACTIVITY_FOOD_LINE)

            # Perform activities for current time step of the day

            # 1. Simulate agents wandering in the camp
            wanderer_ids = activities == ACTIVITY_WANDERING
            self.agents[wanderer_ids, :], new_wd_inf = Camp.simulate_wander(self.agents[wanderer_ids, :], CAMP_SIZE,
                                                                            self.params.relative_strength_of_interaction,
                                                                            self.params.infection_radius,
                                                                            self.prob_spread)

            # 2. Simulate agent's visit to toilet
            new_inf_t = self.simulate_queues(np.argwhere((activities == ACTIVITY_TOILET) & ~in_queue).reshape((-1,)),
                                             "toilet")

            # 3. Simulate agent's visit to food line
            new_inf_f = self.simulate_queues(np.argwhere((activities == ACTIVITY_FOOD_LINE) & ~in_queue).reshape((-1,)),
                                             "food_line")

            # 4. Simulate visit to respective household
            hh_ids = activities == ACTIVITY_HOUSEHOLD
            self.agents[hh_ids, :], new_hh_inf = Camp.simulate_households(self.agents[hh_ids, :], self.prob_spread)

            # 5. Update toilet and food line queues
            self.update_queues(self.params.percentage_of_toilet_queue_cleared_at_each_step)

            # logging.debug("{} new agents were exposed through interactions during wandering".format(new_wd_inf))
            # logging.debug("{} new agents were exposed through interactions in toilet queues".format(new_inf_t))
            # logging.debug("{} new agents were exposed through interactions in food line queues".format(new_inf_f))
            # logging.debug("{} new agents were exposed through household interactions".format(new_hh_inf))

        # Once for loop ends, all activities of the day have ended. At the end of the day, agents should go back to
        # their households. This includes agents in toilet/food line queue as well.
        # Dequeue all agents in the toilet/food line queues. Passing value of 1.0 will dequeue everyone from all queues.
        self.update_queues(1.0, dequeue_activity=ACTIVITY_HOUSEHOLD)
        # Get activities for all agents at the end of the day. Current implementation sends all agents back to their
        # households at the end of the day. TODO: uncomment after confirmation from Gaia and Vera
        # activities = Moria.get_activities(self.agents, 0.0, 0.0, force_activity=ACTIVITY_HOUSEHOLD)
        # hh_ids = activities == ACTIVITY_HOUSEHOLD
        # self.agents[hh_ids, :], _ = Camp.simulate_households(self.agents[hh_ids, :], self.prob_spread)

        # Increment day
        self.t += 1

        # Increase day counter to track number of days in current disease state
        self.agents[:, A_DAY_COUNTER] += 1

        # Disease progress at the end of the day
        self.agents = Camp.disease_progression(self.agents)

        # If P_detect value is present, then isolate/de-isolate agents
        if self.P_detect > SMALL_ERROR:
            # Camp managers can detect agents with symptoms with some probability and isolate them
            self.agents = Moria.detect_and_isolate(self.agents, self.P_detect)
            # If all agents of isolated household are not showing symptoms for some days, then send them back to camp
            self.agents = Moria.check_and_deisolate(self.agents, self.P_n)

    @staticmethod
    @nb.njit
    def get_activities(agents: np.array, prob_food_line: float, prob_toilet: float,
                       force_activity: int = -1) -> np.array:
        """
        Return the activities all agents will do at any point in time.
        This method gets called multiple times in a day depending on the amount of activities agents are doing in camp.

        Parameters
        ----------
        agents: A numpy array containing information of agents in the camp.
        prob_food_line: Probability that agent will go to food line queue. If food line is not opened at any given time,
            this will be 0.
        prob_toilet: Probability that agent will go to the toilet queue.
        force_activity: If value is not -1, then for all agents (who are not under isolation or hospitalized) the
            activity will be set to `force_activity`.

        Returns
        -------
        out: A numpy array containing activity id of all the agents. All ACTIVITY_* variables are used as possible
            values. If an agent has no activity at given time (maybe if agent has deceased), then value for that agent
            wil be -1.

        """

        n = agents.shape[0]  # number of agents
        out = np.zeros((n,), dtype=np.int32) - 1  # empty activities array

        # Iterate all agents
        for i in range(n):

            if agents[i, A_DISEASE] == INF_DECEASED:
                # Deceased agents are no longer processed
                continue

            # Check if agent is showing symptoms
            showing_symptoms = agents[i, A_DISEASE] in (INF_SYMPTOMATIC, INF_MILD, INF_SEVERE)

            # If agent is quarantined or hospitalized, then don't do anything
            if agents[i, A_ACTIVITY] == ACTIVITY_QUARANTINED or agents[i, A_ACTIVITY] == ACTIVITY_HOSPITALIZED:
                out[i] = agents[i, A_ACTIVITY]

            # Check for force activity
            elif force_activity != -1:
                out[i] = force_activity

            # If agent is not showing symptoms, go to toilet with some probability
            # An agent already in the toilet will remain there till the `update_queues` method dequeues it
            elif agents[i, A_ACTIVITY] == ACTIVITY_TOILET or \
                    (not showing_symptoms and random.random() <= prob_toilet):
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
            if disease_states[i] not in [INF_SUSCEPTIBLE, INF_RECOVERED, INF_DECEASED]:
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

        logging.info("INTERVENTION: After applying transmission reduction methods, new Pa={}".format(self.prob_spread))

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
        self._init_queue("food_line", sector_size)

        logging.info("INTERVENTION: Creating sectors in the camp of size ({}x{})".format(sector_size, sector_size))

    def intervention_lockdown(self, rl=None, vl=None):
        # Some countries have attempted to limit the spread of COVID-19 by requiring people to stay in or close to
        # their homes (ref). This intervention has been called "lockdown". We simulated a lockdown in which most
        # individuals are restricted to a home range with radius rl around their households. We assumed that a
        # proportion vl of the population will violate the lockdown. Thus, for each individual in the population,
        # we set their home range to rl with probability (1- vl), and to 0.1 otherwise. By manipulating rl and vl we
        # simulated lockdowns that are more or less restrictive and/or strictly enforced.

        # Parameters:
        #   rl: new home range (for all agents)
        #   vl: proportion of people who violate lockdown

        if rl is None or vl is None:
            return

        # check if agents will violate lockdown
        will_violate = np.random.random(size=(self.num_people,)) < vl

        # assign new home ranges
        self.agents[will_violate, A_HOME_RANGE] = 0.1 * CAMP_SIZE
        self.agents[~will_violate, A_HOME_RANGE] = rl * CAMP_SIZE

        logging.info("INTERVENTION: In lockdown, {} agents are violating home ranges".
                     format(np.count_nonzero(will_violate)))

    def intervention_isolation(self, b=None, n=None):
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

        if b is None or n is None:
            return

        assert 0.0 <= b <= 1.0, "Probability of detecting symptoms must be within [0,1]"
        assert n > 0, "Invalid value for isolation parameter: n"

        self.P_detect = b
        self.P_n = n

        logging.info("INTERVENTION: Camp managers can detect agents with symptoms with probability of {}".format(b))

    def intervention_social_distancing(self, degree):
        # DONE: Apply a repel force between each agent outside the household to simulate social distancing
        # Since there is no social distancing in Tucker's model no need to implement it
        pass

    @staticmethod
    @nb.njit
    def detect_and_isolate(agents: np.array, prob_detect: float) -> np.array:
        # Check if agent needs to be quarantined.
        # An agent in the camp who is showing symptoms can be quarantined with some probability.
        # The detected agent will be removed along with its household.

        n = agents.shape[0]  # number of agents in the camp

        for i in range(n):  # Iterate for each agent in the camp

            # If agent i is already quarantined, don't process
            if agents[i, A_ACTIVITY] == ACTIVITY_QUARANTINED:
                continue

            # An agent who is showing infection symptoms can be detected by camp manager with some probability
            i_detected = (agents[i, A_DISEASE] == INF_SYMPTOMATIC or agents[i, A_DISEASE] == INF_MILD or
                          agents[i, A_DISEASE] == INF_SEVERE) and random.random() <= prob_detect

            # if agent is not detected, skip
            if i_detected == 0:
                continue

            # when agent i is detected by camp managers, isolate everyone in agent i's household
            for j in range(n):

                # Skip if agent j DOES NOT share household with agent i
                # This is checked by calculating the distance between i's household and j's household, if the distance
                # -> 0 then we say agent i and j share household
                d = (agents[i, A_HOUSEHOLD_X] - agents[j, A_HOUSEHOLD_X]) ** 2 + \
                    (agents[i, A_HOUSEHOLD_Y] - agents[j, A_HOUSEHOLD_Y]) ** 2
                d = d ** 0.5
                if d > SMALL_ERROR:
                    continue

                # Quarantine agent j who shares household with detected agent i
                # Since agent i is also covered  in the j loop, no need to explicitly quarantine agent i
                agents[j, A_ACTIVITY] = ACTIVITY_QUARANTINED
                # When agent is quarantined, they will remain in their household
                agents[j, A_X] = agents[j, A_HOUSEHOLD_X]
                agents[j, A_Y] = agents[j, A_HOUSEHOLD_Y]

        return agents

    @staticmethod
    @nb.njit
    def check_and_deisolate(agents: np.array, n: int) -> np.array:
        """
        Check agents who are in isolation and return them back to the camp if no agent in their household is showing
        any symptoms for the past n days.
        From tucker model:
            "We assume that individuals are returned to the camp 7 days after they have recovered, or if they do not
            become infected, 7 days after the last infected person in their household has recovered"
        In our implementation, this "7 days" is parameterized in `P_n` (class level) or n (function level).
        """

        n = agents.shape[0]  # number of agents in the camp

        for i in range(n):  # Iterate for each agent in the camp

            # If agent i is not quarantined, don't process
            if agents[i, A_ACTIVITY] != ACTIVITY_QUARANTINED:
                continue

            # Storing agents who are sharing household with agent i.
            housemate_ids = []
            # Number of agents who are sharing household with agent i and are not showing symptoms for the past n days
            num_not_showing_sym = 0

            # For a quarantined agent i in the camp, check for his/her housemates (who would be also quarantined) and
            # check if they all can now go back to the camp or not.
            for j in range(n):

                # Skip if agent j DOES NOT share household with agent i
                # This is checked by calculating the distance between i's household and j's household, if the distance
                # -> 0 then we say agent i and j share household
                d = (agents[i, A_HOUSEHOLD_X] - agents[j, A_HOUSEHOLD_X]) ** 2 + \
                    (agents[i, A_HOUSEHOLD_Y] - agents[j, A_HOUSEHOLD_Y]) ** 2
                d = d ** 0.5
                if d > SMALL_ERROR:
                    continue

                housemate_ids.append(j)  # add agent j as housemate of agent i

                # Check if agent j (housemate of agent i) is not showing symptoms for the past `P_n` days
                if agents[j, A_DISEASE] not in (INF_SYMPTOMATIC, INF_MILD, INF_SEVERE) \
                        and agents[j, A_DAY_COUNTER] >= n:
                    num_not_showing_sym += 1

            # Skip if all any housemate is not Ok to be back in the camp. All of them should be not showing symptoms
            # for n days in order to be sent back to the camp.
            if len(housemate_ids) == 0 or len(housemate_ids) != num_not_showing_sym:
                continue

            # At this point we know that all housemates of agent i are without symptoms for past n days, so now send
            # them back to camp.
            # Update their activity to household so they can do other activities (like wandering, going to toilet and
            # food line, etc.) in the camp
            agents[np.array(housemate_ids), A_ACTIVITY] = ACTIVITY_HOUSEHOLD

        # return updated agents array
        return agents

    def save_progress(self):
        # Function to save the progress of the simulation at any time step into dataframe

        # Add latest day progress in the data set
        row = [
            self.t,
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_SUSCEPTIBLE),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_EXPOSED),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_PRESYMPTOMATIC),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_SYMPTOMATIC),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_MILD),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_SEVERE),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_ASYMPTOMATIC1),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_ASYMPTOMATIC2),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_RECOVERED),
            np.count_nonzero(self.agents[:, A_DISEASE] == INF_DECEASED),
            np.count_nonzero(self.agents[:, A_ACTIVITY] == ACTIVITY_HOSPITALIZED)
        ]
        self.data_collector_tot.loc[self.data_collector_tot.shape[0]] = row

        # count of infectious and alive agents (age-wise)
        infected_agents = OptimizedOps.is_infected(self.agents[:, A_DISEASE])
        row = [
            self.t,
            np.count_nonzero(infected_agents & ( self.agents[:, A_AGE] < 10 )),
            np.count_nonzero(infected_agents & ( (self.agents[:, A_AGE] >= 10) & (self.agents[:, A_AGE] < 20) )),
            np.count_nonzero(infected_agents & ( (self.agents[:, A_AGE] >= 20) & (self.agents[:, A_AGE] < 30) )),
            np.count_nonzero(infected_agents & ( (self.agents[:, A_AGE] >= 30) & (self.agents[:, A_AGE] < 40) )),
            np.count_nonzero(infected_agents & ( (self.agents[:, A_AGE] >= 40) & (self.agents[:, A_AGE] < 50) )),
            np.count_nonzero(infected_agents & ( (self.agents[:, A_AGE] >= 50) & (self.agents[:, A_AGE] < 60) )),
            np.count_nonzero(infected_agents & ( (self.agents[:, A_AGE] >= 60) & (self.agents[:, A_AGE] < 70) )),
            np.count_nonzero(infected_agents & ( self.agents[:, A_AGE] >= 70 ))
        ]
        self.data_collector_age.loc[self.data_collector_age.shape[0]] = row

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
        camp_capacity = self.params.number_of_people_in_isoboxes + self.params.number_of_people_in_tents
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
        cluster_pts, _ = kmeans(households[:, 2:], len(self.params.ethnic_groups))

        # iterate for all ethnic groups available
        for i, eth in enumerate(self.params.ethnic_groups):
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

        num_iso_boxes = int(np.ceil(
            self.params.number_of_people_in_isoboxes / self.params.number_of_people_in_one_isobox
        ))
        num_tents = int(np.ceil(self.params.number_of_people_in_tents / self.params.number_of_people_in_one_tent))

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
        """
        Assign ethnicity to agents of the camp based on `ethnic_groups` list.
        Ethnic groups are defined in `ethnic_groups` list. Agents in the camp are assigned ethnicity based on the
        proportion of the ethnic population. For example, if 0.1 proportion of the population is Afghan, then ~`N`/10
        agents will be assigned Afghan ethnicity where `N` is the total number of agents in the camp.

        This is slight variation from tucker model. Originally, each household was assigned an ethnicity and households
        with similar ethnicity were clustered.
        Now, ethnicity is assigned for each agent (based on given data) and agents will form spatial clusters in the
        camp during `_assign_households_to_agents` function call.
        """

        # number of ethnic groups
        num_eth = len(self.params.ethnic_groups)

        assert self.num_people >= num_eth, "Minimum {} people required for calculations".format(num_eth)

        # array containing ethnic group ids
        out = np.zeros((self.num_people,), dtype=np.int32)
        o = 0  # counter for `out`

        for i, grp in enumerate(self.params.ethnic_groups):
            # calculate number of people in ethnic group from proportion
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

    def _init_queue(self, queue_name, grid_size) -> None:
        """
        Initialize a queue (toilet or food line).
        Steps for initialization:
            1. Uniformly position queues throughout the camp
            2. Mark all queues as empty in the beginning i.e. no person is standing/waiting in the line
            3. Find the queue nearest to each agent's household and assign it to him/her. The agent will always use
                the assigned queue.

        Parameters
        ----------
        queue_name: Name of the queue to initialize. Possible values are "toilet" and "food_line"
        grid_size: Number of grids for uniform distribution of queues throughout the camp

        """

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
