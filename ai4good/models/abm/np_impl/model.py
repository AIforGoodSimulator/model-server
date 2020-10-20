import random
import logging
import numpy as np
import numba as nb

from ai4good.models.abm.np_impl.parameters import Parameters


# very small float number to account for floating precision loss
SMALL_ERROR = 0.0000001

FEMALE = 0
MALE = 1

# Individuals remain in the symptomatic or 1st asymptomatic states for 5 days, and are infectious during this period.
# This period of "5 days" as defined by Tucker Gilman is parameterized in `SYMP_PERIOD`
SYMP_PERIOD = 5

# Disease states of the agents
INF_SUSCEPTIBLE = 0  # Agent is Ok
INF_EXPOSED = 1  # Agent got in contact with another infected agent and now has virus inside them
INF_PRESYMPTOMATIC = 2  # Agent is infected but is not showing symptoms yet
INF_SYMPTOMATIC = 3  # Agent is infected and showing early symptoms
INF_MILD = 4  # Agent is infected and showing mild severity
INF_SEVERE = 5  # Agent is infected and has severe condition
INF_ASYMPTOMATIC1 = 6  # Agent is infected but not showing symptoms (first phase)
INF_ASYMPTOMATIC2 = 7  # Agent is infected but not showing symptoms (second phase)
INF_RECOVERED = 8  # Agent has recovered or died due to infection. Recovered agents will not contract infection again
INF_DECEASED = 9  # Agent has passed away due to infection severity

# Activity routes for the agents
ACTIVITY_HOUSEHOLD = 0  # Agent is inside their household
ACTIVITY_WANDERING = 1  # Agent is wandering outside their household
ACTIVITY_TOILET = 2  # Agent is in the toilet queue
ACTIVITY_FOOD_LINE = 3  # Agent is in the food line queue
ACTIVITY_QUARANTINED = 4  # Agent is under quarantine
ACTIVITY_HOSPITALIZED = 5   # Agent is hospitalized


# Number of features of each agent
A_FEATURES = 15
# Features of each agent
A_X = 0  # x co-ordinate at any given point in time
A_Y = 1  # y co-ordinate at any given point in time
A_AGE = 2  # Age of the agent
A_GENDER = 3  # Gender of the agent
A_DISEASE = 4  # Disease state of the agent
A_INCUBATION_PERIOD = 5  # Incubation period of the agent
A_HOME_RANGE = 6  # Home range of the agent around their household
A_ETHNICITY = 7  # Ethnicity of the agent
A_HOUSEHOLD_X = 8  # x co-ordinate of the household where agent lives
A_HOUSEHOLD_Y = 9  # y co-ordinate of the household where agent lives
A_TOILET = 10  # Id of the toilet closest to the agent's household
A_FOOD_LINE = 11  # Id of the food line closest to the agent's household
A_ACTIVITY = 12  # Current activity route of the agent such as wandering, inside household, in toilet, etc.
# Number of days in the current disease state. After reaching a new disease state, this counter is reset to 0.
# Exception: counter does not reset when disease state reaches `INF_EXPOSED` in order to track incubation period.
A_DAY_COUNTER = 13
# Flag 0/1 to check if agent is asymptomatic
# All children under the age of 16 become asymptomatic (ref), and others become asymptomatic with probability 0.178
# (Mizumoto et al. 2020)
A_IS_ASYMPTOMATIC = 14


class OptimizedOps(object):
    """
    Helper class with numba optimized static methods
    """

    def __init__(self):
        pass

    @staticmethod
    @nb.njit
    def distance_matrix(pos: np.array) -> np.array:
        """
        Calculate and returns the distance matrix given (x,y) positions. For `n` positions, distance matrix returns a
        `n`x`n` matrix where element `i`,`j` gives the Euclidean distance between position `i` and position `j`.
        Parameters
        ----------
        pos: Position matrix of size n x 2 containing (x, y) co-ordinates

        Returns
        -------
        out: Distance matrix of size n x n

        """
        n = pos.shape[0]  # number of positions
        mat = np.zeros(shape=(n, n), dtype=np.float32)  # initialize distance matrix

        # loop through all pairs
        for i in range(n):
            for j in range(n):
                # calculate Euclidean distance between (i) and (j)
                dij = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2
                dij = dij ** 0.5
                # store result in distance matrix
                mat[i, j] = dij
        # return distance matrix
        return mat

    @staticmethod
    @nb.njit
    def position_blocks(grid_size: int, camp_size: float) -> np.array:
        """
        Uniform placement of blocks (typically food line or toilet) in the camp.

        Parameters
        ----------
            grid_size: Size of the square grid where placement of food line/toilet happens
            camp_size: Side length of the square sized camp

        Returns
        -------
            out: A (grid_size * grid_size, 2) shaped array containing (x, y) co-ordinates of the blocks

        """

        # since the placement will be uniform and equidistant, there will be a fixed distance between two blocks along
        # an axis. We call this distance as step
        step = camp_size / grid_size

        # bottom left position of the first block. This serves as both the x and y co-ordinate since camp is a square
        pos0 = step / 2.0

        # output position matrix
        out = np.zeros(shape=(grid_size * grid_size, 2))
        k = 0  # counter for out array

        for i in range(grid_size):  # along x-axis
            for j in range(grid_size):  # along y-axis
                # new position calculated by moving `step` distance along each axis
                out[k, :] = [pos0 + i * step, pos0 + j * step]
                k += 1  # increment counter

        # return the co-ordinates array
        return out

    @staticmethod
    @nb.njit(fastmath=True)
    def find_nearest(pos, others, condn=None):
        # Find and return the index of the entity nearest to subject positioned at `pos`
        # The co-ordinates of the entities are defined in `others` array (?, 2)
        # Additionally, an optional `condn` boolean array can be used to filter `others`

        d_min = 10000000000.0  # a large number in terms of distance
        d_min_index = -1  # index in `others` which is nearest to the subject positioned at `pos`

        # number of entities around subject positioned at `pos`
        n = others.shape[0]

        for i in range(n):  # iterate all entities in `others` array
            # distance between entity `i` and subject
            dij = (others[i, 0] - pos[0]) ** 2 + (others[i, 1] - pos[1]) ** 2
            # dij = dij ** 0.5 : this step is not needed since relative distance is needed

            # update nearest entity based on distance
            if dij < d_min and (condn is None or condn[i] == 1):
                d_min = dij
                d_min_index = i

        # return index of the nearest entity and the nearest distance associated with that entity
        return d_min_index, d_min

    @staticmethod
    @nb.njit
    def showing_symptoms(disease_state):
        # return boolean array with True if agents are showing symptoms else False
        n = disease_state.shape[0]
        out = np.zeros_like(disease_state, dtype=np.int32)
        for i in range(n):
            out[i] = (disease_state[i] == INF_SYMPTOMATIC or disease_state[i] == INF_MILD or
                      disease_state[i] == INF_SEVERE)
        return out

    @staticmethod
    @nb.njit
    def is_infected(disease_states):
        # return boolean array denoting if agents are infected or not
        n = disease_states.shape[0]
        out = np.zeros_like(disease_states, dtype=np.int32)
        for i in range(n):
            out[i] = not (disease_states[i] == INF_SUSCEPTIBLE or disease_states[i] == INF_EXPOSED or
                          disease_states[i] == INF_RECOVERED)
        return out


class Camp:

    """
    Base class for camp operations.
    """

    def __init__(self, params: Parameters, camp_size: float):
        self.camp_size = camp_size
        self.params = params
        self.num_people = self.params.number_of_people_in_isoboxes + self.params.number_of_people_in_tents
        self.infection_radius = self.params.infection_radius * self.camp_size
        # If a susceptible and an infectious individual interact, then the infection is transmitted with probability pa
        self.prob_spread = self.params.prob_spread

        self.agents: np.array = None
        self.toilet_queue = {}
        self.food_line_queue = {}

    def set_agents(self, agents: np.array):
        self.agents = agents

    @staticmethod
    @nb.njit
    def simulate_households(agents: np.array, prob_spread: float) -> (np.array, int):
        """
        Function to send people to household and simulate infection dynamics in those households.
        This function is optimized using numba.

        Parameters
        ----------
        agents: A Numpy array containing data of agents who will go inside their households at current simulation step
        prob_spread: The probability if infection transmission if a susceptible and infectious agent interact

        Returns
        -------
        out: Updated agents array and the number of new infections

        """

        n = agents.shape[0]  # number of agents inside their households
        num_new_infections = 0  # number of new infections caused by household interactions

        # Loop through each pair (i,j) of agents who are inside their households.
        # If agent i and agent j shares household AND agent i is susceptible AND agent j is infectious
        # Then, infection can spread from agent j to agent i with some probability
        for i in nb.prange(n):

            # Update current activity route
            agents[i, A_ACTIVITY] = ACTIVITY_HOUSEHOLD
            # Update current location of agent to household location
            agents[i, A_X] = agents[i, A_HOUSEHOLD_X]
            agents[i, A_Y] = agents[i, A_HOUSEHOLD_Y]

            # Agent i will be infected iff he/she is currently susceptible
            if agents[i, A_DISEASE] != INF_SUSCEPTIBLE:
                # Skip if agent i is not susceptible
                continue

            num_infectious_hh = 0  # number of infectious households of agent i

            for j in nb.prange(n):

                # Agent j will infect agent i if agent j is infectious
                if (agents[j, A_DISEASE] == INF_SUSCEPTIBLE or agents[j, A_DISEASE] == INF_EXPOSED or
                        agents[j, A_DISEASE] == INF_RECOVERED):
                    # Skip if agent j is not infectious
                    continue

                # Distance between households of i and j
                dij = (agents[i, A_HOUSEHOLD_X] - agents[j, A_HOUSEHOLD_X]) ** 2 + \
                      (agents[i, A_HOUSEHOLD_Y] - agents[j, A_HOUSEHOLD_Y]) ** 2
                dij = dij ** 0.5

                if dij > SMALL_ERROR:
                    # If agents i and j don't share household, then skip
                    continue

                num_infectious_hh += 1  # agent j shares household with agent i and is also infectious

            # Probability of infection spread inside household for agent i
            # From tucker model: 𝑝𝑖𝑑ℎ = 1 − (1 − 𝑝ℎ)^ℎ𝑐𝑖𝑑.
            p = 1.0 - (1.0 - prob_spread) ** num_infectious_hh

            if random.random() <= p:  # infect agent i based on calculated probability
                agents[i, A_DISEASE] = INF_EXPOSED
                agents[i, A_DAY_COUNTER] = 0
                num_new_infections += 1

        return agents, num_new_infections

    @staticmethod
    @nb.njit
    def simulate_wander(agents: np.array, camp_size: float, relative_strength_of_interaction: float,
                        infection_radius: float, prob_spread: float) -> (np.array, int):
        """
        Simulate wandering of the agents in the camp. During wandering, agents will also infect others or get infected
        by others.

        Parameters
        ----------
        agents: A numpy array containing information of agents who are wandering at current simulation time step
        camp_size: Size of the square sized camp
        relative_strength_of_interaction: Relative encounter rate between agents of same ethnicity (gij in tucker model)
        infection_radius: Distance around each agent where infection spread can happen
        prob_spread: Probability that infection will spread from infectious to susceptible person when they interact

        Returns
        -------
        out: Updated agents array

        """
        n = agents.shape[0]  # number of agents in the camp who are wandering
        num_new_infections = 0  # number of new infections caused by household interactions

        # Wander people around their household based on home range
        for i in nb.prange(n):  # using nb.prange helps it run in parallel
            # Change the activity route of the agent
            agents[i, A_ACTIVITY] = ACTIVITY_WANDERING

            # Find r and θ for finding random point in circle centered at agent's household.
            # This r and θ values are then used to calculate the new position of the agents around their households.
            r = A_HOME_RANGE * random.random()
            theta = 2.0 * np.pi * random.random()

            # Calculate new co-ordinates from r and θ.
            agents[i, A_X] = agents[i, A_HOUSEHOLD_X] + r * np.cos(theta)
            agents[i, A_Y] = agents[i, A_HOUSEHOLD_Y] + r * np.sin(theta)

            # Clip co-ordinate values so that agents don't go outside the camp during simulation
            agents[i, A_X] = 0.0 if agents[i, A_X] < 0.0 else (camp_size if agents[i, A_X] > camp_size else
                                                               agents[i, A_X])
            agents[i, A_Y] = 0.0 if agents[i, A_Y] < 0.0 else (camp_size if agents[i, A_Y] > camp_size else
                                                               agents[i, A_Y])

        # Simulate infection dynamics for the wanderers.
        # Susceptible agent i contract infection from an infectious agent j.
        for i in range(n):
            # Agent i will be infected iff he/she is susceptible
            if agents[i, A_DISEASE] != INF_SUSCEPTIBLE:
                # Skip if agent i is not susceptible
                continue

            num_inf_interactions = 0  # Number of infectious interactions agent i has with other wanderers

            for j in range(n):
                # Agent j will infect agent i if agent j is infectious
                if (agents[j, A_DISEASE] == INF_SUSCEPTIBLE or agents[j, A_DISEASE] == INF_EXPOSED or
                        agents[j, A_DISEASE] == INF_RECOVERED):
                    # Skip if agent j is not infectious
                    continue

                # Account for relative encounter rate between agents of same ethnicity
                # gij = 1 if individuals i and j have the same background, and gij = 0.2 otherwise.
                gij = 1.0 if agents[i, A_ETHNICITY] == agents[j, A_ETHNICITY] else relative_strength_of_interaction

                # Distance between agents i and j
                dij = (agents[i, A_X] - agents[j, A_X]) ** 2 + (agents[i, A_Y] - agents[j, A_Y]) ** 2
                dij = dij ** 0.5

                # Check if agents i and j will interact. This is primarily based on distance.
                # This also accounts in the ethnicity of the agents i.e. for agents with different ethnicities the
                # distance dij will be scaled up by factor of `1/gij`
                num_inf_interactions += ((dij / gij) <= infection_radius)

            # probability of infection spread inside household for agent i
            p = 1.0 - (1.0 - prob_spread) ** num_inf_interactions

            if random.random() <= p:  # check if agent i contracts infection based on calculated probability
                # set disease state of newly exposed agent i as `INF_EXPOSED`
                agents[i, A_DISEASE] = INF_EXPOSED
                agents[i, A_DAY_COUNTER] = 0
                num_new_infections += 1

        # return updated agents array
        return agents, num_new_infections

    def simulate_queues(self, ids, queue_name):
        """
        Simulate agent visits to toilet and food line queues. The queue is assumed to be a single line (per toilet and
        food line), hence one agent will either interact with the agent on his front and/or the agent on his back.
        Depending on whether agents around one agent are infectious or not, the middle agent will contract infection.

        Parameters
        ----------
        ids: Boolean array containing `True` at indices where agent will go to queue else `False`
        queue_name: Name of the queue. Possible values ["toilet", "food_line"]

        Returns
        -------
        out: Number of new infections caused by interactions in queues

        """

        agents = self.agents[ids, :].reshape((-1, A_FEATURES))

        # get the queues where agents are going
        queue = agents[:, A_TOILET if queue_name == "toilet" else A_FOOD_LINE].copy()

        # add agents to the queue (in random order)
        np.random.shuffle(queue)

        for i, q in enumerate(queue):
            # add agent to queue if he/she is not already in the queue
            if agents[i, A_ACTIVITY] != ACTIVITY_TOILET and queue_name == "toilet":
                self.toilet_queue[q].append(ids[i])
                self.agents[ids[i], A_ACTIVITY] = ACTIVITY_TOILET
            elif agents[i, A_ACTIVITY] != ACTIVITY_FOOD_LINE and queue_name == "food_line":
                self.food_line_queue[q].append(ids[i])
                self.agents[ids[i], A_ACTIVITY] = ACTIVITY_FOOD_LINE

        # Simulate infection spread

        # Array to store the number of infected people interaction for each person
        interactions = np.zeros((self.agents.shape[0],), dtype=np.int32)

        if queue_name == "toilet":
            for t in self.toilet_queue:
                t_ids = self.toilet_queue[t]
                for i in range(len(t_ids)):
                    # For each agent `t_ids[i]` in the queue, check if agent in front `i-1` and back `i+1` are
                    # infectious. If they are, then add to the `interactions` array
                    if i-1 >= 0:
                        interactions[t_ids[i]] += (self.agents[t_ids[i-1], A_DISEASE]
                                                   not in (INF_SUSCEPTIBLE, INF_EXPOSED, INF_RECOVERED))
                    if i+1 <= len(t_ids)-1:
                        interactions[t_ids[i]] += (self.agents[t_ids[i+1], A_DISEASE]
                                                   not in (INF_SUSCEPTIBLE, INF_EXPOSED, INF_RECOVERED))
        elif queue_name == "food_line":
            for f in self.food_line_queue:
                f_ids = self.food_line_queue[f]
                for i in range(len(f_ids)):
                    # For each agent `f_ids[i]` in the queue, check if agent in front `i-1` and back `i+1` are
                    # infectious. If they are, then add to the `interactions` array
                    if i-1 >= 0:
                        interactions[f_ids[i]] += (self.agents[f_ids[i-1], A_DISEASE]
                                                   not in (INF_SUSCEPTIBLE, INF_EXPOSED, INF_RECOVERED))
                    if i+1 <= len(f_ids)-1:
                        interactions[f_ids[i]] += (self.agents[f_ids[i+1], A_DISEASE]
                                                   not in (INF_SUSCEPTIBLE, INF_EXPOSED, INF_RECOVERED))

        # find probability of infection spread per agent via queue interaction
        prob = 1.0 - (1.0 - self.prob_spread) ** interactions
        newly_exposed_ids = (self.agents[:, A_DISEASE] == INF_SUSCEPTIBLE) * \
                            (np.random.random((self.agents.shape[0],)) <= prob)
        self.agents[newly_exposed_ids, A_DISEASE] = INF_EXPOSED
        self.agents[newly_exposed_ids, A_DAY_COUNTER] = 0

        return np.count_nonzero(newly_exposed_ids)

    def update_queues(self, pct_dequeue: float, dequeue_activity: int = ACTIVITY_WANDERING) -> None:
        # remove agents from the front of the queues and send them to location defined by `dequeue_activity`
        for t in self.toilet_queue:
            # at each step during the day, we clear 80% of all agents in the queue
            dequeue_count = int(np.ceil(pct_dequeue * len(self.toilet_queue[t])))
            try:
                # get first `dequeue_count` agents at front of the queue
                front = self.toilet_queue[t][:dequeue_count]
                # remove them from the queue
                self.toilet_queue[t] = self.toilet_queue[t][dequeue_count:]
                # change activity status to wandering for people who left toilet
                self.agents[front, A_ACTIVITY] = dequeue_activity
            except IndexError:
                pass
        for f in self.food_line_queue:
            # at each step of the day, we clear 80% of all agents in the queue
            dequeue_count = int(np.ceil(pct_dequeue * len(self.food_line_queue[f])))
            try:
                # get first `dequeue_count` agents at front of the queue
                front = self.food_line_queue[f][:dequeue_count]
                # remove them from the queue
                self.food_line_queue[f] = self.food_line_queue[f][dequeue_count:]
                # change activity status to wandering for people who left food line
                self.agents[front, A_ACTIVITY] = dequeue_activity
            except IndexError:
                pass

    @staticmethod
    @nb.njit
    def disease_progression(agents: np.array) -> np.array:
        """
        Update the disease state of the agent defined by `agents` numpy array. This is inspired partly by tucker model.
        This method is called at the end of each day.
        Parameters
        ----------
        agents: The numpy array containing the agents information

        Returns
        -------
        out: The updated agents numpy array

        """

        n = agents.shape[0]  # number of agents in the camp

        # After 5 days showing symptoms, individuals pass from the symptomatic to the “mild” or “severe” states, with
        # age- and condition-dependent probabilities following Verity and colleagues (2020) and Tuite and colleagues
        # (preprint).

        # Verity (low-risk) : probability values for each age slot [0-10, 10-20, ...90+]
        # Symptomatic agent (with low risk) in age slot `a` will have `P_symp2mild[a]` probability of turning mild
        p_symp2mild = [0, .000408, .0104, .0343, .0425, .0816, .118, .166, .184]

        # Tuite (high-risk) : probability values for each age slot [0-10, 10-20, ...90+]
        # Symptomatic agent (with high risk) in age slot `a` will have `P_symp2sevr[a]` probability of turning severe
        p_symp2sevr = [.0101, .0209, .0410, .0642, .0721, .2173, .2483, .6921, .6987]

        # Probability that a severely infected agent in a given age slot will be hospitalized
        # These probability values are taken from "Estimates of the severity of coronavirus disease 2019: a model-based
        # analysis" paper by Robert Verity et al. (DOI https://doi.org/10.1016/S1473-3099(20)30243-7)
        p_sevr2hosp = [0.0, 0.000408, 0.0104, 0.0343, 0.0425, 0.0816, 0.118, 0.166, 0.184]

        # Probability that a hospitalized agent in a given age slot will die
        p_hosp2dead = [0.0000161, 0.0000695, 0.000309, 0.000844, 0.00161, 0.00595, 0.0193, 0.0428, 0.0780]

        # Iterate all agents one by one and update the disease state
        for i in range(n):

            # read current attributes of the agent

            is_high_risk = int(agents[i, A_AGE] > 80)  # define which agents are considered as high risk

            disease_state = int(agents[i, A_DISEASE])  # current disease state
            activity = int(agents[i, A_ACTIVITY])  # current activity of the agent
            day_count = int(agents[i, A_DAY_COUNTER])  # current disease state day count
            is_asymptomatic = int(agents[i, A_IS_ASYMPTOMATIC])  # flag if agent is asymptomatic by nature
            incubation_period = int(agents[i, A_INCUBATION_PERIOD])  # incubation period of the agent
            age_slot = int(agents[i, A_AGE]/10.0)  # age slot of the agent

            # In the first half of this period, the individual is “exposed” but not infectious. In the second half, the
            # individual is “pre-symptomatic” and infectious
            if disease_state == INF_EXPOSED and day_count >= incubation_period / 2.0:
                disease_state = INF_PRESYMPTOMATIC

            # After the incubation period, the individual enters one of two states: “symptomatic” or “1st asymptomatic.”
            # All children under the age of 16 become asymptomatic (ref), and others become asymptomatic with
            # probability 0.178 (Mizumoto et al. 2020). Individuals remain in the symptomatic or 1st asymptomatic states
            # for 5 days, and are infectious during this period
            elif disease_state == INF_PRESYMPTOMATIC and day_count >= incubation_period \
                    and is_asymptomatic == 1:
                disease_state = INF_ASYMPTOMATIC1
                day_count = 0

            elif disease_state == INF_PRESYMPTOMATIC and day_count >= incubation_period \
                    and is_asymptomatic == 0:
                disease_state = INF_SYMPTOMATIC
                day_count = 0

            # After 5 days, individuals pass from the symptomatic to the “mild” or “severe” states, with age- and
            # condition-dependent probabilities following Verity and colleagues (2020) and Tuite and colleagues
            # (preprint). All individuals in the 1st asymptomatic state pass to the “2nd asymptomatic” state
            elif disease_state == INF_SYMPTOMATIC and day_count >= SYMP_PERIOD and \
                    is_high_risk == 0 and random.random() <= p_symp2mild[age_slot]:
                # A low-risk agent under symptomatic condition for more than `SYMP_PERIOD` days will go into mild state
                disease_state = INF_MILD
                day_count = 0

            elif disease_state == INF_SYMPTOMATIC and day_count >= SYMP_PERIOD and \
                    is_high_risk == 1 and random.random() <= p_symp2sevr[age_slot]:
                # A high-risk agent under symptomatic condition for more than SYMP_PERIOD days will go into severe state
                disease_state = INF_SEVERE
                day_count = 0

            elif disease_state == INF_ASYMPTOMATIC1 and day_count >= SYMP_PERIOD:
                # An agent in asymptomatic 1 condition for more than SYMP_PERIOD days will go into asymptomatic 2 state
                disease_state = INF_ASYMPTOMATIC2
                day_count = 0

            # On each day, individuals in the mild or 2nd asymptomatic state pass to the recovered state with
            # probability 0.37 (Lui et al. 2020), and individuals in the severe state pass to the recovered state with
            # probability 0.071 (Cai et al., preprint).
            elif (agents[i, A_DISEASE] == INF_MILD or disease_state == INF_ASYMPTOMATIC2) and random.random() <= 0.37:
                disease_state = INF_RECOVERED
                day_count = 0

            elif disease_state == INF_SEVERE:
                p = random.random()
                if day_count > 10 or (activity == ACTIVITY_HOSPITALIZED and p <= p_hosp2dead[age_slot]):
                    # Agents in severe state for more than 10 days die AND
                    # Agents in severe state and in hospital die with some probability
                    disease_state = INF_DECEASED
                    if activity == ACTIVITY_HOSPITALIZED:
                        activity = -2  # died in hospital
                    else:
                        activity = -1  # died outside hospital
                elif p <= 0.071:
                    # Severe case recovered
                    disease_state = INF_RECOVERED
                    day_count = 0
                elif activity != ACTIVITY_HOSPITALIZED and p <= p_sevr2hosp[age_slot]:
                    # Severe case hospitalized
                    activity = ACTIVITY_HOSPITALIZED

            # Agents who recovered in hospital can go back to household
            if disease_state == INF_RECOVERED and activity == ACTIVITY_HOSPITALIZED:
                activity = ACTIVITY_HOUSEHOLD

            # Update array with updated values
            agents[i, A_DISEASE] = disease_state
            agents[i, A_ACTIVITY] = activity
            agents[i, A_DAY_COUNTER] = day_count

        return agents
