import random
import logging
import numpy as np
import numba as nb

from ai4good.models.abm.initialise_parameters import Parameters


# very small float number to account for floating precision loss
SMALL_ERROR = 0.0000001

FEMALE = 0
MALE = 1

# Individuals remain in the symptomatic or 1st asymptomatic states for 5 days, and are infectious during this period.
# This period of "5 days" can be parameterized in `SYMP_PERIOD`
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
        # DONE: parameterize infection radius
        self.infection_radius = params.infection_radius * self.camp_size
        self.prob_spread = 0.1

        self.agents: np.array = None
        self.toilet_queue = {}
        self.food_line_queue = {}

    def set_agents(self, agents: np.array):
        self.agents = agents

    def simulate_households(self, ids):
        # Function to send people to household
        # `ids` is a boolean array with `True` value for agents which will be in household
        agents = self.agents[ids, :]
        agents[:, A_ACTIVITY] = ACTIVITY_HOUSEHOLD

        # send people to households
        agents[:, A_X] = agents[:, A_HOUSEHOLD_X]
        agents[:, A_Y] = agents[:, A_HOUSEHOLD_Y]

        # Simulate infection spread inside households
        # find agents sharing households -> distance between them will be ~0
        dij = OptimizedOps.distance_matrix(agents[:, [A_X, A_Y]]) <= SMALL_ERROR
        # get agents who are currently susceptible (across rows)
        sus = (agents[:, A_DISEASE] == INF_SUSCEPTIBLE).reshape((-1, 1))
        # get agents who are currently infected (across columns)
        inf = OptimizedOps.is_infected(agents[:, A_DISEASE]).reshape((1, -1))

        # get interactions where column agent was infected and row agent was susceptible
        inf_interactions = (dij * sus) * inf

        # find number of agents who are sharing household with each agent
        h = np.sum(inf_interactions, axis=1)
        # probability of infection spread inside household for each agent
        p = 1.0 - (1.0 - self.prob_spread) ** h
        # update agents who got exposed
        newly_exposed_ids = np.random.random((agents.shape[0],)) <= p
        agents[newly_exposed_ids, A_DISEASE] = INF_EXPOSED
        agents[newly_exposed_ids, A_DAY_COUNTER] = 0

        logging.debug("{} new agents were exposed through household interactions".
                      format(np.count_nonzero(newly_exposed_ids)))

        self.agents[ids, :] = agents

    def simulate_wander(self, ids: np.array) -> None:
        """
        Simulate wandering of the agents in the camp. During wandering, agents will also infect others or get infected
        by others.
        Parameters
        ----------
        ids: A boolean array containing True/False if agent i will wander or not

        """

        # Get the agents who will wander around in the camp
        agents = self.agents[ids, :]
        # Change the activity route of the agents
        agents[:, A_ACTIVITY] = ACTIVITY_WANDERING

        # Find r and θ for finding random point in circle centered at agent's household.
        # This r and θ values are then used to calculate the new position of the agents around their households.
        r = A_HOME_RANGE * np.random.random((agents.shape[0],))
        theta = 2.0 * np.pi * np.random.random((agents.shape[0],))

        # Calculate new co-ordinates from r and θ.
        agents[:, A_X] = agents[:, A_HOUSEHOLD_X] + r * np.cos(theta)
        agents[:, A_Y] = agents[:, A_HOUSEHOLD_Y] + r * np.sin(theta)

        # Clip co-ordinate values so that agents don't go outside the camp during simulation
        agents[:, A_X] = np.clip(agents[:, A_X], 0.0, self.camp_size)
        agents[:, A_Y] = np.clip(agents[:, A_Y], 0.0, self.camp_size)

        # Simulate infection spread while wandering
        # First, check which agents will interact with each other
        pos = agents[:, [A_X, A_Y]]
        # distance between each agent
        dij = OptimizedOps.distance_matrix(pos)

        # ethnicities of the agents
        eth = agents[:, A_ETHNICITY]
        # find which agents share ethnicities
        # `shared_eth[i, j]` = 1 when agents i and j share same ethnicity, else 0
        shared_eth = eth.reshape((-1, 1)) == eth.reshape((1, -1))
        # Account for relative encounter rate between agents of same ethnicity
        # i.e. (`gij` or `relative_strength_of_interaction`) with value [0, 1]
        # `shared_eth_rel[i, j]` = 1 when agents i and j share same ethnicity, else `gij`
        shared_eth_rel = self.params.relative_strength_of_interaction + \
                         (1.0 - self.params.relative_strength_of_interaction) * shared_eth

        # Get agents who interact with each other (close proximity during wandering).
        # This also accounts in the ethnicity of the agents i.e. for agents with different ethnicities the distance dij
        # will be scaled up by factor of `1/gij`
        interactions = (dij/shared_eth_rel) <= self.infection_radius
        # get agents who are currently susceptible (across rows)
        sus = (agents[:, A_DISEASE] == INF_SUSCEPTIBLE).reshape((-1, 1))
        # get agents who are currently infected (across columns)
        inf = OptimizedOps.is_infected(agents[:, A_DISEASE]).reshape((1, -1))

        # get interactions where column agent was infected and row agent was susceptible
        inf_interactions = (interactions * sus) * inf

        # find number of agents who are sharing household with each agent
        h = np.sum(inf_interactions, axis=1)
        # probability of infection spread inside household for each agent
        p = 1.0 - (1.0 - self.prob_spread) ** h
        # update agents who got exposed
        newly_exposed_ids = np.random.random((agents.shape[0],)) <= p

        # set disease state of newly exposed agents as `INF_EXPOSED`
        # newly_exposed_ids = np.argwhere(inf_interactions.sum(axis=1) >= SMALL_ERROR)[:, 0]
        agents[newly_exposed_ids, A_DISEASE] = INF_EXPOSED
        agents[newly_exposed_ids, A_DAY_COUNTER] = 0

        logging.debug("{} new agents were exposed through wandering".format(newly_exposed_ids.shape[0]))

        self.agents[ids, :] = agents

    def simulate_queues(self, ids, queue_name):
        # Function to send agents to the queues (toilet and food line)
        # `ids` is a list of agent indices

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

        # find number of infected people interaction
        interactions = np.zeros((self.agents.shape[0],), dtype=np.int32)

        if queue_name == "toilet":
            for t in self.toilet_queue:
                t_ids = self.toilet_queue[t]
                for i in range(len(t_ids)):
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

        logging.debug("{} new agents were exposed through {}".format(np.count_nonzero(newly_exposed_ids), queue_name))

    def update_queues(self):
        # remove agents from the front of the queues
        for t in self.toilet_queue:
            # at each step during the day, we clear 80% of all agents in the queue
            # DONE: how can we parameterize it?
            # parameter: percentage_of_toilet_queue_cleared_at_each_step
            # I am not sure if we need it but OK.
            dequeue_count = int(np.ceil(self.percentage_of_toilet_queue_cleared_at_each_step *
                                        len(self.toilet_queue[t])))
            try:
                # get first `dequeue_count` agents at front of the queue
                front = self.toilet_queue[t][:dequeue_count]
                # remove them from the queue
                self.toilet_queue[t] = self.toilet_queue[t][dequeue_count:]
                # change activity status to wandering for people who left toilet
                self.agents[front, A_ACTIVITY] = ACTIVITY_WANDERING
            except IndexError:
                pass
        for f in self.food_line_queue:
            # at each step of the day, we clear 80% of all agents in the queue
            dequeue_count = int(np.ceil(0.8 * len(self.food_line_queue[f])))
            try:
                # get first `dequeue_count` agents at front of the queue
                front = self.food_line_queue[f][:dequeue_count]
                # remove them from the queue
                self.food_line_queue[f] = self.food_line_queue[f][dequeue_count:]
                # change activity status to wandering for people who left food line
                self.agents[front, A_ACTIVITY] = ACTIVITY_WANDERING
            except IndexError:
                pass

    @staticmethod
    @nb.njit
    def disease_progression(agents: np.array) -> np.array:
        """
        Update the disease state of the agent defined by `agents` numpy array. This is inspired partly by tucker model.
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
        asp = np.array([0, .000408, .0104, .0343, .0425, .0816, .118, .166, .184])
        # Tuite (high-risk) : probability values for each age slot [0-10, 10-20, ...90+]
        aspc = np.array([.0101, .0209, .0410, .0642, .0721, .2173, .2483, .6921, .6987])

        # Iterate all agents one by one and update the disease state
        for i in range(n):

            is_high_risk = int(agents[i, A_AGE] > 80)  # define which agents are considered as high risk
            disease_state = int(agents[i, A_DISEASE])  # current disease state
            is_asymptomatic = int(agents[i, A_IS_ASYMPTOMATIC])  # flag if agent is asymptomatic by nature
            incubation_period = int(agents[i, A_INCUBATION_PERIOD])  # incubation period of the agent

            # In the first half of this period, the individual is “exposed” but not infectious. In the second half, the
            # individual is “pre-symptomatic” and infectious
            if disease_state == INF_EXPOSED and agents[i, A_DAY_COUNTER] >= incubation_period / 2.0:
                disease_state = INF_PRESYMPTOMATIC

            # After the incubation period, the individual enters one of two states: “symptomatic” or “1st asymptomatic.”
            # All children under the age of 16 become asymptomatic (ref), and others become asymptomatic with
            # probability 0.178 (Mizumoto et al. 2020). Individuals remain in the symptomatic or 1st asymptomatic states
            # for 5 days, and are infectious during this period
            elif disease_state == INF_PRESYMPTOMATIC and agents[i, A_DAY_COUNTER] >= incubation_period \
                    and is_asymptomatic == 1:
                disease_state = INF_ASYMPTOMATIC1
                agents[i, A_DAY_COUNTER] = 0

            elif disease_state == INF_PRESYMPTOMATIC and agents[i, A_DAY_COUNTER] >= incubation_period \
                    and is_asymptomatic == 0:
                disease_state = INF_SYMPTOMATIC
                agents[i, A_DAY_COUNTER] = 0

            # After 5 days, individuals pass from the symptomatic to the “mild” or “severe” states, with age- and
            # condition-dependent probabilities following Verity and colleagues (2020) and Tuite and colleagues
            # (preprint). All individuals in the 1st asymptomatic state pass to the “2nd asymptomatic” state
            elif disease_state == INF_SYMPTOMATIC and agents[i, A_DAY_COUNTER] >= SYMP_PERIOD and \
                    is_high_risk == 0 and random.random() <= asp[int(agents[i, A_AGE]/10.0)]:
                disease_state = INF_MILD
                agents[i, A_DAY_COUNTER] = 0

            elif disease_state == INF_SYMPTOMATIC and agents[i, A_DAY_COUNTER] >= SYMP_PERIOD and \
                    is_high_risk == 1 and random.random() <= aspc[int(agents[i, A_AGE]/10.0)]:
                disease_state = INF_SEVERE
                agents[i, A_ACTIVITY] = ACTIVITY_HOSPITALIZED
                agents[i, A_DAY_COUNTER] = 0

            elif disease_state == INF_ASYMPTOMATIC1 and agents[i, A_DAY_COUNTER] >= SYMP_PERIOD:
                disease_state = INF_ASYMPTOMATIC2
                agents[i, A_DAY_COUNTER] = 0

            # On each day, individuals in the mild or 2nd asymptomatic state pass to the recovered state with
            # probability 0.37 (Lui et al. 2020), and individuals in the severe state pass to the recovered state with
            # probability 0.071 (Cai et al., preprint).
            elif (agents[i, A_DISEASE] == INF_MILD or disease_state == INF_ASYMPTOMATIC2) and random.random() <= 0.37:
                disease_state = INF_RECOVERED
                agents[i, A_DAY_COUNTER] = 0
                # Agents who recovered in hospital can go back to household
                if agents[i, A_ACTIVITY] == ACTIVITY_HOSPITALIZED:
                    agents[i, A_ACTIVITY] = ACTIVITY_HOUSEHOLD

            elif disease_state == INF_SEVERE and random.random() <= 0.071:
                disease_state = INF_RECOVERED
                agents[i, A_DAY_COUNTER] = 0
                # Agents who recovered in hospital can go back to household
                if agents[i, A_ACTIVITY] == ACTIVITY_HOSPITALIZED:
                    agents[i, A_ACTIVITY] = ACTIVITY_HOUSEHOLD

            agents[i, A_DISEASE] = disease_state

        return agents
