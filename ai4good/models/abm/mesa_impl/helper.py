import math
import random
import numpy as np
from numba import njit

from ai4good.models.abm.mesa_impl.common import *


class CampHelper(object):
    """
    Helper class for camp functions
    """

    @staticmethod
    @njit
    def _position_blocks(grid_size: int) -> np.array:
        """
        Uniform placement of blocks (typically food line or toilet) in the camp.

        Parameters
        ----------
            grid_size: Size of the square grid where placement of food line/toilet happens

        Returns
        -------
            out: (grid_size * grid_size, 2) shaped array containing (x, y) co-ordinates of the blocks

        """

        # since the placement will be uniform and equidistant, there will be a fixed distance between two blocks along
        # an axis. We call this distance as step
        step = CAMP_SIZE / grid_size

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
    @njit
    def filter_agents(people, skip_agent_id=-1, route=-1, household_id=-1, is_infected=-1, has_symptoms=-1,
                      ethnic_grp=-1):
        """
        Filter agents by various parameters. The indices of the filtered agents are returned

        Parameters
        ----------
            people: A 2D array where each row contains [route, household_id, disease_state, ethnic group, home range]
                for agent
            skip_agent_id: Unique id of the agent to skip in result (value: -1 means ignore filter)
            route: Current route of the agents (value: -1 means ignore filter)
            household_id: Household id of the agents (value: -1 means ignore filter)
            is_infected: To check if agent is infected (1) or not (0) (value: -1 means ignore filter)
            has_symptoms: To check if agent is showing symptoms (1) or not (0) (value: -1 means ignore filter)
            ethnic_grp: Ethnic group id of the agents (value: -1 means ignore filter)

        Returns
        -------
            out: Index of the filtered agents

        """

        n = people.shape[0]  # number of people
        out = []  # output index array

        for i in range(n):
            if (
                # (-1 means don't apply filter)
                (route == -1 or people[i, 0] == route)  # filter based on route
                and (household_id == -1 or people[i, 1] == household_id)  # filter based on household
                and (
                    is_infected == -1 or (
                        # filter infected people
                        is_infected == 1 and people[i, 2] in [SYMPTOMATIC, ASYMPTOMATIC1,
                                                              ASYMPTOMATIC2, MILD,
                                                              SEVERE, PRESYMPTOMATIC]
                    ) or (
                        # filter uninfected people
                        is_infected == 0 and people[i, 2] in [SUSCEPTIBLE, EXPOSED,
                                                              RECOVERED]
                    )
                )
                and (
                    has_symptoms == -1 or (
                        # filter people showing symptoms
                        has_symptoms == 1 and people[i, 2] in [SYMPTOMATIC, MILD,
                                                               SEVERE]
                    ) or (
                        # filter people showing no symptoms
                        has_symptoms == 0 and people[i, 2] in [SUSCEPTIBLE, EXPOSED,
                                                               PRESYMPTOMATIC, RECOVERED,
                                                               ASYMPTOMATIC1, ASYMPTOMATIC2]
                    )
                )
                and (skip_agent_id == -1 or i != skip_agent_id)  # skip agent
                and (ethnic_grp == -1 or people[i, 3] == ethnic_grp)  # filter based on ethnicity
            ):
                out.append(i)

        # return as numpy array
        return np.array(out)


class PersonHelper(object):
    """
    Helper class for Person functions
    """

    @staticmethod
    @njit(fastmath=True)
    def _move(center, radius):
        # random wandering simulation: person will move within the home range centered at household
        r = random.random() * radius
        theta = random.random() * (2 * math.pi)

        # get random position within home range
        new_x = center[0] + r * math.cos(theta)
        new_y = center[1] + r * math.sin(theta)

        # clip position so as to not move outside the camp
        new_x = 0.0 if new_x < 0.0 else (CAMP_SIZE if new_x > CAMP_SIZE else new_x)
        new_y = 0.0 if new_y < 0.0 else (CAMP_SIZE if new_y > CAMP_SIZE else new_y)

        return new_x, new_y

    @staticmethod
    @njit(fastmath=True)
    def find_nearest(pos, others, condn=None):
        # Find and return the index of the entity nearest to subject positioned at `pos`
        # The co-ordinates of the entities are defined in `others` array (?, 2)
        # Additionally, an optional `condn` boolean array can be used to filter `others`

        d_min = CAMP_SIZE * 10000.0  # a large number in terms of distance
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
    @njit(fastmath=True)
    def wandering_infection_spread(people: np.array, i: int, hh_pos: np.array) -> float:
        # Calculate the probability that infection spread will happen while agent `i` is wandering
        # Agent `i` will be susceptible
        # [0:route, 1:household, 2:disease state, 3:ethnicity, 4:home range, 5:x, 6:y]

        num = 0.0
        num_ppl = people.shape[0]  # number of people

        hi = int(people[i, 1])  # household id of agent `i`
        ri = people[i, 4]  # home range of agent `i`

        # Loop through all agents `j` in the camp
        # For each agent `j`, we check if `j` infects `i`
        for j in range(num_ppl):

            # skip agent `j`:
            # if agent `i` and `j` are same OR
            # if `j` share household with `i` OR
            # if `j` is not infected
            # if `j` is under isolated/quarantined because isolated agents/households cannot infect other agents
            if i == j \
                    or int(people[i, 1]) == int(people[j, 1]) \
                    or int(people[j, 2]) in (SUSCEPTIBLE, EXPOSED, RECOVERED, DECEASED) \
                    or int(people[j, 0]) == QUARANTINED:
                continue

            hj = int(people[j, 1])  # household id of agent `j`
            rj = people[j, 4]  # home range of agent `j`

            # get distance between household centers of agent i and j
            d = (hh_pos[hi, 0] - hh_pos[hj, 0]) ** 2 + (hh_pos[hi, 1] - hh_pos[hj, 1]) ** 2
            d = d ** 0.5

            if d >= (ri + rj):
                # the circles centered at households of `i` and `j` must overlap in order for infection spread
                # if the circles don't overlap, skip agent `j`
                continue

            # Check if agents `i` and `j` are inside overlapping area of their house ranges
            # di1: distance between agent i and agent i's household center
            di1 = (people[i, 5] - hh_pos[hi, 0]) ** 2 + (people[i, 6] - hh_pos[hi, 1]) ** 2
            # di2: distance between agent i and agent j's household center
            di2 = (people[i, 5] - hh_pos[hj, 0]) ** 2 + (people[i, 6] - hh_pos[hj, 1]) ** 2
            # dj1: distance between agent j and agent i's household center
            dj1 = (people[j, 5] - hh_pos[hi, 0]) ** 2 + (people[j, 6] - hh_pos[hi, 1]) ** 2
            # dj2: distance between agent j and agent j's household center
            dj2 = (people[j, 5] - hh_pos[hj, 0]) ** 2 + (people[j, 6] - hh_pos[hj, 1]) ** 2

            # if agent `i` and agent `j` are not both inside the area of overlap of their home ranges, then skip as
            #  no infection spread possible between `i` and `j`
            # else, agent `i` and `j` are both inside the area of overlap, add to result
            if di1 < ri * ri and di2 < rj * rj and \
                    dj1 < ri * ri and dj2 < rj * rj:
                # agent `j` can infect agent `i` since both `i` and `j` are insider overlapping region AND
                # agent `j` is infected

                # To obtain the interaction rate between individuals i and j from the relative encounter rate, we scale
                # by a factor gij to account for ethnicity or country of origin. In particular, gij = 1 if individuals i
                # and j have the same background, and gij = 0.2 otherwise
                # In particular, gij = 1 if individuals i and j have the same background, and gij = 0.2 otherwise
                gij = 1 if int(people[i, 3]) == int(people[j, 3]) else 0.2

                # add to interaction rate for agent `i`
                num += gij

        return num

    @staticmethod
    def _is_showing_symptoms(disease_state):
        # returns true if `disease_state` is one where agent shows symptoms
        return disease_state in (
            SYMPTOMATIC, MILD, SEVERE
        )
