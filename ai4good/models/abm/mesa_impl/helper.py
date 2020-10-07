import math
import random
import numpy as np
from numba import njit

from ai4good.models.abm.mesa_impl.utils import _clip_coordinates
from ai4good.models.abm.mesa_impl.common import *


class CampHelper(object):
    """
    Helper class for camp functions
    """

    @staticmethod
    @njit
    def _position_blocks(grid_size):
        """
        Uniform placement of blocks (typically foodline or toilet) in the camp.

        Parameters
        ----------
            grid_size: Size of the square grid where placement of foodline/toilet happens

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
    def filter_agents(people, skip_agent_id, route, household_id, is_infected, has_symptoms):
        """
        Filter agents by various parameters. The indices of the filtered agents are returned

        Parameters
        ----------
            people: A 2D array where each row contains [route, household_id, disease_state] for agent
            skip_agent_id: Unique id of the agent to skip in result (value: -1 means ignore filter)
            route: Current route of the agents (value: -1 means ignore filter)
            household_id: Household id of the agents (value: -1 means ignore filter)
            is_infected: To check if agent is infected (1) or not (0) (value: -1 means ignore filter)
            has_symptoms: To check if agent is showing symptoms (1) or not (0) (value: -1 means ignore filter)

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
            ):
                out.append(i)

        # return as numpy array
        return np.array(out)

    @staticmethod
    @njit
    def _prob_m(hh_pos, people):
        """
        TODO: Implement this in agent.py
        The probability that susceptible individual i becomes infected on day d while moving about its home range.
        To understand the math behind this method, refer to "Infection as individuals move about the camp" section
        of tucker model.

        Parameters
        ----------
            hh_pos: (?, 2) array containing co-ordinates of households
            people: A 2D array containing [household id, home range, ethnic group id, disease state] for each agent

        Returns
        -------

        """

        # number of people
        n_ppl = people.shape[0]

        # probability response value
        out = []

        # iterate through each pair of individual in the camp
        for i in range(n_ppl):

            hh_id = people[i, 0]  # household id for person i
            hh_pos_i = hh_pos[hh_id, :]  # position of household for individual i
            ri = people[i, 1]  # home range for individual i

            # the rate at which individual i interacts with infected individuals in its home range on day d
            qid = 0.0

            # assumption: only individuals without symptoms interact in their home ranges
            if people[i, 3] not in [SUSCEPTIBLE, EXPOSED, PRESYMPTOMATIC,
                                    ASYMPTOMATIC1, ASYMPTOMATIC2, RECOVERED]:
                continue

            for j in range(n_ppl):

                if i == j:
                    continue

                # assumption: only individuals without symptoms interact in their home ranges
                if people[j, 3] not in [SUSCEPTIBLE, EXPOSED, PRESYMPTOMATIC,
                                        ASYMPTOMATIC1, ASYMPTOMATIC2,
                                        RECOVERED]:
                    continue

                hh_id = people[j, 0]  # household id for person j
                hh_pos_j = hh_pos[hh_id, :]  # position of household for individual j
                rj = people[j, 1]  # home range for individual j

                # the summation runs over all individuals in the model that do not share a household with individual i
                if people[i, 0] == people[j, 0]:
                    continue

                # distance between households of individuals i and j
                dij = (hh_pos_i[0] - hh_pos_j[0]) ** 2 + (hh_pos_i[1] - hh_pos_j[1]) ** 2
                dij = dij ** 0.5

                if dij > (ri + rj):
                    # no overlap condition
                    continue

                # factor to account ethnicity
                # In particular, gij = 1 if individuals i and j have the same background, and gij = 0.2 otherwise
                gij = 1 if people[i, 2] == people[j, 2] else 0.2

                # area of overlap in home ranges (equation (5) of tucker model)
                aij = ri * ri * math.acos((dij * dij + ri * ri - rj * rj) / (2.0 * dij * ri)) + \
                      rj * rj * math.acos((dij * dij - ri * ri + rj * rj) / (2.0 * dij * rj)) - \
                      (((-dij + ri + rj) * (dij + ri - rj) * (dij - ri + rj) * (dij + ri + rj)) ** 0.5) / 2.0

                # The proportion of time that individuals i and j spend together in the area of overlap
                # (equation (6) of tucker model)
                # sij = (aij * aij) / (math.pi * math.pi * ri*ri * rj*rj) : redundant for code

                # The relative encounter rate between individuals i and j (equation (7) of tucker model)
                # rer = sij/aij : redundant for code

                # the daily rate of interaction between individuals i and j (equation (8) of tucker model)
                fij = 0.02 * 0.02 * aij * gij / (math.pi * ri * ri * rj * rj)

                qid += fij

            # The probability that susceptible individual i becomes infected on day d while moving about its home range
            p_idm = 1 - math.exp(-qid * 0.1)
            out.append(p_idm)

        return out


class PersonHelper(object):
    """
    Helper class for Person functions
    """

    @staticmethod
    @njit
    def _move(center, radius):
        # random wandering simulation: person will move within the home range centered at household
        r = random.random() * radius
        theta = random.random() * (2 * math.pi)

        # get random position within home range
        new_x = center[0] + r * math.cos(theta)
        new_y = center[1] + r * math.sin(theta)

        # clip position so as to not move outside the camp
        new_x = _clip_coordinates(new_x)
        new_y = _clip_coordinates(new_y)

        return new_x, new_y

    @staticmethod
    @njit
    def find_nearest(pos, others):
        # Find and return the index of the entity nearest to subject positioned at `pos`
        # The co-ordinates of the entities are defined in `others` array (?, 2)

        d_min = CAMP_SIZE * 10000.0  # a large number in terms of distance
        d_min_index = -1  # index in `others` which is nearest to the subject positioned at `pos`

        # number of entities around subject positioned at `pos`
        n = others.shape[0]

        for i in range(n):  # iterate all entities in `others` array
            # distance between entity `i` and subject
            dij = (others[i, 0] - pos[0]) ** 2 + (others[i, 1] - pos[1]) ** 2
            # dij = dij ** 0.5 : this step is not needed since relative distance is needed

            # update nearest entity based on distance
            if dij < d_min:
                d_min = dij
                d_min_index = i

        # return index of the nearest entity and the nearest distance associated with that entity
        return d_min_index, d_min

    @staticmethod
    def _is_showing_symptoms(disease_state):
        # returns true if `disease_state` is one where agent shows symptoms
        return disease_state in (
            SYMPTOMATIC, MILD, SEVERE
        )
