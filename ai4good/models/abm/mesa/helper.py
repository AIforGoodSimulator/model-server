import math
import random
import numpy as np
from numba import njit

from ai4good.models.abm.mesa.utils import _clip_coordinates
from ai4good.models.abm.mesa.common import DiseaseStage
from ai4good.models.abm.mesa.model import Camp


class CampHelper(object):
    """
    Helper class for camp functions
    """

    def _find_nearest_toilet(self):
        pass

    @staticmethod
    @njit
    def _prob_m(hh_pos, people):
        # The probability that susceptible individual i becomes infected on day d while moving about its home range.
        # To understand the math behind this method, refer to "Infection as individuals move about the camp" section
        # of tucker model.
        # TODO: this is not abm

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
            if people[i, 3] not in [DiseaseStage.SUSCEPTIBLE, DiseaseStage.EXPOSED, DiseaseStage.PRESYMPTOMATIC,
                                    DiseaseStage.ASYMPTOMATIC1, DiseaseStage.ASYMPTOMATIC2, DiseaseStage.RECOVERED]:
                continue

            for j in range(n_ppl):

                if i == j:
                    continue

                # assumption: only individuals without symptoms interact in their home ranges
                if people[j, 3] not in [DiseaseStage.SUSCEPTIBLE, DiseaseStage.EXPOSED, DiseaseStage.PRESYMPTOMATIC,
                                        DiseaseStage.ASYMPTOMATIC1, DiseaseStage.ASYMPTOMATIC2,
                                        DiseaseStage.RECOVERED]:
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
            p_idm = 1 - math.exp(-qid * Camp.Pa)
            out.append(p_idm)

        return out


class PersonHelper(object):

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
