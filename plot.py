import time
import numba
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ai4good.runner.facade import Facade
import ai4good.models.abm.mesa_impl.model as mm
from ai4good.models.abm.mesa_impl.common import *
from ai4good.models.model_registry import create_params

logging.basicConfig(level=logging.INFO)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def get_params():
    _model = 'agent-based-model'
    _profile = 'baseline'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


class Plotter(object):

    def __init__(self):
        params = get_params()
        self.camp = mm.Camp(params=params)

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim([0, CAMP_SIZE])
        self.ax.set_ylim([0, CAMP_SIZE])
        # self.ax.axis('equal')
        # self.scat = self.ax.scatter(self.camp.agents_pos[:, 0], self.camp.agents_pos[:, 1], c='blue', s=2)
        self.anim = None

        # highlight households
        # hh = self.ax.scatter(self.camp.households[:, 2], self.camp.households[:, 3], c='black', s=5)

        self.highlight_central_sq()

        logging.info("Starting simulation of x{} agents for x{} days".format(self.camp.people_count,
                                                                             self.camp.params.number_of_steps))
        logging.info("Number of tents: {}".format(self.camp.params.number_of_tents))
        logging.info("Number of iso-boxes: {}".format(self.camp.params.number_of_isoboxes))
        logging.info("Area covered by iso-boxes: {}".format(self.camp.params.area_covered_by_isoboxes))
        logging.info("Probability of infection in household: {}".format(self.camp.params.probability_infecting_person_in_household_per_day))

    def plot(self):
        self.anim = FuncAnimation(self.fig, self.animate, interval=500, frames=self.camp.params.number_of_steps - 1)
        # plt.draw()
        # plt.show()

    def animate(self, t):
        self.camp.step()

        logging.info("{}. SUS={}, EXP={}, PRE={}, SYM={}, MIL={}, SEV={}, AS1={}, AS2={}, REC={}".format(
            t,
            np.count_nonzero(self.camp.agents_disease_states == SUSCEPTIBLE),
            np.count_nonzero(self.camp.agents_disease_states == EXPOSED),
            np.count_nonzero(self.camp.agents_disease_states == PRESYMPTOMATIC),
            np.count_nonzero(self.camp.agents_disease_states == SYMPTOMATIC),
            np.count_nonzero(self.camp.agents_disease_states == MILD),
            np.count_nonzero(self.camp.agents_disease_states == SEVERE),
            np.count_nonzero(self.camp.agents_disease_states == ASYMPTOMATIC1),
            np.count_nonzero(self.camp.agents_disease_states == ASYMPTOMATIC2),
            np.count_nonzero(self.camp.agents_disease_states == RECOVERED)
        ))

        logging.info("{}. HSH={}, TLT={}, FDL={}, WDR={}, QRT={}, HSP={}".format(
            t,
            np.count_nonzero(self.camp.agents_route == HOUSEHOLD),
            np.count_nonzero(self.camp.agents_route == TOILET),
            np.count_nonzero(self.camp.agents_route == FOOD_LINE),
            np.count_nonzero(self.camp.agents_route == WANDERING),
            np.count_nonzero(self.camp.agents_route == QUARANTINED),
            np.count_nonzero(self.camp.agents_route == HOSPITALIZED)
        ))

        # self.scat.set_offsets(self.camp.agents_pos)
        # self.scat.set_color(Plotter.get_colors(self.camp.agents_disease_states))

    def highlight_central_sq(self):
        center_sq_side = CAMP_SIZE * self.camp.params.area_covered_by_isoboxes ** 0.5

        # minimum and maximum co-ordinates for central square
        p_min = (CAMP_SIZE - center_sq_side) / 2.0
        p_max = (CAMP_SIZE + center_sq_side) / 2.0

        logging.info("Central square : {}->{}".format(p_min, p_max))

        # self.ax.plot([p_min, p_max, p_max, p_min, p_min], [p_min, p_min, p_max, p_max, p_min], c='gray')

    @staticmethod
    @numba.jit(nopython=True)
    def get_colors(disease_statae):
        colors = np.empty((disease_statae.shape[0], 3))

        for i in range(colors.shape[0]):
            if disease_statae[i] in [SUSCEPTIBLE, RECOVERED, DECEASED]:
                colors[i] = [0.0, 0.0, 1.0]
            elif disease_statae[i] in [EXPOSED, PRESYMPTOMATIC, ASYMPTOMATIC1, ASYMPTOMATIC2]:
                colors[i] = [1.0, 1.0, 0.0]
            elif disease_statae[i] in [SYMPTOMATIC, MILD, SEVERE]:
                colors[i] = [1.0, 0.0, 0.0]

        return colors


if __name__ == "__main__":
    plotter = Plotter()

    for i in range(100):
        t1 = time.time()
        plotter.animate(i)
        t2 = time.time()
        logging.info("Completed step {} in {} seconds".format(i, t2-t1))

# Baseline model profiling
# INFO:root:Completed camp initialization in 0.9704179763793945 seconds
# INFO:root:Starting simulation for x200 days
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████
# ██████████████████████████████████| 200/200 [13:10<00:00,  3.95s/it]
# INFO:root:Completed simulation in 790.33283162117 seconds
