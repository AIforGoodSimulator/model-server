import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from ai4good.runner.facade import Facade
from ai4good.models.abm.np_impl.moria import *
from ai4good.models.model_registry import create_params


# Don't add info/debug logs from numba
logging.basicConfig(level=logging.INFO)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def get_params():
    _model = 'agent-based-model'
    # Possible values: "baseline", "small", ...
    _profile = 'BaselineHTHI'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


class TestRun(object):

    def __init__(self):
        params = get_params()
        self.camp = Moria(params=params)

        logging.info("Starting simulation of x{} agents for x{} days".format(self.camp.num_people,
                                                                             self.camp.params.number_of_steps))

    def run(self):
        self.camp.simulate()

    def plot(self):

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim([0, self.camp.camp_size])
        self.ax.set_ylim([0, self.camp.camp_size])
        # self.ax.axis('equal')
        self.scat = self.ax.scatter(self.camp.agents[:, A_X], self.camp.agents[:, A_Y], c='blue', s=2)
        self.anim = None

        # highlight households
        self.ax.scatter(self.camp.households[:, 2], self.camp.households[:, 3], c='black', s=5)

        self.highlight_central_sq()

        self.anim = FuncAnimation(self.fig, self.animate, interval=500, frames=self.camp.params.number_of_steps - 1)
        plt.draw()
        plt.show()

    def animate(self, t):
        self.camp.day()

        self.scat.set_offsets(self.camp.agents[:, [A_X, A_Y]])
        self.scat.set_color(TestRun.get_colors(self.camp.agents[:, A_DISEASE]))

    def highlight_central_sq(self):
        center_sq_side = CAMP_SIZE * self.camp.params.area_covered_by_isoboxes ** 0.5

        # minimum and maximum co-ordinates for central square
        p_min = (CAMP_SIZE - center_sq_side) / 2.0
        p_max = (CAMP_SIZE + center_sq_side) / 2.0

        self.ax.plot([p_min, p_max, p_max, p_min, p_min], [p_min, p_min, p_max, p_max, p_min], c='gray')

    @staticmethod
    @nb.jit(nopython=True)
    def get_colors(disease_state):
        colors = np.empty((disease_state.shape[0], 3))

        for i in range(colors.shape[0]):
            if disease_state[i] in [INF_SUSCEPTIBLE, INF_RECOVERED]:
                colors[i] = [0.0, 0.0, 1.0]
            elif disease_state[i] in [INF_EXPOSED, INF_PRESYMPTOMATIC, INF_ASYMPTOMATIC1, INF_ASYMPTOMATIC2]:
                colors[i] = [1.0, 1.0, 0.0]
            elif disease_state[i] in [INF_SYMPTOMATIC, INF_MILD, INF_SEVERE]:
                colors[i] = [1.0, 0.0, 0.0]

        return colors

    def plot_households(self):
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        household_pos = self.camp.households[:, 2:]
        x = household_pos[:, 0]
        y = household_pos[:, 1]

        # calculate distance between each household and agent
        dx2 = (x.reshape((-1, 1)) - self.camp.agents[:, A_X].reshape((1, -1))) ** 2
        dy2 = (y.reshape((-1, 1)) - self.camp.agents[:, A_Y].reshape((1, -1))) ** 2
        d = ((dx2 + dy2) ** 0.5) < SMALL_ERROR  # get agents who belong to the household

        top = d.sum(axis=1)  # number of agents per household

        bottom = np.zeros_like(top)
        width = depth = 1

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax1.set_title("Number of agents per household")

        ax2.scatter(self.camp.agents[:, A_X], self.camp.agents[:, A_Y], c=self.camp.agents[:, A_ETHNICITY])
        ax2.set_title("Agents ethnicities")

        plt.show()


if __name__ == "__main__":
    test = TestRun()

    # Uncomment following code blocks to execute diff things

    ################################
    # for plotting households in 3d
    # test.plot_households()

    ################################
    # for running on console
    test.run()

    ################################
    # for real time plotting. This can be very slow for large data
    # test.plot()
    #
    # for t in range(test.camp.params.number_of_steps):
    #     test.animate(t)
