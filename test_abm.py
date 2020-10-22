import sys
import matplotlib.pyplot as plt

from ai4good.runner.facade import Facade
from ai4good.models.abm.np_impl.moria import *
from ai4good.models.model_registry import create_params
from ai4good.utils.logger_util import get_logger

# Don't add info/debug logs from numba
logger = get_logger(__name__)


def get_params(_profile='BaselineHTHI'):
    _model = 'agent-based-model'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


def simulate(i, prf):
    logger.info("{}. Starting profile {} at {}"
                 .format(i + 1, prf, datetime.datetime.strftime(datetime.datetime.now(), "%d%m%Y_%H%M")))
    param = get_params(prf)
    camp = Moria(params=param, profile=prf + "_{}".format(i+1))
    camp.simulate()
    logger.info("{}. Completed profile {} at {}"
                 .format(i + 1, prf, datetime.datetime.strftime(datetime.datetime.now(), "%d%m%Y_%H%M")))


class SampleRun:

    def __init__(self, profiles: list):
        for i, prf in enumerate(profiles):
            simulate(i, prf)

    @staticmethod
    def plot_df(f_name):
        df = pd.read_csv(f_name)

        # Plot total count of infectious people in the camp on each day
        # pad = 1
        # fig, ax = plt.subplots(1, figsize=(10, 6))
        #
        # a0 = df.loc[:, 'INF_AGE0-9']
        # a1 = df.loc[:, 'INF_AGE10-19']
        # a2 = df.loc[:, 'INF_AGE20-29']
        # a3 = df.loc[:, 'INF_AGE30-39']
        # a4 = df.loc[:, 'INF_AGE40-49']
        # a5 = df.loc[:, 'INF_AGE50-59']
        # a6 = df.loc[:, 'INF_AGE60-69']
        # a7 = df.loc[:, 'INF_AGE70+']
        #
        # ax.bar(range(0, pad*df.shape[0], pad), a0, 1, label='Age 0-9')
        # ax.bar(range(0, pad*df.shape[0], pad), a1, 1, bottom=a0, label='Age 10-19')
        # ax.bar(range(0, pad*df.shape[0], pad), a2, 1, bottom=a0+a1, label='Age 20-29')
        # ax.bar(range(0, pad*df.shape[0], pad), a3, 1, bottom=a0+a1+a2, label='Age 30-39')
        # ax.bar(range(0, pad*df.shape[0], pad), a4, 1, bottom=a0+a1+a2+a3, label='Age 40-49')
        # ax.bar(range(0, pad*df.shape[0], pad), a5, 1, bottom=a0+a1+a2+a3+a4, label='Age 50-59')
        # ax.bar(range(0, pad*df.shape[0], pad), a6, 1, bottom=a0+a1+a2+a3+a4+a5, label='Age 60-69')
        # ax.bar(range(0, pad*df.shape[0], pad), a7, 1, bottom=a0+a1+a2+a3+a4+a5+a6, label='Age 70+')
        # ax.title.set_text("Count of infectious people (including asymptomatic) in the camp")
        # plt.legend()
        # plt.show()

        # Plot total count of agents per disease state on each day
        fig, ax = plt.subplots(1, figsize=(10, 6))

        t = df.loc[:, 'DAY']
        ax.plot(t, df.loc[:, 'SUSCEPTIBLE'], label='Susceptible')
        ax.plot(t, df.loc[:, 'EXPOSED'], label='Exposed')
        ax.plot(t, df.loc[:, 'PRESYMPTOMATIC'], label='Presymptomatic')
        ax.plot(t, df.loc[:, 'SYMPTOMATIC'], label='Symptomatic')
        ax.plot(t, df.loc[:, 'MILD'], label='Mild')
        ax.plot(t, df.loc[:, 'SEVERE'], label='Severe')
        ax.plot(t, df.loc[:, 'ASYMPTOMATIC1'], label='Asymptomatic1')
        ax.plot(t, df.loc[:, 'ASYMPTOMATIC2'], label='Asymptomatic2')
        ax.plot(t, df.loc[:, 'RECOVERED'], label='Recovered')
        ax.plot(t, df.loc[:, 'HOSPITALIZED'], label='Hospitalized')
        ax.title.set_text("Total count in each disease state")

        plt.legend()
        plt.show()

        # Plot total count of agents per disease state on each day
        fig, ax = plt.subplots(1, figsize=(10, 6))

        t = df.loc[:, 'DAY']
        ax.plot(t, df.loc[:, 'SUSCEPTIBLE'], label='Susceptible')
        ax.plot(t, df.loc[:, ['EXPOSED', 'PRESYMPTOMATIC', 'SYMPTOMATIC', 'MILD', 'SEVERE', 'ASYMPTOMATIC1',
                              'ASYMPTOMATIC2']].sum(axis=1), label='Infected')
        ax.plot(t, df.loc[:, 'RECOVERED'], label='Recovered')
        ax.title.set_text("SIR")

        plt.legend()
        plt.show()

        # Plot count of new infections for each activity on each day
        fig, ax = plt.subplots(1, figsize=(10, 6))

        h = df.loc[:, 'NEW_INF_HOUSEHOLD']
        w = df.loc[:, 'NEW_INF_WANDERING']
        t = df.loc[:, 'NEW_INF_TOILET']
        f = df.loc[:, 'NEW_INF_FOOD_LINE']

        pad = 1
        ax.bar(range(0, pad * df.shape[0], pad), h, 1, label='In household')
        ax.bar(range(0, pad * df.shape[0], pad), w, 1, bottom=h, label='Wandering')
        ax.bar(range(0, pad * df.shape[0], pad), t, 1, bottom=h+w, label='In toilet queue')
        ax.bar(range(0, pad * df.shape[0], pad), f, 1, bottom=h+w+t, label='In food line queue')
        ax.title.set_text("Count of new infections (location wise)")
        plt.legend()
        plt.show()

    def plot(self):

        prf = self.profiles[0]
        param = get_params(prf)
        camp = Moria(params=param)
        camp.day()

        markers = ['*', 'v', 'o']

        fig, ax1 = plt.subplots(1, figsize=(10, 6))
        ax1.axis('equal')

        capacities = camp.households[:, 1].tolist()
        unq_capacities = list(np.unique(capacities))
        for i in range(len(unq_capacities)):
            c = unq_capacities[i]
            hh_xy = camp.households[capacities == c, 2:]
            ax1.scatter(hh_xy[:, 0], hh_xy[:, 1], marker=markers[i], s=4)
        for a in range(camp.agents.shape[0]):
            ax1.add_patch(plt.Circle(camp.agents[a, [A_X, A_Y]], radius=camp.agents[a, A_HOME_RANGE], color='yellow',
                                     fill=False, linewidth=1))

        ax1.scatter(camp.agents[:, A_X], camp.agents[:, A_Y], marker='.', s=1)

        plt.show()


if __name__ == "__main__":

    if "--plot" in sys.argv:
        SampleRun.plot_df(sys.argv[2])

    else:

        profile = "BaselineLTHI"
        num_simulations = 1  # number of simulations to run

        # run simulations
        sampleRun = SampleRun([profile] * num_simulations)
