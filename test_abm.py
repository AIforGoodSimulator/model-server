import sys
import multiprocessing
import matplotlib.pyplot as plt

from ai4good.runner.facade import Facade
from ai4good.models.abm.np_impl.moria import *
from ai4good.models.model_registry import create_params


# Don't add info/debug logs from numba
logging.basicConfig(level=logging.INFO)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def get_params(_profile='BaselineHTHI'):
    _model = 'agent-based-model'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


def simulate(i, prf):
    logging.info("{}. Starting profile {} at {}"
                 .format(i + 1, prf, datetime.datetime.strftime(datetime.datetime.now(), "%d%m%Y_%H%M")))
    param = get_params(prf)
    camp = Moria(params=param, profile=prf)
    camp.simulate()
    logging.info("{}. Completed profile {} at {}"
                 .format(i + 1, prf, datetime.datetime.strftime(datetime.datetime.now(), "%d%m%Y_%H%M")))


class SampleRun:

    def __init__(self, profiles: list):
        self.profiles = profiles
        self.jobs = []
        for i, prf in enumerate(self.profiles):
            process = multiprocessing.Process(target=simulate, args=(i, prf))
            self.jobs.append(process)

    def run(self):
        for job in self.jobs:
            job.start()

    @staticmethod
    def plot_age_df(f_name):
        df = pd.read_csv(f_name)
        pad = 1
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE0-9'], 1, label='Age 0-9')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE10-19'], 1, bottom=df.loc[:, 'INF_AGE0-9'], label='Age 10-19')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE20-29'], 1, bottom=df.loc[:, 'INF_AGE10-19'], label='Age 20-29')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE30-39'], 1, bottom=df.loc[:, 'INF_AGE20-29'], label='Age 30-39')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE40-49'], 1, bottom=df.loc[:, 'INF_AGE30-39'], label='Age 40-49')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE50-59'], 1, bottom=df.loc[:, 'INF_AGE40-49'], label='Age 50-59')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE60-69'], 1, bottom=df.loc[:, 'INF_AGE50-59'], label='Age 60-69')
        plt.bar(range(0, pad*df.shape[0], pad), df.loc[:, 'INF_AGE70+'], 1, bottom=df.loc[:, 'INF_AGE60-69'], label='Age 70+')
        plt.title("Count of infectious people in the camp")
        plt.legend()
        plt.show()

    def plot(self):

        prf = self.profiles[0]
        param = get_params(prf)
        camp = Moria(params=param, profile=prf)
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

    # Just pass the profiles in order here
    sampleRun = SampleRun([
        "QuarantineHTHI"
    ])

    if "--plot" in sys.argv:
        SampleRun.plot_age_df(sys.argv[2])
    else:
        sampleRun.run()
