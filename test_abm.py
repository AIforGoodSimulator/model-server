import multiprocessing

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
        jobs = []
        for i, prf in enumerate(profiles):
            process = multiprocessing.Process(target=simulate, args=(i, prf))
            jobs.append(process)
        for job in jobs:
            job.start()


if __name__ == "__main__":
    # Just pass the profiles in order here
    sampleRun = SampleRun([
        "BaselineHTHI"
    ])
