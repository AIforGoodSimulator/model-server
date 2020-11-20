from ai4good.webapp.model_runner import ModelScheduleRunResult
from ai4good.webapp.model_results_config import model_profile_config
from ai4good.webapp.model_runner import ModelRunner, _sid
from ai4good.webapp.apps import facade, _redis, dask_client
import time
from queue import Queue
from collections import defaultdict
from ai4good.utils.logger_util import get_logger

logger = get_logger(__name__)


# model name possibiltiies: ['compartmental-model', 'network-model', 'agent-based-model']


class ModelQueueItem:
    def __init__(self, model, profile):
        self.model = model
        self.profile = profile


def run_model_results_for_messages(model_runner,message_keys):
    run_config = {}
    for message_key in message_keys:
        for model in model_profile_config[message_key].keys():
            run_config[model] = (model_profile_config[message_key][model])
    res = model_runner.batch_run_model(run_config)
    return res


def check_model_results_for_messages(model_runner,message_keys):
    results_ready = False
    for message_key in message_keys:
        for model in model_profile_config[message_key].keys():
            if len(model_profile_config[message_key][model])>0:
                for profile in model_profile_config[message_key][model]:
                    if not model_runner.results_exist(model,profile):
                        return results_ready
    results_ready = True
    return results_ready

