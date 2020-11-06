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
    queue = []
    for message_key in message_keys:
        for model in model_profile_config[message_key].keys():
            run_config[model] = (model_profile_config[message_key][model])
    # for model, profile_group in run_config.items():
    #     for profiles in profile_group:
    #         for profile in profiles:
    #             logger.info(f"trying to get profile {profile} into model runner")
    #             res = model_runner.run_model(model, profile)
    #             if res == ModelScheduleRunResult.SCHEDULED:
    #                 logger.info("Model run scheduled")
    #             elif res == ModelScheduleRunResult.CAPACITY:
    #                 queue.append('|'.join([model, profile]))
    #                 logger.info("Can not run model now, added to queue")
    #             elif res == ModelScheduleRunResult.ALREADY_RUNNING:
    #                 logger.info("Already running")
    #             else:
    #                 raise RuntimeError("Unsupported result type: " + str(res))
    model_runner.batch_run_model(run_config)
    return queue
    # while q.empty() is False:
    #     #check every 10 seconds
    #     time.sleep(10)
    #     if model_runner.models_running_now.run_available():
    #         model, profile = q.get()
    #         res = model_runner.run_model(model, profile)
    #         if res == ModelScheduleRunResult.SCHEDULED:
    #             logger.info("Model run scheduled")
    #             continue
    #         elif res == ModelScheduleRunResult.CAPACITY:
    #             # put the pair back to the queue
    #             q.put([model, profile])
    #             logger.info("Can not run model now, added to queue")
    #         elif res == ModelScheduleRunResult.ALREADY_RUNNING:
    #             logger.info("Already running")
    #         else:
    #             raise RuntimeError("Unsupported result type: " + str(res))
    # logger.info("all model runs are finished")
    # return None

## here we will need two buttons - one is simualtion the other is view results and the view results button look into model_result to see if that is ready to be shown
