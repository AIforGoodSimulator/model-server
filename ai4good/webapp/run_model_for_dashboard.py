from ai4good.webapp.model_runner import ModelScheduleRunResult
from ai4good.webapp.model_results_config import model_profile_config
from ai4good.webapp.model_runner import ModelRunner, _sid
from ai4good.webapp.apps import facade, _redis, dask_client
import time
from ai4good.utils.logger_util import get_logger

logger = get_logger(__name__)


# model name possibiltiies: ['compartmental-model', 'network-model', 'agent-based-model']
model_runner = ModelRunner(facade, _redis, dask_client, _sid)
queue = []

class queueItem:
    def __init__(self, model, profile):
        self.model = model
        self.profile = profile


def run_model_results_for_message(message_key):
    for model in model_profile_config[message_key].keys():
        if len(model_profile_config[message_key][model])>0:
            for profile in model_profile_config[message_key][model]:
                res = model_runner.run_model(model, profile)
                if res == ModelScheduleRunResult.SCHEDULED:
                    print("Model run scheduled")
                elif res == ModelScheduleRunResult.CAPACITY:
                    queue.append(queueItem(model, profile))
                    print("Can not run model now, added to queue position: ",len(queue))
                elif res == ModelScheduleRunResult.ALREADY_RUNNING:
                    print("Already running")
                else:
                    raise RuntimeError("Unsupported result type: "+str(res))
        while len(queue) > 0:
            time.sleep(10)
            print(str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())), " Queue Len: ",len(queue))
            for item in queue:
                res = model_runner.run_model(item.model, item.profile)
                if res == ModelScheduleRunResult.SCHEDULED:
                    print("Model run scheduled")
                    queue.remove(item)
                elif res == ModelScheduleRunResult.CAPACITY:
                    print("Can not run model now, staying in queue")
                elif res == ModelScheduleRunResult.ALREADY_RUNNING:
                    print("Already running")
                else:
                    raise RuntimeError("Unsupported result type: "+str(res))

        return None

## here we will need two buttons - one is simualtion the other is view results and the view results button look into model_result to see if that is ready to be shown
