from ai4good.webapp.apps import model_runner
from ai4good.webapp.model_runner import ModelScheduleRunResult
from ai4good.webapp.model_results_config import model_profile_config
# model name possibiltiies: ['compartmental-model', 'network-model', 'agent-based-model']

CAMP = 'Moria' #this logic needs to be changed later where the user input params will be stored

def run_model_results_for_message(message_key):
    for model in model_profile_config[message_key].keys():
        if len(model_profile_config[message_key][model])>0:
            for profile in model_profile_config[message_key][model]:
                res = model_runner.run_model(model, profile, CAMP)
                if res == ModelScheduleRunResult.SCHEDULED:
                    print("Model run scheduled")
                elif res == ModelScheduleRunResult.CAPACITY:
                    print("Can not run model now, over capacity, try again later")
                elif res == ModelScheduleRunResult.ALREADY_RUNNING:
                    print("Already running")
                else:
                    raise RuntimeError("Unsupported result type: "+str(res))
        return None

## here we will need two buttons - one is simualtion the other is view results and the view results button look into model_result to see if that is ready to be shown

if __name__=='__main__':
    run_model_results_for_message("message_1")
