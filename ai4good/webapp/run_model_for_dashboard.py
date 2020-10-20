from ai4good.webapp.apps import model_runner
from ai4good.webapp.model_runner import ModelScheduleRunResult

# model name possibiltiies: ['compartmental-model', 'network-model', 'agent-based-model']


def run_model_results_for_message_1():
    model = 'compartmental-model'
    profiles = ['baseline','better_hygiene_one_month','better_hygiene_three_month','better_hygiene_six_month']
    camp = 'Moria' #this logic needs to be changed later where the user input params will be stored
    for profile in profiles:
        res = model_runner.run_model(model, profile, camp)
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
    run_model_results_for_message_1()
