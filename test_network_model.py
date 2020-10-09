from ai4good.models.nm.nm_model import *
from ai4good.models.nm.parameters.initialise_parameters import Parameters

nm = NetworkModel()
# should be run for atleast 200 steps in prod
result = nm.run(p=Parameters(t_steps=5))

print(result.result_data['result_base_model'])
print(result.result_data['result_single_food_queue'])
