import hashlib
from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import ParamStore
from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.models.cm.functions import Simulator


@typechecked
class CompartmentalModel(Model):
    RID_PATTERN = "Camp=%s_%shygieneT=%s_remInfRate=%s_remInfT=%s_Shield=%s_RemHrRate=%s_RemHrTime=%s_ICU=%s_NumIts=%s"
    ID = 'compartmental-model'

    def __init__(self, ps: ParamStore):
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()
        # return self.RID_PATTERN % (
        #     p.camp,
        #     p.control_dict['better_hygiene']['value'],
        #     p.control_dict['better_hygiene']['timing'],
        #     ceil(p.population * p.control_dict['remove_symptomatic']['rate']),
        #     p.control_dict['remove_symptomatic']['timing'],
        #     p.control_dict['shielding']['used'],
        #     ceil(p.population * p.control_dict['remove_high_risk']['rate']),
        #     p.control_dict['remove_high_risk']['timing'],
        #     ceil(p.population * p.control_dict['ICU_capacity']['value']),
        #     p.control_dict['numberOfIterations']
        # )

    def run(self, p: Parameters) -> ModelResult:
        sim = Simulator(p)
        sols_raw, standard_sol, percentiles, config_dict = sim.simulate_over_parameter_range_parallel(
            p.control_dict['numberOfIterations'], p.control_dict['t_sim'],  p.control_dict['nProcesses'])
        return ModelResult(self.result_id(p), {
            'sols_raw': sols_raw,
            'standard_sol': standard_sol,
            'percentiles': percentiles,
            'config_dict': config_dict,
            'params': p
        })
