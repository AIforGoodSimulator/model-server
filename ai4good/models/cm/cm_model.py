from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import ParamStore
from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.models.cm.functions import Simulator
from ai4good.models.cm.functions import generate_csv


@typechecked
class CompartmentalModel(Model):

    ID = 'compartmental-model'

    def __init__(self, ps: ParamStore):
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def run(self, p: Parameters) -> ModelResult:
        sim = Simulator(p)
        sols_raw, standard_sol, percentiles, config_dict = sim.simulate_over_parameter_range_parallel(
            p.control_dict['numberOfIterations'], p.control_dict['t_sim'],  p.control_dict['nProcesses'])
        report = generate_csv(sols_raw, p, input_type='raw')
        return ModelResult(self.result_id(p), {
            'sols_raw': sols_raw,
            'standard_sol': standard_sol,
            'percentiles': percentiles,
            'config_dict': config_dict,
            'params': p,
            'report': report
        })


