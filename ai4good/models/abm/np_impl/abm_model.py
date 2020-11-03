from typeguard import typechecked

from ai4good.models.abm.np_impl.moria import *
from ai4good.params.param_store import ParamStore
from ai4good.models.model import Model, ModelResult


@typechecked
class ABM(Model):

    ID = 'agent-based-model'

    def __init__(self, ps: ParamStore):
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def run(self, params: Parameters) -> ModelResult:
        # Run simulation

        if params.camp != "Moria":
            raise NotImplementedError("Simulation on {} camp is not implemented yet".format(params.camp))

        model = Moria(params=params, profile="")
        model.simulate()

        model_result = ModelResult(self.result_id(params), {
            "output_df": model.data_collector
        })

        # at end of simulation, return model result
        return model_result
