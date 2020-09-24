from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import ParamStore
from ai4good.models.nm.parameters.initialise_parameters import Parameters
from ai4good.models.nm.models.nm_base_model import *
from ai4good.models.nm.models.nm_intervention_model_single_food_queue import *
from ai4good.models.nm.models.nm_intervention_multiple_food_queues import *


@typechecked
class NM(Model):
    ID = 'network-model'

    def __init__(self, ps: ParamStore):
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def run(self, p: Parameters) -> ModelResult:
        # create_new_graph()

        # For the alpha version, we will focus on loading graphs instead of creating them
        graph, nodes_per_struct = load_graph(f"../data/Moria_wNeighbors")
        p.initialise_age_parameters(graph)

        result_bm = process_graph_bm(p, graph, nodes_per_struct)
        result_sq = load_and_process_graph_sq()
        result_mq = load_and_process_graph_mq([1, 2, 4])

        return ModelResult(self.result_id(p), {
            'params': p,
            'result_base_model': result_bm,
            'result_single_food_queue': result_sq,
            'result_multiple_food_queue': result_mq
        })
