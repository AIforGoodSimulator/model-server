from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import SimpleParamStore
from ai4good.models.nm.parameters.initialise_parameters import Parameters
from ai4good.models.nm.models.nm_base_model import *
from ai4good.models.nm.models.nm_intervention_model_single_food_queue import *
from ai4good.models.nm.models.nm_intervention_multiple_food_queues import *
import logging


@typechecked
class NM(Model):
    ID = 'network-model'

    def __init__(self, ps=SimpleParamStore()):
        logging.info("Initialised network model")
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def run(self, p: Parameters) -> ModelResult:
        graph, nodes_per_struct = create_new_graph()

        p.initialise_age_parameters(graph)

        result_bm = process_graph_bm(p, graph, nodes_per_struct)
        result_sq = process_graph_sq(p, graph, nodes_per_struct)
        # result_mq = load_and_process_graph_mq([1, 2, 4])

        return ModelResult(self.result_id(p), {
            'params': p,
            'result_base_model': result_bm,
            'result_single_food_queue': result_sq
        })
