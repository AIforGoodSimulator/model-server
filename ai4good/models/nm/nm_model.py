from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import SimpleParamStore
from ai4good.models.nm.parameters.initialise_parameters import Parameters
from ai4good.models.nm.models.nm_base_model import *
from ai4good.models.nm.models.nm_intervention_model_single_food_queue import *
import logging


@typechecked
class NetworkModel(Model):
    ID = 'network-model'

    def __init__(self, ps=SimpleParamStore()):
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def run(self, p: Parameters) -> ModelResult:
        logging.info("Generating network graph...")
        graph, nodes_per_struct = create_new_graph()
        logging.info("Running network model...")
        p.initialise_age_parameters(graph)

        result_bm = process_graph_bm(p, graph, nodes_per_struct)
        result_sq = process_graph_sq(p, graph, nodes_per_struct)

        return ModelResult(self.result_id(p), {
            'params': p,
            'result_base_model': result_bm,
            'result_single_fq': result_sq
        })
