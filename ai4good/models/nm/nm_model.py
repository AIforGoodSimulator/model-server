from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.models.nm.models.nm_multiple_food_queues import *
from ai4good.models.nm.models.nm_multiple_food_queues_interventions import *
from ai4good.params.param_store import SimpleParamStore
from ai4good.models.nm.parameters.initialise_parameters import Parameters
from ai4good.models.nm.models.nm_baseline import *
from ai4good.models.nm.models.nm_baseline_interventions import *
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
        graph, nodes_per_struct = create_new_graph(p)
        logging.info("Running network model...")
        p.initialise_age_parameters(graph)

        # Baseline model, 1 food queue
        result_baseline = process_graph_bm(p, graph, nodes_per_struct)

        # Baseline model, 1 food queue, WITH interventions
        result_baseline_interventions = process_graph_bm_interventions(p, graph, nodes_per_struct)

        # Multiple food queuess model: 4, 8
        result_4_food_queues = process_graph_mq(p, graph, nodes_per_struct, 2)
        result_8_food_queues = process_graph_mq(p, graph, nodes_per_struct, 4)

        # Multiple food queuess model: 4, 8 WITH interventions
        result_4_food_queues_interventions = process_graph_mq_interventions(p, graph, nodes_per_struct, 2)
        result_8_food_queues_interventions = process_graph_mq_interventions(p, graph, nodes_per_struct, 4)

        return ModelResult(self.result_id(p), {
            'params': p,
            'result': result_baseline
        })
