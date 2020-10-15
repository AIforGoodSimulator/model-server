from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.models.nm.models.nm_multiple_food_queues import *
from ai4good.models.nm.models.nm_multiple_food_queues_interventions import *
from ai4good.params.param_store import SimpleParamStore
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

        if p.profile_name == 'baseline':
            # Baseline model, 1 food queue
            result = process_graph_bm(p, graph, nodes_per_struct)
        elif p.profile_name == 'interventions':
            # Baseline model, 1 food queue, WITH interventions
            result = process_graph_bm_interventions(p, graph, nodes_per_struct)
        elif p.profile_name == '4_food_queues':
            # Multiple food queues model: 4
            result = process_graph_mq(p, graph, nodes_per_struct, 2)
        elif p.profile_name == '8_food_queues':
            # Multiple food queues model: 8
            result = process_graph_mq(p, graph, nodes_per_struct, 4)
        elif p.profile_name == 'interventions_4_food_queues':
            # Multiple food queues model: 4 WITH interventions
            result = process_graph_mq_interventions(p, graph, nodes_per_struct, 2)
        elif p.profile_name == 'interventions_8_food_queues':
            # Multiple food queues model: 8 WITH interventions
            result = process_graph_mq_interventions(p, graph, nodes_per_struct, 4)
        else:
            # Run Baseline Model
            print('No matching profile found. Running Baseline model, 1 food queue')
            result = process_graph_bm(p, graph, nodes_per_struct)

        return ModelResult(self.result_id(p), {
            'params': p,
            'result': result
        })
