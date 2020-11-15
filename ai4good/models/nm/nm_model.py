from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import SimpleParamStore
from ai4good.models.nm.models.nm_baseline import *
from ai4good.models.nm.models.nm_interventions import *
import logging
from ai4good.utils.logger_util import get_logger
import dask
from dask.diagnostics import ProgressBar
logger = get_logger(__name__)


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
        logging.info("Generating network graph with neighbors...")
        graph, nodes_per_struct = create_new_graph(p)

        logging.info("Adding food queues...")
        if '4_food_queues' in p.profile_name:  # Multiple food queues model: 4
            graph = create_multiple_food_queues(graph, 2, p.food_weight, nodes_per_struct,
                                                [p.grid_isoboxes, p.grid_block1, p.grid_block2, p.grid_block3])

        elif '8_food_queues' in p.profile_name:  # Multiple food queues model: 8
            graph = create_multiple_food_queues(graph, 4, p.food_weight, nodes_per_struct,
                                                [p.grid_isoboxes, p.grid_block1, p.grid_block2, p.grid_block3])

        else:  # Any other model: 1 food queue
            graph = link_nodes_by_activity(
                graph, nodes_per_struct, percentage_per_struct=0.5, proximity_radius=5,
                edge_weight=p.food_weight, activity_name="food")

        logging.info("Sampling from original graph...")
        new_graph_size = len(graph.nodes) // 4
        sampled_graph = downsample_graph(graph, new_graph_size, "uniform")
        p.update_parameters(sampled_graph)
        p.initialise_age_parameters(sampled_graph)

        logging.info("Running network model...")
        results = run_parallel(p, sampled_graph)
        
        return ModelResult(self.result_id(p), {
            'params': p,
            'result': aggregate_results(results)
        })


def run_parallel(p, sampled_graph):
    lazy_sols = []
    for ii in range(10):
        if "interventions" in p.profile_name:
                lazy_result = dask.delayed(process_graph_interventions)(p, sampled_graph)
        else:
            lazy_result = dask.delayed(process_graph)(p, sampled_graph)
        
        lazy_sols.append(lazy_result)
        
    with dask.config.set(scheduler='single-threaded', num_workers=1):
        with ProgressBar():
            sols = dask.compute(*lazy_sols)
    return sols


def aggregate_results(results):
    return pd.concat(results).groupby(level=0).mean()