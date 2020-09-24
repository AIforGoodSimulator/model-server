from seirsplus.models import *
from ai4good.models.nm.utils.network_utils import *
import ai4good.models.nm.parameters.camp_params as cp


# Load graphs and process
def create_new_graph():
    # TODO: For now, this only creates a camp that resembles Moria, but it could be abstracted more (after alpha version is done)

    household_weight = 0.98  # Edge weight for connections within each structure
    graph, nodes_per_struct = create_graph(cp.n_structs, 0, cp.n_pop, cp.max_pop_per_struct,
                                           edge_weight=household_weight, label="household",
                                           age_list=list(cp.sample_pop["age"]),
                                           sex_list=list(cp.sample_pop["sex"]),
                                           n_ethnicities=cp.n_ethnic_groups)
    # Connect people from neighboring isoboxes
    graph = connect_neighbors(graph, 0, cp.n_isoboxes, nodes_per_struct,
                              cp.grid_isoboxes, cp.neighbor_proximity, cp.neighbor_weight, 'friendship')
    graph = connect_neighbors(graph, cp.dims_isoboxes[0] * cp.dims_isoboxes[1], cp.dims_block1[0] * cp.dims_block1[1],
                              nodes_per_struct,
                              cp.grid_block1, cp.neighbor_proximity, cp.neighbor_weight, 'friendship')
    graph = connect_neighbors(graph, cp.dims_block1[0] * cp.dims_block1[1], cp.dims_block2[0] * cp.dims_block2[1],
                              nodes_per_struct,
                              cp.grid_block2, cp.neighbor_proximity, cp.neighbor_weight, 'friendship')
    graph = connect_neighbors(graph, cp.dims_block2[0] * cp.dims_block2[1], cp.dims_block3[0] * cp.dims_block3[1],
                              nodes_per_struct,
                              cp.grid_block3, cp.neighbor_proximity, cp.neighbor_weight, 'friendship')
    return graph, nodes_per_struct


def process_graph_bm(p, graph, nodes_per_struct):
    # Base model with 1 food queue
    graph = connect_food_queue(
        graph, nodes_per_struct, cp.food_weight, "food")

    # Model construction
    model = ExtSEIRSNetworkModel(G=graph, p=p.p_global_interaction,
                                 beta=p.beta, sigma=p.sigma, lamda=p.lamda, gamma=p.gamma,
                                 gamma_asym=p.gamma, eta=p.eta, gamma_H=p.gamma_H, mu_H=p.mu_H,
                                 a=p.pct_asymptomatic, h=p.pct_hospitalized, f=p.pct_fatality,
                                 alpha=p.alpha, beta_pairwise_mode=p.beta_pairwise_mode,
                                 delta_pairwise_mode=p.delta_pairwise_mode,
                                 initE=p.init_exposed)

    # Run model
    node_states, simulation_results = run_simulation(model, p.t_steps)

    # Construct results dataframe
    output_df = results_to_df(simulation_results)

    return output_df
