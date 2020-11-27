# from seirsplus.models import *
from ai4good.models.nm.custom_models import *
from ai4good.models.nm.utils.network_utils import *
from ai4good.models.nm.initialise_parameters import Parameters


# Load graphs and process
def create_new_graph(p: Parameters):
    graph, nodes_per_struct = create_graph(p.n_structs, 0, p.total_population, p.max_pop_per_struct,
                                           edge_weight=p.household_weight, label="household",
                                           age_list=list(p.sample_pop["age"]),
                                           sex_list=list(p.sample_pop["sex"]),
                                           n_ethnicities=p.number_of_ethnic_groups)
    # Create node groups
    node_groups = create_node_groups(graph)

    # Connect people from neighboring isoboxes
    graph = link_nodes_by_property(graph, 0, p.n_isoboxes, nodes_per_struct,
                                   p.grid_isoboxes, p.neighbor_proximity, "ethnicity", p.neighbor_weight, 'friendship')
    graph = link_nodes_by_property(graph, p.dims_isoboxes[0] * p.dims_isoboxes[1], p.dims_block1[0] * p.dims_block1[1],
                                   nodes_per_struct,
                                   p.grid_block1, p.neighbor_proximity, "ethnicity", p.neighbor_weight, 'friendship')
    graph = link_nodes_by_property(graph, p.dims_block1[0] * p.dims_block1[1], p.dims_block2[0] * p.dims_block2[1],
                                   nodes_per_struct,
                                   p.grid_block2, p.neighbor_proximity, "ethnicity", p.neighbor_weight, 'friendship')
    graph = link_nodes_by_property(graph, p.dims_block2[0] * p.dims_block2[1], p.dims_block3[0] * p.dims_block3[1],
                                   nodes_per_struct,
                                   p.grid_block3, p.neighbor_proximity, "ethnicity", p.neighbor_weight, 'friendship')
    return graph, nodes_per_struct, node_groups


def process_graph(p, graph, node_groups):
    model = ExtSEIRSNetworkModel(G=graph, p=p.p_global_interaction,
                                 beta=p.beta, sigma=p.sigma, lamda=p.lamda, gamma=p.gamma,
                                 gamma_asym=p.gamma, eta=p.eta, gamma_H=p.gamma_H, mu_H=p.mu_H,
                                 a=p.pct_asymptomatic, h=p.pct_hospitalized, f=p.pct_fatality,
                                 alpha=p.alpha, beta_pairwise_mode=p.beta_pairwise_mode,
                                 delta_pairwise_mode=p.delta_pairwise_mode,
                                 initE=p.init_exposed, node_groups=node_groups)

    # Run model
    node_states, simulation_results = run_simulation(model, p.t_steps)
    # Construct results dataframe
    output_df = results_to_df(simulation_results)

    return output_df
