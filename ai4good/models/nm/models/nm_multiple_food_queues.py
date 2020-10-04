from seirsplus.models import *
from ai4good.models.nm.utils.network_utils import *


def process_graph_mq(p, graph, nodes_per_struct, food_queue_number):
    # Add multiple food queues
    graph = create_multiple_food_queues(graph, food_queue_number, p.food_weight, nodes_per_struct,
                                        [p.grid_isoboxes, p.grid_block1, p.grid_block2, p.grid_block3])

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
