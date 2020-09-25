from seirsplus.models import *
from ai4good.models.nm.utils.network_utils import *
from ai4good.models.nm.utils.intervention_utils import *
import ai4good.models.nm.parameters.camp_params as cp


# Load graphs and process
def process_graph_sq(p, graph, nodes_per_struct):
    # Add 1 food queue
    graph = connect_food_queue(
        graph, nodes_per_struct, cp.food_weight, "food")

    # Create quarantine graph - This also includes neighbor/friendship edges
    quarantine_graph = remove_edges_from_graph(graph, scale=2, edge_label_list=[
                                               "food", "friendship"], min_num_edges=2)

    # Create interventions
    interventions = Interventions()

    # Simulate quarantine + masks
    q_start = 30
    interventions.add(quarantine_graph, q_start, beta=p.beta_q)

    # Simulate HALT of quarantine but people still have to wear masks
    q_end = 60
    interventions.add(graph, q_end, beta=p.beta_q)

    # Simulate HALT of wearing masks
    m_end = 90
    interventions.add(graph, m_end, beta=p.beta)

    checkpoints = interventions.get_checkpoints()

    # Model construction
    model = ExtSEIRSNetworkModel(G=graph, p=p.p_global_interaction, q=p.q_global_interactions,
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
