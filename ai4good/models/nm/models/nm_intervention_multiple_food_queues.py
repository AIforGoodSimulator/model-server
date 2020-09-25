from seirsplus.models import *
from ai4good.models.nm.utils.network_utils import *
from ai4good.models.nm.utils.intervention_utils import *
from ai4good.models.nm.parameters.nm_model_parameters import *


# Load graphs and process
def load_and_process_graph_mq(food_queues):
    base_graph, nodes_per_struct = load_graph(f"experiments/networks/Moria_wNeighbors")

    result = dict()

    for food_queue_number in food_queues:

        # Base model with 1 food queue
        food_weight = 0.407
        graph = create_multiple_food_queues(base_graph, food_queue_number, food_weight, nodes_per_struct,
                                            [grid_isoboxes, grid_block1, grid_block2, grid_block3])

        # Create quarantine graph - This also includes neighbor/friendship edges
        quarantine_graph = remove_edges_from_graph(graph, scale=2, edge_label_list=["food", "friendship"],
                                                   min_num_edges=2)

        # Create interventions
        interventions = Interventions()

        # Amount by which wearing masks reduces the transmission rate
        reduction_percentage = 0.3

        # Simulate quarantine + masks
        q_start = 0
        interventions.add(quarantine_graph, q_start, beta=transmission_rate * reduction_percentage)

        # Simulate HALT of quarantine but people still have to wear masks
        q_end = 150
        interventions.add(graph, q_end, beta=transmission_rate * reduction_percentage)

        # Simulate HALT of wearing masks
        m_end = 200
        interventions.add(graph, m_end, beta=transmission_rate)

        checkpoints = interventions.get_checkpoints()

        result_queue = list()

        for transmission_rate, progression_rate, recovery_rate, hosp_rate, death_rate in zip(transmission_rate,
                                                                                             progression_rate,
                                                                                             recovery_rate_list,
                                                                                             hosp_rate_list,
                                                                                             death_rate_list):
            # Model construction
            model = ExtSEIRSNetworkModel(G=graph, Q=quarantine_graph, p=prob_global_contact,
                                         q=prob_quarantine_global_contact, beta=transmission_rate,
                                         sigma=progression_rate, gamma=recovery_rate, lamda=progression_rate,
                                         mu_H=crit_rate, eta=hosp_rate, a=prob_asymptomatic, f=death_rate,
                                         h=prob_symp_to_hosp, initI_sym=init_symp_cases, initI_asym=init_asymp_cases,
                                         store_Xseries=True)

            # Run model
            node_states, simulation_results = run_simulation(model, t_steps, checkpoints)

            # Model name for storage
            fig_name = f"InterventionsMultFQ{food_queue_number}_Model"
            add_model_name("experiments/model_names.csv", fig_name, household_weight, neighbor_weight,
                           food_weight, transmission_rate, recovery_rate, progression_rate, hosp_rate,
                           round(sum(crit_rate) / len(crit_rate), 3), death_rate, init_symp_cases,
                           init_asymp_cases, t_steps, f"{q_start}-{q_end}", reduction_percentage,
                           f"{q_start}-{m_end}")

            # Construct results dataframe
            output_df = results_to_df(simulation_results, store=True,
                                      store_name=f"experiments/results/{fig_name}.csv")

            result_queue.append(output_df)

            # Plot and store
            fig, ax = model.figure_basic(show=False)  # vlines=interventions.get_checkpoints()['t'])
            fig.savefig(f"experiments/plots/{fig_name}_fig.png")

        result[food_queue_number] = result_queue

    return result
