from seirsplus.models import *
from ai4good.models.nm.utils.network_utils import *
from ai4good.models.nm.parameters.nm_model_parameters import *


# Load graphs and process
def create_new_graph():
    household_weight = 0.98  # Edge weight for connections within each structure
    graph, nodes_per_struct = create_graph(n_structs, 0, n_pop, max_pop_per_struct,
                                           edge_weight=household_weight, label="household",
                                           age_list=list(sample_pop["age"]),
                                           sex_list=list(sample_pop["sex"]),
                                           n_ethnicities=n_ethnic_groups)
    save_graph(graph, nodes_per_struct, f"experiments/networks/Moria_wNeighbors")


def load_and_process_graph_bm():
    graph, nodes_per_struct = load_graph(f"experiments/networks/Moria_wNeighbors")

    # Base model with 1 food queue
    food_weight = 0.407
    graph = connect_food_queue(graph, nodes_per_struct, food_weight, "food")

    # Iterate through the tweakable parameters
    param_combo_i = 0
    result = list()
    for transmission_rate, progression_rate, recovery_rate, hosp_rate, death_rate in zip(transmission_rate,
                                                                                         progression_rate,
                                                                                         recovery_rate_list,
                                                                                         hosp_rate_list,
                                                                                         death_rate_list):
        # Model construction
        model = ExtSEIRSNetworkModel(G=graph_1fq, p=prob_global_contact, beta=transmission_rate, sigma=progression_rate,
                                     gamma=recovery_rate, lamda=progression_rate, mu_H=crit_rate, eta=hosp_rate,
                                     a=prob_asymptomatic, f=death_rate, h=prob_symp_to_hosp, initI_sym=init_symp_cases,
                                     initI_asym=init_asymp_cases, store_Xseries=True)

        # Run model
        node_states, simulation_results = run_simulation(model, t_steps)

        # Model name for storage + store the model params in csv
        fig_name = f"BaseSympModel_{param_combo_i}"
        add_model_name("experiments/model_names.csv", fig_name, household_weight, neighbor_weight,
                       food_weight,
                       transmission_rate, recovery_rate, progression_rate, hosp_rate,
                       round(sum(crit_rate) / len(crit_rate), 3), death_rate, init_symp_cases,
                       init_asymp_cases,
                       t_steps)

        # Construct results dataframe
        output_df = results_to_df(simulation_results, store=True,
                                  store_name=f"experiments/results/{fig_name}.csv")
        result.append(output_df)

        # Plot and store
        fig, ax = model.figure_basic(show=False)  # vlines=interventions.get_checkpoints()['t'])
        fig.savefig(f"plots/{fig_name}_fig.png")

        param_combo_i += 1

    return output_df
