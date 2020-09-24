import networkx as nx
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle as pkl

STATE_DICTIONARY = {
    "Susceptible": 1,
    "Exposed": 2,
    "Infectious_Presymptomatic": 3,
    "Infectious_Symptomatic": 4,
    "Infectious_Asymptomatic": 5,
    "Hospitalized": 6,
    "Recovered": 7,
    "Deceased": 8,
    "Detected_Exposed": 9,
    "Detected_Presymptomatic": 10,
    "Detected_Symptomatic": 11,
    "Detected_Asymptomatic": 12}


####### NETWORK CREATION UTILS! ################
def create_graph(
        n_structures,
        start_idx,
        population,
        max_pop_per_struct,
        **kwargs):
    """ Creates a networkX graph containing all the population in the camp that is in a given structure (currently just isoboxes).
        Draws edges between people from the same isobox and returns the networkX graph and an adjacency list
    """

    # Graph is a networkX graph object
    g = nx.Graph()

    # Keep track of how many nodes we have put in an isobox already
    struct_count = np.zeros(shape=n_structures)

    # Store the indices of the nodes we store in each isobox in a 2D array
    # where array[i] contains the nodes in isobox i
    nodes_per_struct = [[] for i in range(n_structures)]

    available_structs = list(range(n_structures))

    # Add the nodes to the graph
    for node in tqdm(range(start_idx, population)):
        g.add_node(node)

        # Assign nodes to isoboxes randomly, until we reach the capacity of
        # that isobox
        struct_num = np.random.choice(available_structs)

        # Assign properties to nodes
        g.nodes[node]["age"] = kwargs["age_list"][node]
        g.nodes[node]["sex"] = kwargs["sex_list"][node]
        g.nodes[node]["location"] = struct_num
        g.nodes[node]["ethnicity"] = np.random.choice(
            range(kwargs["n_ethnicities"]))

        # Update number of nodes per isobox and which nodes were added to
        # iso_num
        struct_count[struct_num] += 1
        nodes_per_struct[struct_num].append(node)

        if struct_count[struct_num] > max_pop_per_struct[struct_num]:
            available_structs.remove(struct_num)

    # Now we connect nodes inside of the same isobox
    for node_list in nodes_per_struct:
        # Use the cartesian product to get all possible edges within the nodes in an isobox
        # and only add if they are not the same node
        edge_list = [
            tup for tup in list(
                itertools.product(
                    node_list,
                    repeat=2)) if tup[0] != tup[1]]
        g.add_edges_from(
            edge_list,
            weight=kwargs["edge_weight"],
            label=kwargs["label"])

    return g, nodes_per_struct


def remove_edges_from_graph(base_graph, edge_label_list, scale, min_num_edges):
    """ Randomly remove some of the edges that have a label included in the label list (i.e 'food', 'neighbors', etc)
        The scale is a parameter for the exponential distribution and the min_num_edges the minimum number of edges
        a node should keep (this will be the peak of the distribution) """

    graph = base_graph.copy()

    for node in graph:
        # Select all the neighbors that share an edge of a particular label
        neighbors = [neighbor for neighbor in list(graph[node].keys())
                     if graph.edges[node, neighbor]["label"] in edge_label_list]

        # If there are no neighbors that have a label from edge_label_list, we
        # continue
        if neighbors:
            # Randomly draw the number of edges to keep
            quarantine_edge_num = int(max(min(np.random.exponential(
                scale=scale, size=1), len(neighbors)), min_num_edges))

            if quarantine_edge_num <= len(neighbors):
                # Create the list of neighbors to keep
                quarantine_keep_neighbors = np.random.choice(
                    neighbors, size=quarantine_edge_num, replace=True)

                # Remove edges that are not in te list of neighbors to keep
                for neighbor in neighbors:
                    if neighbor not in quarantine_keep_neighbors:
                        graph.remove_edge(node, neighbor)

    return graph


def remove_all_edges(base_graph, edge_label_list):
    graph = base_graph.copy()
    for node in graph:
        neighbors = [neighbor for neighbor in list(graph[node].keys())
                     if graph.edges[node, neighbor]["label"] in edge_label_list]

        for neighbor in neighbors:
            graph.remove_edge(node, neighbor)

    return graph


def divide_grid(grid, n_slices):
    """ Returns a list of evenly sized blocks of the camp grid """
    return np.array_split(grid, n_slices, axis=1)


def create_grid(width, height, starting_n):
    """ Create a grid of isoboxes that resembles the isobox area of the camp, for ease of measuring proximity
        between nodes. Returns a numpy array of shape (width, height) """

    grid = np.zeros(shape=(width, height)).astype(int)
    n = starting_n

    for i in range(width):
        for j in range(height):
            grid[i][j] = n
            n += 1

    return grid


def get_neighbors(grid, structure_num, proximity):
    """ Given a grid of structures, returns the closest proximity neighbors to the given structure

        params:
        - Grid: 2D numpy array
        - structure_num: int
        - proximity: int

        :returns
        - A list of neighboring structures to the current structure_num
    """

    # Get the number of columns for ease of access
    width = len(grid)
    height = len(grid[0])

    # We'll make it a set initially to avoid duplicate neighbors
    neighbors = set()

    for i in range(-proximity, proximity + 1):
        for j in range(-proximity, proximity + 1):
            if not (i == 0 and j == 0):
                x = min(max((structure_num // height) - i, 0), width - 1)
                y = min(max((structure_num % height) - j, 0), height - 1)

                if grid[x][y] != structure_num:
                    neighbors.add(grid[x][y])

    return list(neighbors)


def connect_neighbors(
        base_graph,
        start_idx,
        n_structures,
        nodes_per_structure,
        grid,
        proximity,
        edge_weight,
        label):
    """ Draw edges in the given graph between people of neighboring structures (currently isoboxes)
        f they have the same ethnicity """

    graph = base_graph.copy()

    # For every possible structure:
    for structure in range(start_idx, n_structures + start_idx):

        # Given an isobox number get its neighbor isoboxes
        neighbors = get_neighbors(grid, structure, proximity)

        # For every neighbor isobox:
        for neighbor in neighbors:
            # If they share the same properties, draw an edge between them
            graph.add_edges_from([(i, j) for i in nodes_per_structure[structure]
                                  for j in nodes_per_structure[neighbor] if
                                  graph.nodes[i]["ethnicity"] == graph.nodes[j]["ethnicity"]],
                                 weight=edge_weight, label=label)

    return graph


def connect_food_queue(base_graph, nodes_per_structure, edge_weight, label):
    """ Connect 1-2 people per structure (currently just isoboxes) randomly to represent that they go to the food queue
        We have 3 options:
            - Either have a range of people (2-5 per isobox) that go to food queue, same edge weights
            - Connect all people in the food queue, same edge weights
            - Connect all people in food queue with different edge weights
    """

    graph = base_graph.copy()

    food_bois = set()

    # Choose half of the people randomly from each structure
    for node_list in nodes_per_structure:
        for i in range(len(node_list) // 2):
            food_bois.add(np.random.choice(node_list))

    # This list represents the food queue
    food_bois = list(food_bois)
    np.random.shuffle(food_bois)

    # Draw an edge between everyone in the list in order, since we have
    # already shuffled them
    for i in range(len(food_bois) - 6):
        for j in range(i + 1, i + 6):
            if not graph.has_edge(food_bois[i], food_bois[j]):
                graph.add_edge(
                    food_bois[i],
                    food_bois[j],
                    weight=edge_weight,
                    label=label)
    return graph


def create_multiple_food_queues(base_graph, n_food_queues_per_block, food_weight, nodes_per_struct, grids):
    graph = base_graph.copy()
    queue_num = 0

    for grid in grids:
        longest_axis = np.argmax(grid.shape)
        index_limit = grid.shape[longest_axis] // n_food_queues_per_block
        subgrids = []
        if not longest_axis:
            for i in range(n_food_queues_per_block):
                subgrids.append(grid[i * index_limit:(i + 1) * index_limit, :])
        else:
            for i in range(n_food_queues_per_block):
                subgrids.append(grid[:, i * index_limit:(i + 1) * index_limit])

        for i in range(len(subgrids)):
            subgrid = subgrids[i]
            nodes_per_struct_subgrid = [nodes_per_struct[subgrid[i][j]] for i in range(len(subgrid)) for j in
                                        range(len(subgrid[i]))]

            graph = connect_food_queue(graph, nodes_per_struct_subgrid, food_weight, f"food_{queue_num}")
            queue_num += 1

    return graph


def create_node_groups(graph):
    """
    create node groups for each 10-year age bucket so the main simulation can track the results for people in each age bucket
    """
    AGE_BUCKET = 9
    graph_data = list(graph.nodes(data='age'))
    node_groups = {}
    for age in range(AGE_BUCKET):
        nodeList = []
        if age == 8:
            groupName = f'age>{age * 10}'
        else:
            groupName = f'age{age * 10}-{(age + 1) * 10}'
        for node in graph_data:
            if age == 8:
                if node[1] >= age * 10:
                    nodeList.append(node[0])
            else:
                if node[1] >= age * 10 and node[1] < (age + 1) * 10:
                    nodeList.append(node[0])
        node_groups[groupName] = nodeList
    return node_groups


####### MODEL UTILS! ################

def run_simulation(model, t, checkpoints=None, simulation_results=None, nodes_states=None, print_info=False,
                   store_every=1):
    if not simulation_results:
        node_states = dict()
        simulation_results = defaultdict(list)

    print(f"Running simulation for {t} steps...\n")

    for i in tqdm(range(1, t + 1)):
        model.run(T=1, verbose=print_info, checkpoints=checkpoints)

        if i % store_every == 0:
            # Store the node states - an array of size (1, num_nodes) -> we
            # store a copy because the array gets updated
            node_states[i] = np.copy(model.X.T)

            # Store the quantities of the last time step t
            simulation_results["Susceptible"].append(model.numS[-1])
            simulation_results["Exposed"].append(model.numE[-1])
            simulation_results["Infected_Presymptomatic"].append(
                model.numI_pre[-1])
            simulation_results["Infected_Symptomatic"].append(model.numI_sym[-1])
            simulation_results["Infected_Asymptomatic"].append(
                model.numI_asym[-1])
            simulation_results["Hospitalized"].append(model.numH[-1])
            simulation_results["Recovered"].append(model.numR[-1])
            simulation_results["Fatalities"].append(model.numF[-1])
            simulation_results["T_index"].append(model.tidx)

    return node_states, simulation_results


def find_whole_time(tseries, T):
    idt = []
    for t in range(0, T + 1):
        idt.append(np.abs(tseries - t).argmin())
    return idt


def run_simulation_test(model, t, print_info=False):
    node_states = dict()
    simulation_results = defaultdict(list)

    print(f"Running simulation for {t} steps...\n")
    model.run(T=t, verbose=print_info)
    # Store the quantities of the time step closest to the integer
    time_stamps = model.tseries
    idt = find_whole_time(time_stamps, t)
    # Store the quantities of each integer time steps
    simulation_results["Susceptible"] = model.numS[idt]
    simulation_results["Exposed"] = model.numE[idt]
    simulation_results["Infected_Presymptomatic"] = model.numI_pre[idt]
    simulation_results["Infected_Symptomatic"] = model.numI_S[idt]
    simulation_results["Infected_Asymptomatic"] = model.numI_A[idt]
    simulation_results["Hospitalized"] = model.numH[idt]
    simulation_results["Recovered"] = model.numR[idt]
    simulation_results["Fatalities"] = model.numF[idt]
    simulation_results["Detected_Presymptomatic"] = model.numD_pre[idt]
    simulation_results["Detected_Symptomatic"] = model.numD_S[idt]
    simulation_results["Detected_Asymptomatic"] = model.numD_A[idt]

    return simulation_results


def results_to_df(simulation_results, store=False, store_name=None):
    """ Convers the simulation results to a dataframe, adding a "timestep" column to it
        Returns a dataframe with the information requested above, and stores it if store=True """

    simulation_results["Time"] = list(
        range(1, len(simulation_results["T_index"]) + 1))
    output = pd.DataFrame(simulation_results)

    if store:
        output.to_csv(store_name)

    return output


def add_model_name(name_csv, fig_name,
                   household_weight, neighbor_weight, food_weight,
                   transmission_rate, recovery_rate, progression_rate,
                   hosp_rate, crit_rate, death_rate, init_symp_cases,
                   init_asymp_cases, t_steps, q_time="", q_red="", h_time=""):
    """ Adds the parameters of model fig_name to name_csv """

    name_df = pd.read_csv(name_csv)

    # Add a new model name + parameters as a new row at the end of the df
    idx = 0 if pd.isnull(name_df.index.max()) else name_df.index.max() + 1
    name_df.loc[idx] = [fig_name, household_weight, neighbor_weight, food_weight, transmission_rate, recovery_rate,
                        progression_rate, hosp_rate, crit_rate, death_rate, init_symp_cases, init_asymp_cases, t_steps,
                        q_time, q_red, h_time]

    # Store as csv again
    name_df.to_csv(name_csv)


###########################################

# Some helper functions - all of which could be added to networkx class!
def min_degree(graph):
    """ Return the minimum degree in the graph """
    return min(graph.degree, key=lambda kv: kv[1])[1]


def max_degree(graph):
    """ Return the maximum degree in the graph """
    return max(graph.degree, key=lambda kv: kv[1])[1]


def get_nodes_per_state(X, graph, state):
    """ Get the nodes that have a given state in the latest timestep of the SEIRS+ model
        Since nodes are represented by their properties in this case, this will return a list of property dicts """
    return [node for node in graph.nodes if X[node] == state]


def save_graph(graph, nodes_per_struct, name):
    with open(name + ".graph", "wb") as f:
        pkl.dump(graph, f)

    with open(name + ".nps", "wb") as f:
        pkl.dump(nodes_per_struct, f)


def load_graph(name):
    with open(name + ".graph", "rb") as f:
        graph = pkl.load(f)

    with open(name + ".nps", "rb") as f:
        nodes_per_struct = pkl.load(f)

    return graph, nodes_per_struct
