# * This is a file containing the main helper functions used in the project *

# -- Imports -- ## 

import os
import random
import time
import scipy

import numpy as np
import pandas as pd

import networkx as nx
import nx_parallel as nxp
import pickle

from scipy.sparse import csr_matrix

# Modify eliorc's implementation
from eliorc_mod.node2vec import Node2Vec
from gensim.models import Word2Vec

import itertools
from collections import Counter

from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score #, accuracy_score


# -- Functions -- ## 

# Need to change max value to a cluster number or something
def generate_random_number(min_value=0, max_value=4):
    return random.randint(min_value, max_value)


def generate_execution_timestamp():
    '''
        Generates execution timestamp
    '''
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as required
    formatted_datetime = current_datetime.strftime("%d/%m/%Y - %H:%M")
    # formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    # Return the formatted date and time
    return formatted_datetime


def mat_load_func(filepath):
    '''
        Takes .mat file as input and exports to 
        networkx graph and group DataFrame
    '''

    data = scipy.io.loadmat(filepath)

    # Load initial data
    network_sparse = data['network']
    initial_graph = nx.from_scipy_sparse_array(network_sparse)

    # Extract non-zero elements (node indices and labels) from the sparse matrix
    node_indices, label_indices = data['group'].nonzero()

    # Create a DataFrame with node numbers and their respective labels
    group_df = pd.DataFrame({'node_num': node_indices, 'group': label_indices})

    return initial_graph, group_df



def model_gen(graph, params, existing_filename=None, quiet_bool=True):
    '''
        Generates and saves model, returns node2vec and model fit
    '''

    if existing_filename:
        model = Word2Vec.load(existing_filename)
        print("Model loaded.")

        return '', model

    [d, r, l, p, q] = params

    node2vec = Node2Vec(graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=64, temp_folder='temp_folder', quiet=quiet_bool)  # Use temp_folder for big graphs

    # Embed nodes (check arguments, maybe unneccessary?)
    model = node2vec.fit(window=10, min_count=1, batch_words=4) #, ns_exponent=1)
    
    # # Save model
    # model_filename = f"{graph.name}_d{d}_r{r}_l{l}_p{p}_q{q}.mdl"
    # model.save(f"./models/{model_filename}")

    N, E = [graph.number_of_nodes(), graph.number_of_edges()]

    print(f"Model generated - (|V| = {N} , |E| = {E})")

    return node2vec, model


def param_dict_gen(params, output_list):
    '''
        Generates a dictionary of parameters and the output list
        from frequency and degree distributions
    '''

    [d, r, l, p, q] = params
    key = f"d{d}_r{r}_l{l}_p{p}_q{q}"

    param_dict = {key: output_list}

    return param_dict


# Check on how to improve this (select only filename or walks)
def freq_gen(walks_filename=None, walks=None):
    '''
        Generates the cumulative probabilities used in the plot
    '''

    if walks_filename != None:
        # Open the file in binary read mode
        with open(f"walks/{walks_filename}", "rb") as file:
            # Load the data from the file using pickle.load
            walks = pickle.load(file)


    # Flatten the list of lists into a single list
    node_freq_list = [item for sublist in walks for item in sublist]

    # Get the value counts using Counter
    node_freq_dict = dict(Counter(node_freq_list))

    visit_counts = np.array(list(node_freq_dict.values()))
    visit_counts.sort()

    cumulative_visit_freq = np.cumsum(visit_counts[::-1])[::-1]
    cumulative_freq_prob = cumulative_visit_freq / cumulative_visit_freq[0]

    return node_freq_dict, visit_counts, cumulative_freq_prob


def degrees_distribution_gen(graph):
    '''
        Generates degrees distribution used for plot
    '''

    # Calculate degree distribution
    degrees = sorted([graph.degree(i) for i in graph], reverse=True)
    degree_counts = np.bincount(degrees)

    # Calculate probability that the degree of a node is at least x
    total_nodes = len(degrees)
    cumulative_deg_prob = np.cumsum(degree_counts[::-1])[::-1] / total_nodes

    return degrees, cumulative_deg_prob


# Refactor this function
def groups_assign(initial_graph, subgraph, group_df=pd.DataFrame()):
    '''
        Assign group numbers to nodes for node classification
    '''

    if group_df.shape == (0, 0):

        if subgraph != initial_graph:
            _, new_graph = connect_subgraph(initial_graph, subgraph)
        else:
            new_graph = initial_graph

        # Create an empty DataFrame with the same length as the list
        group_df = pd.DataFrame(index=range(new_graph.number_of_nodes()))

        # Assign the original list to the first column
        group_df['node_num'] = group_df.index

        # Generate random numbers (you can use the function or directly use random.randint)
        random_numbers = [generate_random_number(min_value=0, max_value=4) for _ in range(len(group_df))]

        # Add a new column with the random numbers
        group_df['group'] = random_numbers

    else:
        temp_df = pd.DataFrame(index=sorted(subgraph.nodes()))

        temp_df['node_num'] = temp_df.index

        random_numbers = [generate_random_number() for _ in range(len(temp_df))]

        temp_df['group'] = random_numbers

        group_df = pd.concat([group_df, temp_df], ignore_index=True)  # Optional: reset index

    groups_dict = {int(node_num): group_df.loc[group_df['node_num'] == node_num, 'group'].values[0] for node_num in list(group_df['node_num'])}

    return groups_dict


def emb_group_gen(groups_dict, model):
    '''
        Generates embeddings with their corresponding groups
    '''

    # Create a dictionary to map node IDs to vectors
    node_vectors_dict = {}
    for node_id in model.wv.index_to_key:
        node_vectors_dict[int(node_id)] = model.wv.get_vector(node_id)

    X = list(node_vectors_dict.values())
    y = [groups_dict[int(key)] for key in node_vectors_dict if key in groups_dict] # the mapping keeps the same order

    return X, y, node_vectors_dict # remove in future


def ovr_classifier(X, y, test_size):
    '''
        Outputs an OvR classifier
    '''

    f1_scores = dict()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Encode class labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Train one-vs-rest logistic regression models
    ovr_logistic_models = []
    for class_label in label_encoder.classes_:
        # Create a binary classification task for each class
        y_train_binary = (y_train == class_label).astype(int)
        
        # Train logistic regression model with L2 regularization
        ovr_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
        ovr_model.fit(X_train, y_train_binary)
        ovr_logistic_models.append(ovr_model)

    # Predict probabilities for each class
    y_pred_prob = []
    for ovr_model in ovr_logistic_models:
        y_pred_prob.append(ovr_model.predict_proba(X_test)[:, 1])

    # Choose the class with the highest probability as the predicted class
    y_pred = label_encoder.inverse_transform([np.argmax(pred) for pred in np.array(y_pred_prob).T])

    # # Calculate accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    
    # F1 Micro Score
    f1_micro = f1_score(y_test, y_pred, average='micro')
    
    # F1 Macro Score
    f1_macro = f1_score(y_test, y_pred, average='macro')

    training_size = 1-test_size

    f1_scores[float(f"{training_size:.1f}")] = [f1_micro, f1_macro]

    return f1_scores


def node2vec_configs():
    '''
        Generates list of configs
    '''
    # D_values = [128]
    # R_values = [40, 80] 
    # L_values = [80, 160]
    # P_values = [0.25, 0.5, 1, 2, 4]
    # Q_values = [0.25, 0.5, 1, 2, 4] 


    # D_values = [128]
    # R_values = [40, 80] 
    # L_values = [80]
    # P_values = [0.25, 1, 2, 4]
    # Q_values = [1, 2, 4] 

    D_values = [128]
    R_values = [40, 80] 
    L_values = [80]
    P_values = [0.25]
    Q_values = [1] 

    # Generate all possible combinations of orders and seasonal orders
    parameter_values = [D_values, R_values, L_values, P_values, Q_values]
    parameter_combinations = list(itertools.product(*parameter_values))

    return parameter_combinations


# def node2vec_configs():
#     '''
#         Generates list of configs
#     '''
#     D_values = [64, 128]
#     R_values = [10, 20, 40, 80] 
#     L_values = [80, 160, 240]
#     P_values = [0.25, 0.5, 1, 2, 4]
#     Q_values = [0.25, 0.5, 1, 2, 4] 

#     # Generate all possible combinations of orders and seasonal orders
#     parameter_values = [D_values, R_values, L_values, P_values, Q_values]
#     parameter_combinations = list(itertools.product(*parameter_values))

#     return parameter_combinations


# Could extend by adjusting k 
def connect_subgraph(initial_graph, subgraph):
    '''
        Connects a subgraph to a main graph by adding an edge between a
        node from the main graph and a node from the subgraph
    '''

    new_graph = initial_graph.copy()

    # Pick random nodes to connect the two graphs
    node_main = random.choice(list(new_graph.nodes()))
    node_sub = random.choice(list(subgraph.nodes()))
    
    # Add an edge between the chosen nodes
    new_graph.add_edge(node_main, node_sub)
    
    # Add the subgraph to the new graph
    new_graph.add_nodes_from(subgraph.nodes(data=True))
    new_graph.add_edges_from(subgraph.edges())

    return [node_main, node_sub], new_graph



def get_neighborhood(initial_graph, initial_node, max_steps):
    """
    Finds the neighborhood of a node in a graph within a specified hop limit.
    
    Parameters:
    initial_graph (Graph): The input graph.
    initial_node (Node): The initial node.
    max_steps (int): The maximum number of hops.
    
    Returns:
    Graph: The subgraph induced by the neighborhood.
    """
    # Initialize sets for current neighbors and all neighbors
    current_neighbors = set(initial_graph.neighbors(initial_node))
    all_neighbors = set(current_neighbors)
    
    # Iterate for the given number of steps
    for _ in range(max_steps - 1):
        new_neighbors = set()
        
        # Collect neighbors of current neighbors
        for neighbor in current_neighbors:
            new_neighbors.update(initial_graph.neighbors(neighbor))
        
        # Update the sets for the next iteration
        current_neighbors = new_neighbors - all_neighbors
        all_neighbors.update(current_neighbors)
    
    # Remove the initial node from the neighborhood
    all_neighbors.discard(initial_node)
    
    # Create the subgraph for the neighborhood
    neighborhood_subgraph = initial_graph.subgraph(all_neighbors)
    
    return neighborhood_subgraph


    

def relabel_subgraph(initial_graph, ext_subgraph):
    '''
        Modifies extending subgraph based on initial graph node labels
    '''

    # Finds starting number to relabel the nodes of the extending graph
    max_main_num = max(map(int, initial_graph.nodes())) + 1

    # Create a mapping dictionary to relabel nodes
    mapping = {old_label: old_label + max_main_num for old_label in ext_subgraph.nodes()}

    # Relabel nodes in the graph
    ext_subgraph = nx.relabel_nodes(ext_subgraph, mapping)

    return ext_subgraph


# Extend to multiple main graph nodes
# Also check if disconnected?
def enhanced_ext_subgraph_func(initial_graph, ext_subgraph, node_main, max_step=2):
    '''
        Enhances extending subgraph by including the connecting nodes
        in the main graph and some of its neighbors
    '''

    main_neighbors = get_neighborhood(initial_graph, node_main, max_step)

    initial_subgraph = initial_graph.subgraph(main_neighbors)
    
    _, enh_ext_subgraph = connect_subgraph(initial_subgraph, ext_subgraph)

    return enh_ext_subgraph


def generate_extended_embeddings(initial_graph, ext_subgraph, params, groups_dict):
    '''
        Generates the extending subgraph embeddings and groups
        by taking into account only the extended node embeddings
    '''

    # Connect the subgraph to the main graph
    [main_node, _], _ = connect_subgraph(initial_graph, ext_subgraph)

    # Generate enhanced extended subgraph
    enh_ext_subgraph = enhanced_ext_subgraph_func(initial_graph, ext_subgraph, main_node, max_step=2)

    # Generate model and embeddings for the enhanced extended subgraph
    _, model_ext = model_gen(enh_ext_subgraph, params)
    _, _, node_vectors_dict_ext = emb_group_gen(groups_dict, model_ext)

    # ## Previous method, slow
    # # Generate model and embeddings for the initial graph
    # _, model_initial = model_gen(initial_graph, params)
    # _, _, node_vectors_dict_initial = emb_group_gen(groups_dict, model_initial)

    # # Find nodes that are in the extended subgraph but not in the initial graph embeddings
    # kept_nodes = set(enh_ext_subgraph.nodes()) - set(node_vectors_dict_initial.keys()).intersection(enh_ext_subgraph.nodes())
    # print(kept_nodes)

    kept_nodes = list(ext_subgraph.nodes())

    # Gather X_ext and y_ext for the kept nodes
    X_ext = [node_vectors_dict_ext[node] for node in kept_nodes]
    y_ext = [groups_dict[node] for node in kept_nodes]

    return X_ext, y_ext


def initial_extending_embeddings(initial_graph, ext_subgraph, existing_filename=None, params=[64, 10, 80, 0.25, 4], group_df=pd.DataFrame()):
    '''
        Compares manual and modified graph models' embeddings
    '''

    groups_dict = groups_assign(initial_graph, ext_subgraph, group_df)

    # Generate modified graph
    [node_main, node_sub], mod_graph = connect_subgraph(initial_graph, ext_subgraph)

    # Generate model for the initial graph
    _, model_initial = model_gen(initial_graph, params, existing_filename=existing_filename)

    # Generate model for the modified graph
    _, model_mod = model_gen(mod_graph, params)

    # Generate embeddings for the initial graph
    X_initial, y_initial, _ = emb_group_gen(groups_dict, model_initial)

    # Generate embeddings for the modified graph
    X_mod, y_mod, _ = emb_group_gen(groups_dict, model_mod)

    # Generate embeddings for the extending subgraph, without common embeddings with initial graph
    X_ext, y_ext = generate_extended_embeddings(initial_graph, ext_subgraph, params, groups_dict)

    # Combine initial and extended embeddings for manual model
    X_manual = X_initial + X_ext
    y_manual = y_initial + y_ext

    return X_mod, y_mod, X_manual, y_manual


def quick_scores_func(X_local, y_local, X_global, y_global, test_sizes):
    '''
        Summarized f1 scores in a dictionary
    '''
    
    micro_f1_globals = [list(ovr_classifier(X_global, y_global, test_size).values())[0][0] for test_size in test_sizes]
    micro_f1_locals = [list(ovr_classifier(X_local, y_local, test_size).values())[0][0] for test_size in test_sizes]
    macro_f1_globals = [list(ovr_classifier(X_global, y_global, test_size).values())[0][1] for test_size in test_sizes]
    macro_f1_locals = [list(ovr_classifier(X_local, y_local, test_size).values())[0][1] for test_size in test_sizes]


    return {'local': {'micro': micro_f1_locals, 'macro': macro_f1_locals},
            'global': {'micro': micro_f1_globals, 'macro': macro_f1_globals}}



def remove_nodes_connected(initial_graph, num_nodes, removal_process='random'):
    '''
        Remove specific number of nodes while ensuring the graph remains connected.
        Returns pruned graph and removed nodes-edges dictionary.
    '''

    # Set the seed
    random.seed(42)

    graph = initial_graph.copy()
    removed_nodes_edges_dict = {}
    ignore_list = []

    while num_nodes > 0 and len(graph.nodes) > 0:

        # Newly added part - can remove based on centrality as well
        if removal_process == 'random':
            node = random.choice(list(graph.nodes()))
        elif removal_process == 'betweenness_centrality':
            bet_centr_dict = nxp.betweenness_centrality(graph, k=100)
            bet_centr_dict = {key: value for key, value in bet_centr_dict.items() if key not in ignore_list}
            node = max(bet_centr_dict, key=bet_centr_dict.get)
        elif removal_process == 'degree_centrality':
            deg_centr_dict = nx.degree_centrality(graph)
            deg_centr_dict = {key: value for key, value in deg_centr_dict.items() if key not in ignore_list}
            node = max(deg_centr_dict, key=deg_centr_dict.get)

        # Determine the connected component containing the node
        components = list(nx.connected_components(graph))
        component_with_node = None
        for component in components:
            if node in component:
                component_with_node = component
                break

        if component_with_node:
            # Create a subgraph of the connected component
            subgraph = graph.subgraph(component_with_node).copy()

            # Remove the node from the subgraph
            subgraph.remove_node(node)

            # Check if the subgraph remains connected
            if len(subgraph.nodes) > 0 and nx.is_connected(subgraph):
                removed_edges = list(graph.edges(node))
                removed_nodes_edges_dict[node] = removed_edges
                graph.remove_node(node)
                num_nodes -= 1
            else:
                if removal_process in ['betweenness_centrality', 'degree_centrality']:
                    ignore_list.append(node)

    return graph, removed_nodes_edges_dict



def removed_nodes_neighbors_func(initial_graph, removed_nodes, max_step=2):
    '''
        Returns neighbors of removed nodes
    '''

    neighbors = []

    for node in removed_nodes:
        neighbors += list(get_neighborhood(initial_graph, node, max_step)) 

    return list(set(neighbors)) # changed 13/5


def neighbors_subgraph_func(initial_graph, removed_nodes, neighbors):
    '''
        Returns neighbors subgraph taking into account the 
        removal of the removed nodes
    '''

    temp_neighbors_subgraph = initial_graph.subgraph(neighbors)

    neighbors_subgraph = temp_neighbors_subgraph.copy()
    neighbors_subgraph.remove_nodes_from(removed_nodes)

    return neighbors_subgraph


def pruned_modified_embeddings(initial_graph, params, num_nodes=12):
    '''
        Returns embeddings of pruned and modified subgraphs
    '''

    pruned_graph, removed_nodes = remove_nodes_connected(initial_graph, num_nodes)
    neighbors = removed_nodes_neighbors_func(initial_graph, removed_nodes)

    neighbors_subgraph = neighbors_subgraph_func(initial_graph, removed_nodes, neighbors)

    _, model_initial = model_gen(initial_graph, params)
    _, model_neighbors = model_gen(neighbors_subgraph, params)
    _, model_pruned = model_gen(pruned_graph, params)

    groups_dict = groups_assign(initial_graph, initial_graph) # the second subgraph does not matter in this case (?)

    _, _, node_vectors_dict_initial = emb_group_gen(groups_dict, model_initial)
    X_neighbors, y_neighbors, node_vectors_dict_neighbors = emb_group_gen(groups_dict, model_neighbors)
    X_pruned, y_pruned, node_vectors_dict_pruned = emb_group_gen(groups_dict, model_pruned)

    node_vectors_dict_mod = {key: node_vectors_dict_initial[key] for key in node_vectors_dict_initial if key not in removed_nodes}

    # node_vectors_dict_mod = node_vectors_dict_pruned.copy() # before it was dict_initial - still wrong, fix it (just remove the node keys from the removed nodes)
    node_vectors_dict_mod.update(node_vectors_dict_neighbors)

    X_local = list(node_vectors_dict_mod.values())
    y_local = [groups_dict[key] for key in node_vectors_dict_mod] #if key in groups_dict]

    return X_pruned, y_pruned, X_local, y_local


nodes_edges_func = lambda graph: (graph.number_of_nodes(), graph.number_of_edges())


def advanced_info(graph):
    '''
    Displays advanced information about a graph.

    Parameters:
        G (NetworkX graph): The input graph.
    '''
    print("Graph Information:")
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Density:", nx.density(graph))
    print("Is connected:", nx.is_connected(graph))
    print("Average clustering coefficient:", nx.average_clustering(graph))
    print("Directed:", nx.is_directed(graph))


# # Refactor this one
def dynamic_graph_gen(initial_graph, num_nodes_to_remove, save_bool=False, removal_process='random'):
    '''
    Generates a list of dynamically updated graphs starting from a subgraph of the initial graph.
    
    Parameters:
        initial_graph (NetworkX graph): The initial graph.
        num_nodes_to_remove (int): The number of nodes to remove from the initial graph.

    Returns:
        graphs_list: A list of dynamically updated graphs.
    '''

    graphs_list = [initial_graph]
    dynamic_graph = initial_graph
    
    print("Generating list of dynamic graphs:")
    for i in tqdm(range(num_nodes_to_remove)):
        dynamic_graph, _ = remove_nodes_connected(dynamic_graph, 1, removal_process)
        graphs_list.append(dynamic_graph)

    graphs_list = graphs_list[::-1]

    if save_bool:
        graphs_filenames_list = f'{initial_graph.name}_{removal_process}_{num_nodes_to_remove}.pkl'

        # Save the list of graphs to a file
        with open(f'./graphs/{graphs_filenames_list}', 'wb') as f:
            pickle.dump(graphs_list, f)

    return graphs_list



# def dynamic_graph_gen(initial_graph: nx.Graph, num_nodes_to_remove: int, save_bool: bool = False):
#     '''
#     Generates a list of dynamically updated graphs starting from a subgraph of the initial graph.
    
#     Parameters:
#         initial_graph (nx.Graph): The initial graph.
#         num_nodes_to_remove (int): The number of nodes to remove from the initial graph.
#         save_bool (bool): Whether to save the list of graphs to a file.

#     Returns:
#         List[nx.Graph]: A list of dynamically updated graphs.
#     '''

#     graphs_list = [initial_graph]
#     dynamic_graph = initial_graph.copy()  # Use a copy to avoid modifying the original graph
    
#     print("Generating list of dynamic graphs:")
#     for _ in tqdm(range(num_nodes_to_remove)):
#         dynamic_graph, _ = remove_nodes_connected(dynamic_graph, 1)
#         graphs_list.append(dynamic_graph.copy())  # Use a copy to avoid appending the same reference

#     graphs_list.reverse()

#     # if save_bool:
#     #     graphs_filenames_list = f'{initial_graph.name}_{num_nodes_to_remove}.pkl'

#     #     # Save the list of graphs to a file
#     #     with open(f'./graphs/{graphs_filenames_list}', 'wb') as f:
#     #         pickle.dump(graphs_list, f)

#     return graphs_list


def dynamic_extend_compare(initial_graph, added_nodes_num, params, groups_dict, graphs_list, quiet_bool=False):
    '''
        Compares graph with extended dynamic graph (improve description)
    '''

    [d, r, l, p, q] = params

    graph_i = graphs_list[0]
    graph_j = graphs_list[-1]

    print('Graphs:')
    print(nodes_edges_func(graph_i), nodes_edges_func(graph_j))

    nodes_i = set(graph_i.nodes())
    nodes_j = set(graph_j.nodes())

    diff_nodes = nodes_j - nodes_i
    # diff_graph = nx.subgraph(graph_j, diff_nodes)

    # -- This is what we already have from before, should not be timed
    # Initial training in initial graph
    node2vec_i = Node2Vec(graph_i, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=64, temp_folder='temp_folder', quiet=True)  # Use temp_folder for big graphs
    model_i = node2vec_i.fit() #, ns_exponent=1)
    X_i, y_i, node_vectors_dict_i = emb_group_gen(groups_dict, model_i)
    # --

    # ** Start of global embeddings **
    start_global_time = time.time()

    node2vec_j = Node2Vec(graph_j, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=64, temp_folder='temp_folder', quiet=True)  # Use temp_folder for big graphs
    model_j = node2vec_j.fit() #, ns_exponent=1)
    X_j, y_j, node_vectors_dict_j = emb_group_gen(groups_dict, model_j)
    X_global, y_global = X_j, y_j

    end_global_time = time.time()
    total_global_time = end_global_time - start_global_time
    # ** End of global embeddings **

    # ** Start of local embeddings **
    start_local_time = time.time()

    # Updating the embeddings locally in the different nodes
    node2vec_temp = Node2Vec(graph_j, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, temp_folder='temp_folder', 
                    starting_nodes=diff_nodes, quiet=True)  # Use temp_folder for big graphs

    model_temp = node2vec_j.fit() #, ns_exponent=1)
    _, _, node_vectors_dict_temp = emb_group_gen(groups_dict, model_temp)

    traversed_nodes = set.union(*map(set, node2vec_temp.walks))
    traversed_nodes = list(map(int, traversed_nodes))

    node_vectors_dict_manual = node_vectors_dict_i.copy()
    temp_dict = {key: node_vectors_dict_temp[key] for key in traversed_nodes}
    node_vectors_dict_manual.update(temp_dict)

    X_local = list(node_vectors_dict_manual.values())
    y_local = [groups_dict[int(key)] for key in node_vectors_dict_manual if key in groups_dict]

    end_local_time = time.time()
    total_local_time = end_local_time - start_local_time
    # ** End of local embeddings **

    test_sizes = np.arange(0.1, 1, 0.1)

    if not quiet_bool:
        print("\n**From scratch**")
        for test_size in test_sizes:
            print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_global, y_global, test_size).values())}")


        print("\n**Manually**")
        for test_size in test_sizes:
            print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_local, y_local, test_size).values())}")

    return X_global, y_global, X_local, y_local, total_global_time, total_local_time, len(diff_nodes)


def dynamic_prune_compare(initial_graph, removed_nodes_num, params, groups_dict, graphs_list, quiet_bool=False):
    '''
        Compare graph pruned (improve description)
    '''

    [d, r, l, p, q] = params

    graph_pruned = graphs_list[0]
    graph_upd = graphs_list[-1]

    print('Graphs:')
    print(nodes_edges_func(graph_pruned), nodes_edges_func(graph_upd))

    nodes_pruned = set(graph_pruned.nodes())
    nodes_upd = set(graph_upd.nodes())

    diff_nodes = nodes_upd - nodes_pruned
    # print(len(diff_nodes))

    # -- This is what we already have from before, should not be timed
    # Initial model and embeddings
    node2vec_initial = Node2Vec(initial_graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=64, temp_folder='temp_folder', quiet=True)  # Use temp_folder for big graphs
    model_initial = node2vec_initial.fit() #, ns_exponent=1)
    X_initial, y_initial, node_vectors_dict_initial = emb_group_gen(groups_dict, model_initial)
    # --

    # ** Start of global embeddings **
    start_global_time = time.time()

    # node2vec_upd = Node2Vec(graph_upd, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, temp_folder='temp_folder', quiet=True)  # Use temp_folder for big graphs
    node2vec_upd = Node2Vec(graph_pruned, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=64, temp_folder='temp_folder', quiet=True)  # Use temp_folder for big graphs
    model_upd = node2vec_upd.fit() #, ns_exponent=1)
    X_upd, y_upd, node_vectors_dict_upd = emb_group_gen(groups_dict, model_upd)
    X_global, y_global = X_upd, y_upd

    end_global_time = time.time()
    total_global_time = end_global_time - start_global_time
    # ** End of global embeddings **

    # ** Start of local embeddings **
    start_local_time = time.time()

    neighbors = removed_nodes_neighbors_func(graph_upd, diff_nodes, max_step=1)

    # Pruned models
    node2vec_pruned = Node2Vec(graph_pruned, dimensions=d, walk_length=l//2, num_walks=r//2, p=p, q=q, workers=64, temp_folder='temp_folder',
                        starting_nodes=neighbors, quiet=True)  # Use temp_folder for big graphs
 
    model_pruned = node2vec_pruned.fit() #, ns_exponent=1)
    X_pruned, y_pruned, node_vectors_dict_pruned = emb_group_gen(groups_dict, model_pruned)

    # ??
    node_vectors_dict_mod = {key: node_vectors_dict_initial[key] for key in node_vectors_dict_initial if key not in diff_nodes}
    node_vectors_dict_mod.update(node_vectors_dict_pruned)

    # before it was mod
    X_local = list(node_vectors_dict_mod.values()) 
    y_local = [groups_dict[int(key)] for key in node_vectors_dict_mod] #if key in groups_dict]

    end_local_time = time.time()
    total_local_time = end_local_time - start_local_time
    # ** End of local embeddings **

    test_sizes = np.arange(0.1, 1, 0.1)

    if not quiet_bool:
        print("\n**From scratch (global retraining)**")
        for test_size in test_sizes:
            print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_global, y_global, test_size).values())}")


        print("\n**Manually (local retraining)**")
        for test_size in test_sizes:
            print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_local, y_local, test_size).values())}")


    return X_global, y_global, X_local, y_local, total_global_time, total_local_time, len(neighbors)


def results_store_func(initial_graph, mod_type, local_vars, global_vars, test_sizes, mod_nodes_num, starting_nodes_num, params, training_times, removal_process='random'):
    '''
        Function to store the results
    '''

    results_df = pd.read_csv('results.csv')
    
    graph_name = initial_graph.name
    num_nodes, num_edges = nodes_edges_func(initial_graph)
    
    execution_timestamp = generate_execution_timestamp()

    [X_local, y_local] = local_vars  
    [X_global, y_global] = global_vars
    total_global_time, total_local_time = training_times

    training_times_dict = {'local': total_local_time, 'global': total_global_time}

    score_dict = quick_scores_func(X_local, y_local, X_global, y_global, test_sizes)

    for training_type in score_dict:
        f1_types_dict = score_dict[training_type]
        training_time =  training_times_dict[training_type]

        for f1_type in f1_types_dict:
            f1_scores = f1_types_dict[f1_type]

            results_row = [graph_name, num_nodes, num_edges, training_type, f1_type] + f1_scores + [params, training_time, mod_type, mod_nodes_num, starting_nodes_num, removal_process, execution_timestamp]
            results_df = pd.concat([pd.DataFrame([results_row], columns=results_df.columns), results_df], ignore_index=True)

    results_df.to_csv('results.csv', index=False)

    return results_df


def results_output_func(initial_graph, mod_type, mod_nodes_num, params, groups_dict, graphs_list, removal_process='random'):
    '''
        Function that shows the progress of the dynamic update and outputs results df
    '''

    dynamic_func = {'prune': dynamic_prune_compare, 'extend': dynamic_extend_compare}[mod_type]

    print(f'** Modification type: {mod_type} for {mod_nodes_num} nodes **')

    X_global, y_global, X_local, y_local, total_global_time, total_local_time, starting_nodes_num = \
    dynamic_func(initial_graph, mod_nodes_num, params, groups_dict, graphs_list, quiet_bool=True)

    test_sizes = np.arange(0.1, 1, 0.1)

    local_vars = [X_local, y_local]
    global_vars = [X_global, y_global]
    training_times = total_global_time, total_local_time 

    results_df = results_store_func(initial_graph, mod_type, local_vars, global_vars, test_sizes, mod_nodes_num, starting_nodes_num, params, training_times, removal_process)

    return results_df



# # Older method: inefficient and slower? Or better than newer one?
# def dynamic_graph_gen(initial_graph, num_nodes_to_remove):
#     '''
#     Generates a list of dynamically updated graphs starting from a subgraph of the initial graph.
    
#     Parameters:
#         initial_graph (NetworkX graph): The initial graph.
#         num_nodes_to_remove (int): The number of nodes to remove from the initial graph.

#     Returns:
#         graphs_list: A list of dynamically updated graphs.
#     '''

#     pruned_subgraph, removed_nodes_edges_dict = remove_nodes_connected(initial_graph, num_nodes_to_remove)
#     dynamic_graph = pruned_subgraph

#     removed_nodes = list(removed_nodes_edges_dict.keys())
#     new_nodes = removed_nodes

#     graphs_list = []

#     for node in new_nodes:
#         new_dynamic_graph = dynamic_graph.copy()
#         added_edges = removed_nodes_edges_dict[node]

#         new_dynamic_graph.add_edges_from(added_edges)

#         # Check for connectedness
#         if nx.is_connected(new_dynamic_graph):
#             graphs_list.append(new_dynamic_graph)
#             dynamic_graph = new_dynamic_graph.copy()

#     return graphs_list



# def nodes_remove_func(graph, percentage):
#     '''
#         Removes a percentage of nodes from a graph
#     '''

#     all_nodes = list(graph.nodes())

#     # Calculate the number of elements to select based on the percentage
#     k = int(len(all_nodes) * percentage)

#     # Use random.sample to get a random selection of indices from the list
#     indices = random.sample(range(len(all_nodes)), k)

#     # Return removed subset
#     return [all_nodes[i] for i in indices] # for i in range(len(all_nodes)) if i not in indices] 


# def pruned_graph_func(initial_graph, neighbors):
#     '''
#         Returns pruned subgraph of initial graph depending on 
#         the neighbors of the removed nodes
#     '''

#     pruned_graph = initial_graph.subgraph(neighbors)

#     return pruned_graph



# def find_chains(graph):
#     '''
#         Find chains of "path" nodes
#     '''
#     chains = []
    
#     # Iterate over nodes in the graph
#     for node in graph.nodes:
#         # Check if the node is pendant (degree == 1)
#         if graph.degree(node) == 1:
#             chain = [node]
#             next_node = list(graph.neighbors(node))[0]  # Get the neighbor of the pendant node
            
#             # Follow the chain of pendant nodes until a non-pendant node is reached
#             while graph.degree(next_node) == 2:
#                 chain.append(next_node)
#                 next_node = list(graph.neighbors(next_node))[0]  # Get the neighbor of the current node
            
#             # Add the chain to the list of chains
#             chains.append(chain)
    
#     return chains


# def multiple_scores_func(X_mod, y_mod, X_manual, y_manual, test_sizes):
#     '''
#         Outputs f1 scores from the generated X 
#         (scratch and manually for each test size
#     '''
#     print("\n**New graph (from scratch)**")
#     for test_size in test_sizes:
#         print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_mod, y_mod, test_size).values())}")

#     print(2 * '\n')

#     print("**Manually updated graph (extending the graph)**")
#     for test_size in test_sizes:
#         print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_manual, y_manual, test_size).values())}")
