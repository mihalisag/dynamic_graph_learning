import os
import random

import plotly
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
import pandas as pd

import networkx as nx
import pickle

# Modify eliorc's implementation
from eliorc_mod.node2vec import Node2Vec
from gensim.models import Word2Vec

import itertools
from collections import Counter

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score #, accuracy_score

# Need to change max value to a cluster number or something
def generate_random_number(min_value=0, max_value=4):

    return random.randint(min_value, max_value)


quick_info = lambda graph: (graph.number_of_nodes(), graph.number_of_edges())


# def plot_graph(graph):
#     '''
#         Helper function to draw graph
#     '''
    
#     # Compute Kamada-Kawai layout
#     pos = nx.kamada_kawai_layout(graph)

#     plt.figure(figsize=(4, 4))
#     nx.draw_networkx(graph, pos, node_size=75, font_size=6, node_color='lightblue', edge_color='grey')
#     plt.show()


def plot_graph(graph, node_type_list=[]):
    '''
        Helper function to draw graph with specific colors
    '''

    # The order is: removed nodes, neighbors and added vertices
    colors_list = ['red', 'orange', 'green']

    node_num = graph.number_of_nodes()
    node_colors = node_num * ['lightblue']

    for node_list in node_type_list:
        color = colors_list[node_type_list.index(node_list)]

        for node in node_list:
            node_colors[node] = color
        
    is_disconnected = nx.number_connected_components(graph) > 1

    if is_disconnected:
        pos = nx.spring_layout(graph)
    else:
        pos = nx.kamada_kawai_layout(graph)

    plt.figure(figsize=(4, 4))
    nx.draw_networkx(graph, pos, node_size=75, font_size=6, edge_color='grey', node_color=node_colors)
    plt.show()
    

def model_gen(graph, params, existing_filename=None, quiet_bool=True):
    '''
        Generates and saves model, returns node2vec and model fit
    '''

    if existing_filename:
        model = Word2Vec.load(existing_filename)
        print("Model loaded.")

        return '', model

    [d, r, l, p, q] = params

    node2vec = Node2Vec(graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, temp_folder='temp_folder', quiet=quiet_bool)  # Use temp_folder for big graphs

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


def freq_plot(visit_counts, cumulative_freq_prob):
    '''
        Makes a plot of the frequency distribution
    '''
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=visit_counts,
        y=cumulative_freq_prob,
        mode='markers',
        marker=dict(symbol='cross', size=5),
        line=dict(width=2)
    ))


    fig.update_layout(
        title="Log-log plot of visit frequency distribution",
        xaxis_title=r"Vertex visitation count x",
        yaxis_title=r"Probability of vertex appearing at least x times",
        xaxis_type='log',  # Add log scale for x-axis
        yaxis_type='log',  # Add log scale for y-axis
        width=640,
        height=480
    )

    fig.write_image('figures/freq_plot.svg')
    fig.write_image('figures/freq_plot.pdf')
    fig.show()

    return fig


def degree_plot(cumulative_deg_prob):
    '''
        Makes a plot of the degree distribution
    '''

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(cumulative_deg_prob))),
        y=cumulative_deg_prob,
        mode='markers',
        marker=dict(symbol='cross', size=5),
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Log-log plot of degree distribution",
        xaxis_title="Degree (log scale)",
        yaxis_title="Probability of Degree >= x (log scale)",
        xaxis_type='log',
        yaxis_type='log',
        showlegend=False,  # Remove legend for this plot
        yaxis=dict(type="log", autorange=True),
        xaxis=dict(type="log", autorange=True),
        xaxis_showgrid=True,  # Show gridlines for x-axis
        yaxis_showgrid=True,  # Show gridlines for y-axis
        width=640, 
        height=480
    )

    fig.write_image('figures/degree_plot.svg')
    fig.write_image('figures/degree_plot.pdf')
    fig.show()

    return fig


def degree_freq_plot(graph, degrees, node_freq_dict):
    '''
        Makes a scatter plot of the degrees and the node frequencies
    '''

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted(degrees),
        y=sorted(node_freq_dict.values()),
        mode='markers',
        marker=dict(size=8),
        line=dict(width=2)
    ))

    fig.update_layout(
        title=f"Degree vs Frequency Plot for {graph.name} |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
        xaxis_title="Degree",
        yaxis_title="Frequency (Node Visit Count)",
        showlegend=False,  # Remove legend for this plot
        width=640, 
        height=480
    )

    fig.write_image('figures/degree_freq_plot.svg')
    fig.write_image('figures/degree_freq_plot.pdf')
    fig.show()

    return fig


# Refactor this function
def groups_assign(initial_graph, subgraph, group_df=pd.DataFrame()):
    '''
        Assign group numbers to nodes for node classification
    '''

    if group_df.shape == (0, 0):
        _, new_graph = connect_subgraph(initial_graph, subgraph)
        
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

    groups_dict = {node_num: group_df.loc[group_df['node_num'] == node_num, 'group'].values[0] for node_num in list(group_df['node_num'])}

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
    y = [groups_dict[key] for key in node_vectors_dict if key in groups_dict] # the mapping keeps the same order

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


def params_grid_search(graph, params_list):
    '''
        Performs grid search and outputs plots
    '''

    fig1 = go.Figure()
    fig2 = go.Figure()


    for params in tqdm(params_list):

        [d, r, l, p, q] = params

        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, quiet=True)  # Use temp_folder for big graphs
        walks = node2vec.walks
        
        # Degrees and walks' frequencies
        degrees, cumulative_deg_prob = degrees_distribution_gen(graph)
        node_freq_dict, visit_counts, cumulative_freq_prob = freq_gen(walks=walks)

        fig1.add_trace(go.Scatter(
        x=visit_counts,
        y=cumulative_freq_prob,
        mode='markers',
        marker=dict(symbol='cross', size=5),
        line=dict(width=2),
        name=f'p={p}, q={q}',  # Add labels for each point
        ))

        fig2.add_trace(go.Scatter(
            x=sorted(degrees),
            y=sorted(node_freq_dict.values()),
            mode='markers',
            marker=dict(size=6),
            line=dict(width=2),
            name=f'p={p}, q={q}',  # Add labels for each point
        ))


    fig1.update_layout(
        title=f"Log-log plot of visit frequency distribution for {graph.name} |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
        xaxis_title=r"Vertex visitation count x",
        yaxis_title=r"Probability of vertex appearing at least x times",
        xaxis_type='log',  # Add log scale for x-axis
        yaxis_type='log',  # Add log scale for y-axis
        width=640,
        height=480
    )


    fig2.update_layout(
        title=f"Degree vs Frequency Plot for {graph.name} |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
        xaxis_title="Degree",
        yaxis_title="Frequency (Node Visit Count)",
        width=640, 
        height=480
    )


    fig1.show()
    fig2.show()


def node2vec_configs():
    '''
        Generates list of configs
    '''
    D_values = [64, 128, 192, 256]
    R_values = [10] 
    L_values = [80]
    P_values = [0.25, 0.5, 1, 2, 4]
    Q_values = [0.25, 0.5, 1, 2, 4] 

    # Generate all possible combinations of orders and seasonal orders
    parameter_values = [D_values, R_values, L_values, P_values, Q_values]
    parameter_combinations = list(itertools.product(*parameter_values))

    return parameter_combinations


def test_grid_search(graph, X, y, test_sizes=np.arange(0.1, 1, 0.1)):
    '''
        Performs grid search in test size and outputs plots
    '''

    all_f1_scores = dict()

    for test_size in test_sizes[::-1]:
        f1_scores = ovr_classifier(X, y, test_size)
        all_f1_scores.update(f1_scores)

    f1_micro_list = []
    f1_macro_list = []

    for key, value in all_f1_scores.items():
        f1_micro_list.append(value[0])
        f1_macro_list.append(value[1])


    fig = go.Figure()

    fig.add_trace(go.Scatter(
            x=list(all_f1_scores.keys()),
            y=f1_micro_list,
            mode='markers',
            marker=dict(size=6, symbol='circle'),
            line=dict(width=2),
            name="Micro F1 score"
        ))

    fig.add_trace(go.Scatter(
            x=list(all_f1_scores.keys()),
            y=f1_macro_list,
            mode='markers',
            marker=dict(size=6, symbol='circle'),
            line=dict(width=2),
            name="Macro F1 score"
        ))



    fig.update_layout(
        title=f"F1 Score for {graph.name} (p, q) = (), |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
        xaxis_title=r"Training size",
        yaxis_title=r"F1 Score",
        width=640,
        height=480
    )


    fig.show()


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


def get_neighborhood(initial_graph, initial_node, max_step):
    '''
    This function finds the neighborhood of a node in a
    graph within a specified hop limit.
    '''
    
    neighbors = set(initial_graph.neighbors(initial_node))
    all_neighbors = set(initial_graph.neighbors(initial_node))

    for step in range(max_step - 1):
        temp = set()
    
        for node in neighbors:
            temp.update(set(initial_graph.neighbors(node)))
    
        neighbors = temp - all_neighbors

        all_neighbors.update(neighbors)
    
    all_neighbors.discard(initial_node)

    neighborhood = initial_graph.subgraph(all_neighbors) 
    
    return neighborhood

    

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


def compare_models(initial_graph, ext_subgraph, existing_filename=None, params=[64, 10, 80, 0.25, 4], test_sizes=np.arange(0.1, 1, 0.1), group_df=pd.DataFrame()):
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

    print("\n**New graph (from scratch)**")
    for test_size in test_sizes:
        print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_mod, y_mod, test_size).values())}")

    print(2 * '\n')

    print("**Manually updated graph (extending the graph)**")
    for test_size in test_sizes:
        print(f"- For training size: {(1 - test_size):.1f}: {list(ovr_classifier(X_manual, y_manual, test_size).values())}")


def remove_nodes_connected(initial_graph, num_nodes):
    '''
    Remove specific number of nodes while ensuring the graph remains connected.
    Returns pruned graph and removed nodes-edges dictionary.
    '''

    graph = initial_graph.copy()

    removed_nodes_edges_dict = {}

    while num_nodes > 0:
        node = random.choice(list(graph.nodes()))

        temp_graph = graph.copy()
        temp_graph.remove_node(node)

        if nx.is_connected(temp_graph):
            removed_edges = list(graph.edges(node))
            removed_nodes_edges_dict[node] = removed_edges
            graph.remove_node(node)
            num_nodes -= 1

    # removed_nodes = removed_nodes_edges_dict.keys()

    return graph, removed_nodes_edges_dict # removed_nodes, removed_edges


def removed_nodes_neighbors_func(initial_graph, removed_nodes, max_step=2):
    '''
        Returns neighbors of removed nodes
    '''

    neighbors = []

    for node in removed_nodes:
        neighbors += list(get_neighborhood(initial_graph, node, max_step)) 

    return neighbors


def neighbors_subgraph_func(initial_graph, removed_nodes, neighbors):
    '''
        Returns neighbors subgraph taking into account the 
        removal of the removed nodes
    '''

    temp_neighbors_subgraph = initial_graph.subgraph(neighbors)

    neighbors_subgraph = temp_neighbors_subgraph.copy()
    neighbors_subgraph.remove_nodes_from(removed_nodes)

    return neighbors_subgraph


# def pruned_graph_func(initial_graph, neighbors):
#     '''
#         Returns pruned subgraph of initial graph depending on 
#         the neighbors of the removed nodes
#     '''

#     pruned_graph = initial_graph.subgraph(neighbors)

#     return pruned_graph


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

    node_vectors_dict_mod = node_vectors_dict_pruned.copy() # before it was dict_initial - still wrong, fix it (just remove the node keys from the removed nodes)
    node_vectors_dict_mod.update(node_vectors_dict_neighbors)

    X_mod = list(node_vectors_dict_mod.values())
    y_mod = [groups_dict[key] for key in node_vectors_dict_mod] #if key in groups_dict]

    return X_pruned, y_pruned, X_mod, y_mod


def advanced_info(G):
    """
    Displays advanced information about a graph.

    Parameters:
        G (NetworkX graph): The input graph.
    """
    print("Graph Information:")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Density:", nx.density(G))
    print("Is connected:", nx.is_connected(G))
    print("Average clustering coefficient:", nx.average_clustering(G))
    print("Directed:", nx.is_directed(G))


def dynamic_graph_gen(initial_graph, num_nodes_to_remove):
    '''
    Generates a list of dynamically updated graphs starting from a subgraph of the initial graph.
    
    Parameters:
        initial_graph (NetworkX graph): The initial graph.
        num_nodes_to_remove (int): The number of nodes to remove from the initial graph.

    Returns:
        graphs_list: A list of dynamically updated graphs.
    '''

    pruned_subgraph, removed_nodes_edges_dict = remove_nodes_connected(initial_graph, num_nodes_to_remove)
    dynamic_graph = pruned_subgraph

    removed_nodes = list(removed_nodes_edges_dict.keys())
    new_nodes = removed_nodes

    graphs_list = []

    for node in new_nodes:
        new_dynamic_graph = dynamic_graph.copy()
        added_edges = removed_nodes_edges_dict[node]

        new_dynamic_graph.add_edges_from(added_edges)

        graphs_list.append(new_dynamic_graph)
        dynamic_graph = new_dynamic_graph.copy()

    return graphs_list


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