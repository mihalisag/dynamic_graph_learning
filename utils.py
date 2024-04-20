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

from utils import *

# Need to change max value to a cluster number or something
def generate_random_number(min_value=0, max_value=4):
    return random.randint(min_value, max_value)


def plot_graph(graph):
    '''
        Helper function to draw graph
    '''

    plt.figure(figsize=(3, 3))
    nx.draw(graph, with_labels=True)
    plt.show()


def model_gen(graph, params, quiet_bool=True):
    '''
        Generates and saves model, returns node2vec and model fit
    '''

    [d, r, l, p, q] = params

    node2vec = Node2Vec(graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, temp_folder='temp_folder', quiet=quiet_bool)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4) #, ns_exponent=1)
    
    # Save model
    model_filename = f"{graph.name}_d{d}_r{r}_l{l}_p{p}_q{q}.mdl"
    model.save(f"./models/{model_filename}")

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

    # F1 Macro Score
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # F1 Micro Score
    f1_micro = f1_score(y_test, y_pred, average='micro')

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
        temp_df = pd.DataFrame(index=range(subgraph.number_of_nodes()))

        temp_df['node_num'] = temp_df.index

        random_numbers = [generate_random_number() for _ in range(len(temp_df))]

        temp_df['group'] = random_numbers

        group_df = pd.concat([group_df, temp_df], ignore_index=True)  # Optional: reset index

    groups_dict = {node_num: group_df.loc[node_num, 'group'] for node_num in list(group_df['node_num'])}

    return groups_dict
    

def train_test_creator(groups_dict, model):
    '''
        Creates train, test sets based on embeddings and groups (?)
    '''

    # Create a dictionary to map node IDs to vectors
    node_vectors = {}
    for node_id in model.wv.index_to_key:
        node_vectors[int(node_id)] = model.wv.get_vector(node_id)

    X = list(node_vectors.values())
    y = [groups_dict[key] for key in node_vectors if key in groups_dict] # the mapping keeps the same order

    return X, y, node_vectors # remove in future


def ext_subgraph_modify(initial_graph, ext_subgraph):
    '''
        Modifies extending subgraph based on initial graph node labels
    '''

    # Finds starting number to relabel the nodes of the extending graph
    max_main_num = max(initial_graph.nodes()) + 1

    # Create a mapping dictionary to relabel nodes
    mapping = {old_label: old_label + max_main_num for old_label in ext_subgraph.nodes()}

    # Relabel nodes in the graph
    ext_subgraph = nx.relabel_nodes(ext_subgraph, mapping)

    return ext_subgraph


# Extend to multiple main graph nodes
def enhanced_ext_subgraph_func(initial_graph, ext_subgraph, node_main, max_step=2):
    '''
        Enhances extending subgraph by including the connecting nodes
        in the main graph and some of its neighbors
    '''

    main_neighbors = get_neighborhood(initial_graph, node_main, max_step)

    initial_subgraph = initial_graph.subgraph(main_neighbors)

    _, enh_ext_subgraph = connect_subgraph(initial_subgraph, ext_subgraph)

    return enh_ext_subgraph


def remove_nodes_connected(initial_graph, num_nodes):
    '''
        Remove specific number of nodes while ensuring the graph remains connected.
        Returns removed nodes and pruned graph.
    '''

    graph = initial_graph.copy()

    removed_nodes = []

    while num_nodes > 0:
        node = random.choice(list(graph.nodes()))

        temp_graph = graph.copy()
        temp_graph.remove_node(node)

        if nx.is_connected(temp_graph):
            graph.remove_node(node)
            removed_nodes.append(node)
            num_nodes -= 1
        
    return graph, removed_nodes


def pruned_subgraph_func(initial_graph, removed_nodes, max_step=2):
    '''
        Returns pruned subgraph of initial graph depending on 
        the neighbors of the removed nodes
    '''

    neighbors = []

    for node in removed_nodes:
        neighbors += list(get_neighborhood(initial_graph, node, max_step))

    pruned_subgraph = initial_graph.subgraph(neighbors)

    return pruned_subgraph


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