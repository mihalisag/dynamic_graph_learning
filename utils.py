import os
import random

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import scipy
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


def model_gen(G, params, quiet_bool=True):
    '''
        Generates and saves model, returns node2vec and model fit
    '''

    [d, r, l, p, q] = params

    node2vec = Node2Vec(G, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, temp_folder='temp_folder', quiet=quiet_bool)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4) #, ns_exponent=1)
    
    # Save model
    model_filename = f"{G.name}_d{d}_r{r}_l{l}_p{p}_q{q}.mdl"
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


def degrees_distribution_gen(G):
    '''
        Generates degrees distribution used for plot
    '''

    # Calculate degree distribution
    degrees = sorted([G.degree(i) for i in G], reverse=True)
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


    fig.show()


def degreee_plot(cumulative_deg_prob):
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

    fig.show()


def degree_freq_plot(G, degrees, node_freq_dict):
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
        title=f"Degree vs Frequency Plot for {G.name} |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}",
        xaxis_title="Degree",
        yaxis_title="Frequency (Node Visit Count)",
        showlegend=False,  # Remove legend for this plot
        width=640, 
        height=480
    )

    fig.show()


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


def params_grid_search(G, params_list):
    '''
        Performs grid search and outputs plots
    '''

    fig1 = go.Figure()
    fig2 = go.Figure()


    for params in tqdm(params_list):

        [d, r, l, p, q] = params

        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(G, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, quiet=True)  # Use temp_folder for big graphs
        walks = node2vec.walks
        
        # Degrees and walks' frequencies
        degrees, cumulative_deg_prob = degrees_distribution_gen(G)
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
        title=f"Log-log plot of visit frequency distribution for {G.name} |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}",
        xaxis_title=r"Vertex visitation count x",
        yaxis_title=r"Probability of vertex appearing at least x times",
        xaxis_type='log',  # Add log scale for x-axis
        yaxis_type='log',  # Add log scale for y-axis
        width=640,
        height=480
    )


    fig2.update_layout(
        title=f"Degree vs Frequency Plot for {G.name} |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}",
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



def test_grid_search(G, X, y, test_sizes=np.arange(0.1, 1, 0.1)):
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
        title=f"F1 Score for {G.name} (p, q) = (), |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}",
        xaxis_title=r"Training size",
        yaxis_title=r"F1 Score",
        width=640,
        height=480
    )


    fig.show()