# * This is a file containing the plotting helper functions used in the project *

# -- Imports -- ## 

import numpy as np

import plotly
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import networkx as nx

from tqdm import tqdm

# Modify eliorc's implementation
from eliorc_mod.node2vec import Node2Vec
from gensim.models import Word2Vec

from main_utils import degrees_distribution_gen, freq_gen, ovr_classifier


# -- Functions -- ## 

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


def freq_plot(visit_counts, cumulative_freq_prob):
    '''
        Makes a plot of the frequency distribution
    '''
    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(6, 5))
    
    # Show grid lines
    plt.grid(color='lightgrey')

    # Set x-axis and y-axis to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Create a scatter plot
    plt.scatter(x=visit_counts, y=cumulative_freq_prob, s=4, zorder=2, color='#2053A6')

    # Set plot title and axis labels with larger font sizes
    plt.xlabel("Visits $x$", fontdict={'size': 20})
    plt.ylabel("$Pr(V \geq x)$", fontdict={'size': 20})

    # # Increase tick label font size and tick line width
    # plt.tick_params(axis='both', which='major', labelsize=14, linewidth=1.2)
    
    # Adjust spacing and save the plot
    plt.tight_layout()
    plt.savefig('figures/freq_plot.pdf', dpi=100, bbox_inches='tight')
    
    plt.show()


def degree_plot(cumulative_deg_prob):
    '''
        Makes a plot of the degree distribution
    '''
    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text
    plt.figure(figsize=(6, 5))
    
    # Show dashed grid lines
    plt.grid(color='lightgrey')
    # Set x-axis and y-axis to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Create a scatter plot
    plt.scatter(list(range(len(cumulative_deg_prob))), cumulative_deg_prob, s=4, zorder=2, color='#2053A6')

    # Set plot title and axis labels
    # plt.title("$\log$-$\log$ plot of degree distribution")
    plt.xlabel("$deg(v)$", fontdict={'size': 20})
    plt.ylabel("$Pr(\, deg(v) \geq x \,)$", fontdict={'size': 20})

    # # Save and display the plot
    plt.tight_layout()
    plt.savefig('figures/degree_plot.svg', dpi=100, bbox_inches='tight')
    plt.savefig('figures/degree_plot.pdf', dpi=100, bbox_inches='tight')

    plt.show()



def degree_freq_plot(graph, degrees, node_freq_dict):
    '''
        Makes a scatter plot of the degrees and the node frequencies
    '''

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    plt.rcParams['font.size'] = 12 
    plt.figure(figsize=(6, 5))

    # Show dashed grid lines
    plt.grid(color='lightgrey')
    # Create a scatter plot
    plt.scatter(sorted(degrees), sorted(node_freq_dict.values()), s=4, zorder=2, color='#2053A6')

    # Set plot title and axis labels
    # plt.title(r"Degree vs Frequency Plot for {} $|V|={}$, $|E|={}$".format(graph.name, graph.number_of_nodes(), graph.number_of_edges()))
    plt.xlabel("Degree", fontdict={'size': 20})
    plt.ylabel("Visits", fontdict={'size': 20})

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('figures/degree_freq_plot.svg', dpi=100, bbox_inches='tight')
    plt.savefig('figures/degree_freq_plot.pdf', dpi=100, bbox_inches='tight')

    # Show the plots
    plt.show()



def params_grid_search(graph, params_list):
    '''
        Performs grid search and outputs plots using Matplotlib
    '''

    # plt.rcParams['text.usetex'] = True # TeX rendering
    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # Define colors
    colors = {
        'BLUE': '#2053A6',
        'ORANGE': '#FF6011',
        'RED': '#FF1B20',
        'GREEN': '#0CC03E',
        'PURPLE': '#C801FF'
    }
    
    # List of color values
    color_values = list(colors.values())

    # Initialize figures
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for idx, params in enumerate(tqdm(params_list)):
        [d, r, l, p, q] = params

        # Assign color based on the index
        color = color_values[idx % len(color_values)]

        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, quiet=True)  # Use temp_folder for big graphs
        walks = node2vec.walks
        
        # Degrees and walks' frequencies
        degrees, cumulative_deg_prob = degrees_distribution_gen(graph)
        node_freq_dict, visit_counts, cumulative_freq_prob = freq_gen(walks=walks)

        # Plotting the visit frequency distribution
        ax1.scatter(visit_counts, cumulative_freq_prob, label=f'p={p}, q={q}', s=25, marker='x', color=color)
        
        # Plotting the degree vs frequency
        ax2.scatter(sorted(degrees), sorted(node_freq_dict.values()), label=f'p={p}, q={q}', s=25, color=color)

    # Configure the first plot (visit frequency distribution)
    ax1.set_xlabel("Visits $x$", fontdict={'size': 20})
    ax1.set_ylabel("$Pr(V \geq x)$", fontdict={'size': 20})
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.25)

    # Configure the second plot (degree vs frequency)
    ax2.set_xlabel("Degree", fontdict={'size': 20})
    ax2.set_ylabel("Visits", fontdict={'size': 20})
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.25)

    # Optionally save the figures
    fig1.savefig('figures/visit_frequency_distribution.pdf', dpi=100, bbox_inches='tight')
    fig2.savefig('figures/degree_vs_frequency.pdf', dpi=100, bbox_inches='tight')

    # Adjust the legend
    plt.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.20))

    # Show the plots
    plt.show()



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

    # Show the plots
    fig.show()



# def degree_plot(cumulative_deg_prob):
#     '''
#         Makes a plot of the degree distribution
#     '''

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=list(range(len(cumulative_deg_prob))),
#         y=cumulative_deg_prob,
#         mode='markers',
#         marker=dict(symbol='cross', size=5),
#         line=dict(width=2)
#     ))

#     fig.update_layout(
#         title="Log-log plot of degree distribution",
#         xaxis_title="Degree (log scale)",
#         yaxis_title="Probability of Degree >= x (log scale)",
#         xaxis_type='log',
#         yaxis_type='log',
#         showlegend=False,  # Remove legend for this plot
#         yaxis=dict(type="log", autorange=True),
#         xaxis=dict(type="log", autorange=True),
#         xaxis_showgrid=True,  # Show gridlines for x-axis
#         yaxis_showgrid=True,  # Show gridlines for y-axis
#         width=640, 
#         height=480
#     )

#     fig.write_image('figures/degree_plot.svg')
#     fig.write_image('figures/degree_plot.pdf')
#     fig.show()

#     return fig


# def degree_freq_plot(graph, degrees, node_freq_dict):
#     '''
#         Makes a scatter plot of the degrees and the node frequencies
#     '''

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=sorted(degrees),
#         y=sorted(node_freq_dict.values()),
#         mode='markers',
#         marker=dict(size=8),
#         line=dict(width=2)
#     ))

#     fig.update_layout(
#         title=f"Degree vs Frequency Plot for {graph.name} |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
#         xaxis_title="Degree",
#         yaxis_title="Frequency (Node Visit Count)",
#         showlegend=False,  # Remove legend for this plot
#         width=640, 
#         height=480
#     )

#     fig.write_image('figures/degree_freq_plot.svg')
#     fig.write_image('figures/degree_freq_plot.pdf')
#     fig.show()

#     return fig



# def params_grid_search(graph, params_list):
#     '''
#         Performs grid search and outputs plots
#     '''

#     fig1 = go.Figure()
#     fig2 = go.Figure()


#     for params in tqdm(params_list):

#         [d, r, l, p, q] = params

#         # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
#         node2vec = Node2Vec(graph, dimensions=d, walk_length=l, num_walks=r, p=p, q=q, workers=8, quiet=True)  # Use temp_folder for big graphs
#         walks = node2vec.walks
        
#         # Degrees and walks' frequencies
#         degrees, cumulative_deg_prob = degrees_distribution_gen(graph)
#         node_freq_dict, visit_counts, cumulative_freq_prob = freq_gen(walks=walks)

#         fig1.add_trace(go.Scatter(
#         x=visit_counts,
#         y=cumulative_freq_prob,
#         mode='markers',
#         marker=dict(symbol='cross', size=5),
#         line=dict(width=2),
#         name=f'p={p}, q={q}',  # Add labels for each point
#         ))

#         fig2.add_trace(go.Scatter(
#             x=sorted(degrees),
#             y=sorted(node_freq_dict.values()),
#             mode='markers',
#             marker=dict(size=6),
#             line=dict(width=2),
#             name=f'p={p}, q={q}',  # Add labels for each point
#         ))


#     fig1.update_layout(
#         title=f"Log-log plot of visit frequency distribution for {graph.name} |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
#         xaxis_title=r"Vertex visitation count x",
#         yaxis_title=r"Probability of vertex appearing at least x times",
#         xaxis_type='log',  # Add log scale for x-axis
#         yaxis_type='log',  # Add log scale for y-axis
#         width=640,
#         height=480
#     )


#     fig2.update_layout(
#         title=f"Degree vs Frequency Plot for {graph.name} |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}",
#         xaxis_title="Degree",
#         yaxis_title="Frequency (Node Visit Count)",
#         width=640, 
#         height=480
#     )


#     fig1.show()
#     fig2.show()



# def freq_plot(visit_counts, cumulative_freq_prob):
#     '''
#         Makes a plot of the frequency distribution
#     '''
    
#     fig = go.Figure()

#     plt.rcParams['text.usetex'] = True # TeX rendering
#     plt.figure(figsize=(8, 5))
    
#     # Show dashed grid lines
#     plt.grid(color='lightgrey', linestyle='--')

#     # Set x-axis and y-axis to logarithmic scale
#     plt.xscale('log')
#     plt.yscale('log')

#     # Create a scatter plot
#     plt.scatter(x=visit_counts, y=cumulative_freq_prob, s=4, zorder=2)

#     # Set plot title and axis labels
#     plt.title("$\log$-$\log$ plot of visit frequency distribution")
#     plt.xlabel("Vertex visitation count $x$", fontdict={'size': 12})
#     plt.ylabel("Probability of vertex appearing at least $x$ times", fontdict={'size': 12})

#     # Save and display the plot
#     plt.savefig('figures/freq_plot.svg', dpi=300)
#     plt.savefig('figures/freq_plot.pdf', dpi=300)
    
#     plt.show()


# def degree_plot(cumulative_deg_prob):
#     '''
#         Makes a plot of the degree distribution
#     '''

#     plt.rcParams['text.usetex'] = True # TeX rendering
#     plt.figure(figsize=(6, 5))
    
#     # Show dashed grid lines
#     plt.grid(color='lightgrey', linestyle='--')

#     # Set x-axis and y-axis to logarithmic scale
#     plt.xscale('log')
#     plt.yscale('log')

#     # Create a scatter plot
#     plt.scatter(list(range(len(cumulative_deg_prob))), cumulative_deg_prob, s=4, zorder=2)

#     # Set plot title and axis labels
#     plt.title("$\log$-$\log$ plot of degree distribution")
#     plt.xlabel("Degree ($\log$ scale)")
#     plt.ylabel("Probability of Degree $\geq x$ ($\log$ scale)")

#     # # Save and display the plot
#     plt.savefig('figures/degree_plot.svg', dpi=300)
#     plt.savefig('figures/degree_plot.pdf', dpi=300)
    
#     plt.show()


# def degree_freq_plot(graph, degrees, node_freq_dict):
#     '''
#     Makes a scatter plot of the degrees and the node frequencies
#     '''

#     # Edit the font, font size, and axes width
#     # mpl.rcParams['font.family'] = 'Arial'
#     plt.rcParams['font.size'] = 12
#     # plt.rcParams['axes.linewidth'] = 2

#     # plt.rcParams.update({'font.size': 12})
#     plt.rcParams['text.usetex'] = True # TeX rendering
#     plt.figure(figsize=(8, 5))

#     # Show dashed grid lines
#     plt.grid(color='lightgrey', linestyle='--')

#     # Create a scatter plot
#     plt.scatter(sorted(degrees), sorted(node_freq_dict.values()), s=4, zorder=2)

#     # Set plot title and axis labels
#     plt.title(r"Degree vs Frequency Plot for {} $|V|={}$, $|E|={}$".format(graph.name, graph.number_of_nodes(), graph.number_of_edges()))
#     plt.xlabel("Degree")
#     plt.ylabel("Frequency (Node Visit Count)")

#     # Save and display the plot
#     plt.savefig('figures/degree_freq_plot.svg', dpi=300)
#     plt.savefig('figures/degree_freq_plot.pdf', dpi=300)

#     plt.show()