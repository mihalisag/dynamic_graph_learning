# * This is a file containing the analysis helper functions used in the project *

# -- Imports -- ## 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import ast


def extract_elements(param_str):
    '''
        Function to convert string to tuple and extract required elements
    '''
    tuple_obj = ast.literal_eval(param_str)
    return tuple_obj[1], tuple_obj[2]


def accu_time_df_gen(results_df, config):
    '''
        Generates DataFrame of accuracy and training time
        for a specific configuration for a parameter set
    '''
    sub_df = results_df.copy()

    for key in config:
        sub_df = sub_df[sub_df.loc[:, key] == config[key]]
        
    sub_df = sub_df.reset_index(drop=True)
    perc_list = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    sub_df = sub_df[['dataset', 'retraining_type', 'parameters', 'dynamic_update'] + perc_list + ['training_time', 'removal_process']]

    # Calculate the average accuracy for each row
    accuracy_columns = perc_list
    sub_df['average_accuracy'] = sub_df[accuracy_columns].mean(axis=1)

    sub_df.drop(columns=accuracy_columns, inplace=True)
    columns = [col for col in sub_df.columns if col != 'training_time'] + ['training_time']
    sub_df = sub_df[columns]

    return sub_df


def mult_params_accu_time_df_gen(results_df, config, all_parameters):
    '''
        Same as accuracy_time_df_gen but for multiple parameters
    '''

    all_df = pd.DataFrame()

    for parameters in all_parameters:
        config['parameters'] = parameters
        all_df = pd.concat([all_df, accu_time_df_gen(results_df, config)])

    all_df.columns.name = config['dataset']
    all_df = all_df.reset_index(drop=True)

    return all_df
    

def accu_time_plot(all_df):
    '''
        Plots accuracy-time scatter plot
    '''

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # Dataset name
    name = all_df.columns.name

    # Map colors
    colors = all_df['retraining_type'].map({'global': '#2053A6', 'local': '#FF6011'})

    # Scatter plot for each category
    for retraining_type, color in {'global': '#2053A6', 'local': '#FF6011'}.items():
        subset = all_df[all_df['retraining_type'] == retraining_type]
        plt.scatter(subset['training_time'], subset['average_accuracy'], c=color, label=retraining_type)

    # Show dashed grid lines
    plt.rc('axes', axisbelow=True)
    plt.grid(color='lightgrey')
    
    plt.xlabel('Time(s)', fontdict={'size': 20})
    plt.ylabel('Accuracy', fontdict={'size': 20})

    # plt.title(f'Scatter Plot of Accuracy-Time for {name} dataset')

    # Adjust the legend
    # plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.9))
    
    # Adjust spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'figures/accu_time_all_params_{name}_plot.pdf', dpi=100, bbox_inches='tight')
    plt.savefig(f'figures/accu_time_all_params_{name}_plot.svg', dpi=100, bbox_inches='tight')

    plt.show()


def accu_speedup_dataset_plot(all_stats_dict, dynamic_update):

    '''
        Plot accuracy-speedup bar charts for mutliple datasets
    '''

    stats_dict = all_stats_dict[dynamic_update]

    # Extract keys and values
    labels = list(sorted(stats_dict.keys()))
    accuracy_diff = [v[0] for v in stats_dict.values()]
    speedup = [v[1] for v in stats_dict.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # Create the figure and the axes
    fig, ax = plt.subplots()

    # Plot the speedup
    rects1 = ax.bar(x + width/2, speedup, width, label='Speedup', color='#0CC03E')

    # Plot the accuracy difference
    rects2 = ax.bar(x - width/2, accuracy_diff, width, label='Accuracy Difference', color='#FF1B20')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Dataset', fontdict={'size': 20})
    # ax.set_ylabel('Percentage (%)', fontdict={'size': 20})
    # ax.set_title(f'Accuracy Difference and Speedup by Dataset for {dynamic_update}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylim([0, 1])
    # ax.legend()

    # Set the fontsize for y-axis ticks
    ax.tick_params(axis='y', labelsize=20)  # Increase fontsize for y-axis ticks

    # # Reverse the legend items
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1])

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust the legend
    # plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.20))

    # Adjust spacing and save the plot
    plt.tight_layout()
    plt.savefig(f'figures/accu_speedup_dataset_{dynamic_update}_plot.svg', dpi=100, bbox_inches='tight')
    plt.savefig(f'figures/accu_speedup_dataset_{dynamic_update}_plot.pdf', dpi=100, bbox_inches='tight')
        
    # Show the plot
    plt.show()


def table_df_gen(results_df, config):
    '''
        Generates table df (check example)
    '''

    table_df = results_df.copy()

    for key in config:
        table_df = table_df[table_df.loc[:, key] == config[key]]

    table_df = table_df.reset_index(drop=True)
    table_df.columns.name = config['dataset']
    
    return table_df


def table_scores_plot(table_df, metric_name):
    '''
    Plots scatter plots with lines connecting the markers 
    for different retraining types
    '''

    # plt.style.use('ggplot')
    dataset_name = table_df.columns.name

    fig, ax = plt.subplots()
    colors = {'global': '#2053A6', 'local': '#FF1B20'}
    markers = {'global': 'o', 'local': '^'}

    # X-axis values
    x_values = 1 - np.arange(0.1, 1, 0.1)

    # Used for y axis range limits
    min_value = np.inf
    max_value = -np.inf

    for retraining_type in table_df['retraining_type'].unique():
        subset = table_df[table_df['retraining_type'] == retraining_type]
        y_values = subset.iloc[:, 5:14].values
        x_plot = np.tile(x_values, (y_values.shape[0], 1)).flatten()
        y_plot = y_values.flatten()
        
        # Scatter plot
        ax.scatter(x_plot, y_plot, c=colors[retraining_type], marker=markers[retraining_type], label=retraining_type)
        
        # Line plot
        for i in range(y_values.shape[0]):
            ax.plot(x_values, y_values[i], c=colors[retraining_type], linestyle='-', linewidth=0.5)

        min_value = min(min_value, np.min(y_values))
        max_value = max(max_value, np.max(y_values))

    ax.set_xlabel('Training size', fontdict={'size': 20})
    ax.set_ylabel(f'{metric_name} score', fontdict={'size': 20})
    ax.set_title(f'{dataset_name}')
    # ax.legend(title='Retraining Type')
    
    # Change the lower y limit to not be on the edge?
    ax.set_ylim([0.95 * min_value, 1.05 * max_value])
    
    # Add light grey grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.75, color='lightgrey')
    ax.set_axisbelow(True)

    plt.show()



def exp_table_scores_plot(dfs, dataset_names, metrics=['micro', 'macro']):
    '''
    Plots scatter plots with lines connecting the markers for different retraining types
    for a list of DataFrames.

    Parameters:
    dfs (list of DataFrames): List of DataFrames containing the data to plot.
    dataset_names (list of str): List of dataset names corresponding to each DataFrame.
    metrics (list of str): List of metric names to use in the plot (default: ['micro', 'macro']).
    '''

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # X-axis values
    x_values = 1 - np.arange(0.1, 1, 0.1)
    colors = {'global': '#2053A6', 'local': '#FF1B20'}
    markers = {'global': 'o', 'local': '^'}

    # Determine the number of rows and columns for the subplots
    num_datasets = len(dfs)
    num_metrics = len(metrics)

    fig, axes = plt.subplots(num_datasets, num_metrics, figsize=(12, num_datasets * 4), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, (df, dataset_name) in enumerate(zip(dfs, dataset_names)):
        for j, metric_name in enumerate(metrics):
            ax = axes[i, j]

            # Filter the DataFrame for the current metric
            subset_df = df[df['metric'] == metric_name]

            min_value = np.inf
            max_value = -np.inf

            for retraining_type in subset_df['retraining_type'].unique():
                subset = subset_df[subset_df['retraining_type'] == retraining_type]
                y_values = subset.iloc[:, 5:14].values
                x_plot = np.tile(x_values, (y_values.shape[0], 1)).flatten()
                y_plot = y_values.flatten()

                # Scatter plot
                ax.scatter(x_plot, y_plot[::-1], c=colors[retraining_type], marker=markers[retraining_type], label=retraining_type)
                
                # Line plot
                for k in range(y_values.shape[0]):
                    ax.plot(x_values, y_values[k][::-1], c=colors[retraining_type], linestyle='-', linewidth=0.5)

                min_value = min(min_value, np.min(y_values))
                max_value = max(max_value, np.max(y_values))
                

            ax.set_xlabel('Training size', fontdict={'size': 20})
            ax.set_ylabel(f'{metric_name.capitalize()}-F1 score', fontdict={'size': 20})
            
            # Remove the current title
            ax.set_title('')
            
            # if j%2 == 0:
            #     # Add horizontal title next to the y-axis
            #     ax.text(-0.35, 0.5, f'{dataset_name.capitalize()}', transform=ax.transAxes,
            #             rotation=0, ha='center', va='center', fontsize='large', fontweight='bold')

            ax.set_ylim([0.85 * min_value, 1.05 * max_value])
            
            # Add light grey grid
            ax.grid(True, which='both', linestyle='--', linewidth=0.75, color='lightgrey')
            ax.set_axisbelow(True)

    # # Adjust the legend
    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(-0.25, -0.40))

    # Adjust spacing and save the plot
    # plt.tight_layout()
    plt.savefig('figures/datasets_scores.svg', dpi=100, bbox_inches='tight')
    plt.savefig('figures/datasets_scores.pdf', dpi=100, bbox_inches='tight')
    
    # Show the plot
    plt.show()


def dataset_walk_scores_gen(results_df, dynamic_update):
    '''
        Generates macro scores for every dataset
        for different walk lengths
    '''

    # all_datasets = list(results_df['dataset'].unique())
    # dynamic_update = 'extend'   

    all_datasets = ['blog_catalog', 'wikipedia', 'PPI', 'cora']
    all_parameters = list(results_df['parameters'].unique())

    config = {  'dynamic_update': dynamic_update,
                'metric': 'macro',
                'num_different_nodes': 512}

    all_dataset_walk_dict = {}

    for dataset in all_datasets:
        config['dataset'] = dataset
        
        all_df = mult_params_accu_time_df_gen(results_df, config, all_parameters)

        # Apply the function to the 'parameters' column and create new columns
        all_df[['walk_length', 'walks_num']] = all_df['parameters'].apply(lambda x: pd.Series(extract_elements(x)))

        all_df = all_df[['dataset', 'retraining_type', 'dynamic_update', 'walk_length', 'walks_num', 'average_accuracy', 'training_time']]
        all_df = all_df.loc[all_df['walk_length'] != 8]

        walk_df = all_df.groupby(['walk_length']).mean(['average_accuracy', 'training_time']).reset_index()

        dataset_walk_metric_dict = {row['walk_length']: [row['average_accuracy'], row['training_time']] for _, row in walk_df.iterrows()}

        all_dataset_walk_dict[dataset] = dataset_walk_metric_dict

    
    return all_dataset_walk_dict


def dataset_walk_plot(dynamic_update, metric, all_dataset_walk_dict):
    '''
        Plots bar charts of accuracy and time (metric) for different
        walk lengths for each dataset
    '''

    data = all_dataset_walk_dict

    index = {'accuracy': 0, 'time': 1}[metric]
    # suffix = {'accuracy': ' (\%)', 'time': ' (s)'}[metric]
    suffix = {'accuracy': '', 'time': ' (s)'}[metric]

    datasets = list(sorted(data.keys()))
    metric_40 = [data[ds].get(40.0, [0, 0])[index] for ds in datasets]
    metric_80 = [data[ds].get(80.0, [0, 0])[index] for ds in datasets]

    width = 0.35
    spacing = 0.3  # additional spacing between bars (adjust this value as needed)
    x = np.arange(0, len(datasets) * (1 + spacing), 1 + spacing) # Adjust x to include spacing

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # Accuracy bar chart
    fig, ax = plt.subplots()

    ax.bar(x - width/2, metric_40, width, label='Walk length = 40', color='#2053A6')
    ax.bar(x + width/2, metric_80, width, label='Walk length = 80', color='#FF6011')
    # ax.set_xlabel('Dataset')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy by Dataset and Metric')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=20)
    ax.set_ylabel(metric.title()+suffix, fontsize=22)
    # ax.set_ylim([0, 1])

    # ax.legend()

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set the fontsize for y-axis ticks
    ax.tick_params(axis='y', labelsize=20)  # Increase fontsize for y-axis ticks

    # Adjust the legend
    # plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.20))

    # Adjust spacing and save the plot
    plt.tight_layout()
    plt.savefig(f'figures/{metric}_walk_{dynamic_update}_plot.svg', dpi=100, bbox_inches='tight')
    plt.savefig(f'figures/{metric}_walk_{dynamic_update}_plot.pdf', dpi=100, bbox_inches='tight')

    plt.show()


def dataset_removal_plot(sub_df):
    '''
        Plots the average accuracy for every dataset and for
        different node removal processes
    '''

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # Define colors
    process_colors = {
        'degree_centrality': '#2053A6',
        'betweenness_centrality': '#FF6011',
        'random': '#0CC03E'
    }
    

    # Define the order of datasets
    labels = ['PPI', 'blog_catalog', 'cora', 'wikipedia']
    removal_processes = sub_df['removal_process'].unique()
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    # Create the figure and the axes
    fig, ax = plt.subplots() #figsize=(12, 8))

    # Plot bars for each removal process
    for i, removal_process in enumerate(removal_processes):
        # Filter data for the current removal process
        filtered_data = sub_df[sub_df['removal_process'] == removal_process]
        accuracy_values = [filtered_data[filtered_data['dataset'] == label]['average_accuracy'].values[0] if label in filtered_data['dataset'].values else 0 for label in labels]
        
        # Bar color
        bar_color = process_colors[removal_process]

        # Plot the bars
        rects = ax.bar(x + i * width - (len(removal_processes) / 2) * width, accuracy_values, width, label=removal_process, color=bar_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Dataset', fontdict={'size': 20})
    # ax.set_ylabel('Average Accuracy', fontdict={'size': 20})
    # ax.set_title('Average Accuracy by Dataset and Removal Process')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylim([0, 1])
    ax.legend()

    # Reverse the legend items
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust the legend
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3), fontsize=12)

    # Adjust spacing and save the plot
    plt.tight_layout()
    plt.savefig('figures/accuracy_by_dataset_removal_process.svg', dpi=100, bbox_inches='tight')
    plt.savefig('figures/accuracy_by_dataset_removal_process.pdf', dpi=100, bbox_inches='tight')

    # Show the plot
    plt.show()