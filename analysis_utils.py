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
    sub_df = sub_df[['dataset', 'retraining_type', 'parameters', 'dynamic_update'] + perc_list + ['training_time']]

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
    
    plt.xlabel('Training time (s)', fontdict={'size': 20})
    plt.ylabel('Accuracy', fontdict={'size': 20})

    # plt.title(f'Scatter Plot of Accuracy-Time for {name} dataset')

    # Adjust the legend
    # plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.9))
    
    # Adjust spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'figures/accu_time_all_params_{name}_plot.pdf', dpi=100, bbox_inches='tight')

    plt.show()



def accu_speedup_dataset_plot(alL_stats_dict, dynamic_update):

    '''
        Plot accuracy-speedup bar charts for mutliple datasets
    '''

    stats_dict = alL_stats_dict[dynamic_update]

    # Extract keys and values
    labels = list(stats_dict.keys())
    accuracy_diff = [v[0] for v in stats_dict.values()]
    speedup = [v[1] for v in stats_dict.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

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
    ax.set_xticklabels(labels, fontsize=16)
    # ax.set_ylim([0, 100])
    ax.set_ylim([0, 1])
    ax.legend()

    # Reverse the legend items
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set font to Computer Modern
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text

    # Adjust the legend
    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.20))

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
                ax.scatter(x_plot, y_plot, c=colors[retraining_type], marker=markers[retraining_type], label=retraining_type)
                
                # Line plot
                for k in range(y_values.shape[0]):
                    ax.plot(x_values, y_values[k], c=colors[retraining_type], linestyle='-', linewidth=0.5)

                min_value = min(min_value, np.min(y_values))
                max_value = max(max_value, np.max(y_values))
                

            ax.set_xlabel('Training size', fontdict={'size': 20})
            ax.set_ylabel(f'{metric_name.capitalize()}-F1 score', fontdict={'size': 20})
            
            # Remove the current title
            ax.set_title('')
            
            if j%2 == 0:
                # Add horizontal title next to the y-axis
                ax.text(-0.35, 0.5, f'{dataset_name.capitalize()}', transform=ax.transAxes,
                        rotation=0, ha='center', va='center', fontsize='large', fontweight='bold')

            ax.set_ylim([0.85 * min_value, 1.05 * max_value])
            
            # Add light grey grid
            ax.grid(True, which='both', linestyle='--', linewidth=0.75, color='lightgrey')
            ax.set_axisbelow(True)

    # # Adjust the legend
    # plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.20))

    # Adjust spacing and save the plot
    # plt.tight_layout()
    plt.savefig('figures/datasets_scores.svg', dpi=100, bbox_inches='tight')
    plt.savefig('figures/datasets_scores.pdf', dpi=100, bbox_inches='tight')
    
    # Show the plot
    plt.show()


def dataset_walk_plot(dynamic_update, metric, all_dataset_walk_dict):
    '''
        Plots bar charts of accuracy and time (metric) for different
        walk lengths for each dataset
    '''

    data = all_dataset_walk_dict

    index = {'accuracy': 0, 'time': 1}[metric]
    # suffix = {'accuracy': ' (\%)', 'time': ' (s)'}[metric]
    suffix = {'accuracy': '', 'time': ' (s)'}[metric]

    datasets = list(data.keys())
    metric_40 = [data[ds].get(40.0, [0, 0])[index] for ds in datasets]
    metric_80 = [data[ds].get(80.0, [0, 0])[index] for ds in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    # Accuracy bar chart
    fig, ax = plt.subplots()
    ax.bar(x - width/2, metric_40, width, label='Walk length = 40', color='#2053A6')
    ax.bar(x + width/2, metric_80, width, label='Walk length = 80', color='#FF6011')
    # ax.set_xlabel('Dataset')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy by Dataset and Metric')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=16)
    ax.set_ylabel(metric.title()+suffix, fontsize=16)

    ax.legend()

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust the legend
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.20))

    # Adjust spacing and save the plot
    plt.tight_layout()
    plt.savefig(f'figures/{metric}_walk_{dynamic_update}_plot.pdf', dpi=100, bbox_inches='tight')

    plt.show()