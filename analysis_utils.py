import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import ast


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

    # Dataset name
    name = all_df.columns.name

    # Map colors
    colors = all_df['retraining_type'].map({'global': 'blue', 'local': 'red'})

    # Scatter plot for each category
    for retraining_type, color in {'global': 'blue', 'local': 'red'}.items():
        subset = all_df[all_df['retraining_type'] == retraining_type]
        plt.scatter(subset['training_time'], subset['average_accuracy'], c=color, label=retraining_type)

    plt.xlabel('Training time')
    plt.ylabel('Accuracy')
    plt.title(f'Scatter Plot of Accuracy-Time for {name} dataset')
    plt.legend(title='Retraining Type')
    plt.show()



def speed_accu_plot(alL_stats_dict, dynamic_update):

    '''
        Plot
    '''

    stats_dict = alL_stats_dict[dynamic_update]

    # Extract keys and values
    labels = list(stats_dict.keys())
    accuracy_diff = [v[0] for v in stats_dict.values()]
    speedup = [v[1] for v in stats_dict.values()]

    # Find the maximum value for normalization
    max_value = max(max(accuracy_diff), max(speedup))

    # Normalize the values
    normalized_accuracy_diff = accuracy_diff
    normalized_speedup = speedup

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    # Create the figure and the axes
    fig, ax = plt.subplots()

    # Plot the normalized speedup
    rects1 = ax.bar(x + width/2, normalized_speedup, width, label='Speedup (%)', color='#1f77b4')

    # Plot the normalized accuracy difference
    rects2 = ax.bar(x - width/2, normalized_accuracy_diff, width, label='Accuracy Difference (%)', color='#ff7f0e')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Accuracy Difference and Speedup by Dataset for {dynamic_update}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.legend()

    # Reverse the legend items
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust spacing and save the plot
    plt.tight_layout()
    plt.savefig('figures/freq_plot.svg', dpi=100, bbox_inches='tight')
    plt.savefig('figures/freq_plot.pdf', dpi=100, bbox_inches='tight')
        
    # Show the plot
    plt.show()