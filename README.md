# Dynamic Graph Learning

This repository contains the code used in my master thesis named "Optimising node2vec in Dynamic Graphs Through Local Retraining" at the University of Twente (Master of Applied Mathematics, Spring 2024).


## Project Overview

The thesis investigates how to make node2vec more efficient for dynamic graphs by focusing on local retraining of affected areas rather than retraining the entire network after updates. The key research questions addressed are:

1. How can we make the node2vec algorithm more efficient in dynamic graphs?
2. How does approximating node2vec in dynamic graphs affect accuracy and training time?
3. How is accuracy affected depending on the type of dynamic update (removing nodes randomly or according to a specific graph statistic)?

## Key Findings

- Local retraining of node2vec on dynamic graphs achieves similar accuracy to global retraining while providing significant speedup (70-85% faster).
- Random node removal strategy outperforms centrality-based removal in terms of accuracy and computational efficiency.
- Shorter random walk lengths (40 vs 80) offer comparable accuracy with substantial time savings.
- Optimal hyperparameters were found to be (d=128, r=40, l=40, p=0.25, q=1) across datasets.

## Datasets

The experiments were conducted on four datasets:

- BlogCatalog
- PPI (Protein-Protein Interaction)
- Wikipedia
- Cora

## Implementation Details

- Modified version of eliorc's node2vec implementation
- Custom algorithms for dynamic graph generation, extending, and pruning
- Evaluation metrics: Micro-F1 and Macro-F1 scores

## Limitations and Future Work

- Dataset imbalances, particularly with the larger BlogCatalog dataset
- Limited exploration of graph structures and statistics
- Potential for new methods of extending and pruning graphs


## Repository Structure

- `datasets/`: Contains the graph datasets used in experiments (BlogCatalog, PPI, Wikipedia, Cora)
- `draft_notebooks/`: Jupyter notebooks containing all experiments and analysis
- `analysis_utils.py`: Helper functions for data analysis, generating DataFrames, and statistical calculations
- `main_utils.py`: Core functions for graph manipulation, node2vec implementation, and evaluation metrics
- `plot_utils.py`: Functions for visualizing results through various plots

## Hardware

Experiments were run on a Linux server with:
- AMD EPYC 7713P 64-Core Processor
- 128GB RAM
- 64 parallel workers


## Reference

[1] Michail Angelos Goulis, [Optimising node2vec in Dynamic Graphs Through Local Retraining](https://essay.utwente.nl/103078/1/Goulis_MA_EEMCS.pdf), University of Twente 2024.