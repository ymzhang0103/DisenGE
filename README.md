# DisenGE
Code for the paper "DisenGE: Learning Multiple Collaborative Subgraphs for Faithful Graph Neural Network Explanation"

## Overview
The project contains the following folders and files.
- codes
	- GNNmodels: Code for the GNN model to be explained.
 	- explain: Code for the disentangled Explainer.
	- load_dataset.py: Load datasets.
	- Configures.py: Parameter configuration of the GNN model to be explained.
	- metrics.py: Metrics of the evaluation.
	- plot_utils.py: Plot a diagram of a case for instance-level explanations.
- checkpoint: To facilitate the reproduction of the experimental results in the paper, we provide the trained GNNs to be explained in this fold.

## Prerequisites
- python >= 3.9
- torch >= 1.12.1+cu113
- torch-geometric >= 2.2.0

## To run
- Run train_GNNNets.py to train the GNNs to be explained. Change parameter **dataset** per demand.
- Run main.py to explain the pre-trained GNNs. Change **parameters** per demand.
