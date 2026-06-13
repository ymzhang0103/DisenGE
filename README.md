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

## Quantitative Performance Comparison on Alkane Carbonyl Datasets
We conducted additional experiments on the Alkane Carbonyl dataset. The experimental results are presented below. 

The target GNN consists of three GCN layers, each with a hidden dimension of 20. Node representations are $L_2$-normalized at each layer and activated by ReLU. The graph-level representation is obtained by concatenating max-pooling and mean-pooling readouts. The test accuracy is 0.982.
<img width="20%" src ="./results/Alkane_Carbonyl-f.png"/><img width="20%" src ="./results/Alkane_Carbonyl-f+.png"/><img width="20%" src ="./results/Alkane_Carbonyl-f-.png"/><img width="20%" src ="./results/nec-delEdge-Alkane_Carbonyl-f+.png"/><img width="20%" src ="./results/robust-Alkane_Carbonyl-fvaluef.png"/>

As can be seen, DisenGE achieves the best $\textit{Fvalue Fidelity}@k$ ($k < 0.85$), $\textit{NEC}$, and $\textit{Robust Fidelity}$, particularly under high sparsity ($k < 0.15$). Since the primary goal of GNN explanations is identifying the minimal predictive subset[1-2], DisenGE's superiority at small $k$ proves its ability to pinpoint critical substructures efficiently. Even at larger $k$, it consistently ranks in the top two.

Specifically, DisenGE's rapid early ascent on $Fidelity+^{prob}@k$ and $Nec$ demonstrates that it quickly extracts necessary components. Meanwhile, its consistently low $Fidelity-^{prob}@k$ at small $k$ ensures strong sufficiency without redundancy. For $\textit{Robust Fidelity}$, traditional baselines (e.g., GNNE, MixupE, $CF_2$, RDPE) collapse early, and MoE plateaus around $k=0.15$. In contrast, DisenGE ascends significantly faster than PGE and GEM in early sparse stages, ultimately stabilizing in the top tier.

[1] H. Yuan, H. Yu, S. Gui, S. Ji, Explainability in graph neural networks: A taxonomic survey, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (5) (2023) 5782–5799.

[2] Longa A, Azzolin S, Santin G, et al., Longa, A, Azzolin, S, Santin, G, Cencetti, G, Lio, P, Lepri, B, Passerini, A, Explaining the explainers in graph neural networks: a comparative study, ACM Computing Surveys 57.5 (2025): 1-37.
