import time
import os
import torch
import torch.nn as nn
import torch_geometric as pyg
import numpy as np
import networkx as nx 
import random

from typing import *
from tqdm import tqdm
#from omegaconf import DictConfig
from functools import cached_property
from collections import defaultdict
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCN, global_mean_pool, Linear

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Explanation(Data):
    """
    Save information of explainer subgraph.
    """
    plot_dict = {
        "SEED"       : 10,
        "node_dict"  : {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br',
                        7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'},
        # "node_colors": ("#E4A031, #D68438, #C76B60, #B55384, #7C4D77, #474769, #26445E, "
        #                 "#4C7780, #73A5A2, #F6E2C1, #F3DBC1, #B2B6C1, #D6E2E2, #F0EFED").split(', '),
        "node_colors": ['#E49D1C', '#4970C6', '#73A5A2', '#FF5357',
                        'green',  '#F6E2C1', '#F0EA00'],
        "edge_colors": ['gray'],
        "motif_colors": ['red', 'lime']
    }

class SubgraphBuilding(object):
    r"""
    Graph building/Perturbation
    `graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
    """
    def __init__(self, subgraph_building_method: str):
        self.subgraph_building_method = subgraph_building_method

    @staticmethod
    def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
        """subgraph building through masking the unselected nodes with zero features"""
        ret_X = X * node_mask.unsqueeze(1)
        return ret_X, edge_index

    @staticmethod
    def graph_build_split(X, edge_index, node_mask: torch.Tensor):
        """subgraph building through spliting the selected nodes from the original graph"""
        ret_X = X
        row, col = edge_index
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        ret_edge_index = edge_index[:, edge_mask]
        return ret_X, ret_edge_index

    @staticmethod
    def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
        """subgraph building through removing the unselected nodes from the original graph"""
        ret_X = X[node_mask == 1]
        ret_edge_index, _ = pyg.utils.subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
        return ret_X, ret_edge_index

    def __call__(self, *args, **kwargs):
        if self.subgraph_building_method == "zero_filling":
            return self.graph_build_zero_filling(*args, **kwargs)
        elif self.subgraph_building_method == "split":
            return self.graph_build_split(*args, **kwargs)
        elif self.subgraph_building_method == "remove":
            return self.graph_build_remove(*args, **kwargs)
        else:
            raise NotImplementedError(f"the method {self.subgraph_building_method} doesn't exist")

class MaskedDataset(Dataset):
    def __init__(self, data: Data, masks: List[Tensor], subgraph_building_method: str):
        super().__init__()

        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y

        if not isinstance(masks, list):
            masks = [masks]
        self.masks = torch.cat(masks, dim=0).float().to(self.device)
        self.subgraph_building_func = SubgraphBuilding(subgraph_building_method)

    def len(self):
        return self.masks.shape[0]

    def get(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(
            self.x, self.edge_index, self.masks[idx]
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)

        return masked_data

def get_explanation_syn(
    data: Data,
    edge_mask: Tensor,
    max_nodes: Optional[int]  = None,
    max_edges: Optional[int]  = None,
    node_list: Optional[List] = None,
    edge_list: Optional[List] = None,
    **kwargs
):
    """Create an explanation graph from the edge_mask.
    Args:
        data (PyG data object): the initial graph as Data object
        edge_mask (Tensor): the explanation mask(Hard mask)
        node_list:  node test_indices
        max_nodes: max number of nodes
        max_edges: max number of edges
        edge_list: edge list of explanation subgraph
    Returns:
        G_masked (networkx graph): explanatory subgraph
    """
    G = nx.Graph()

    if edge_list is not None:
        G.add_edges_from(edge_list)
        return G

    ### remove self loops
    edge_index = data.edge_index.cpu().numpy()
    assert edge_index.shape[-1] == edge_mask.shape[-1]
    self_loop_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, self_loop_mask]
    edge_mask = edge_mask[self_loop_mask]

    order = (-edge_mask).argsort()

    if node_list is not None:
        G.add_nodes_from(node_list)

    for i in order:
        u, v = edge_index[:, i]
        if max_edges is not None and G.number_of_edges() >= max_edges:
            break

        if max_nodes is not None and G.number_of_nodes() >= max_nodes and \
                (u not in G.nodes or v not in G.nodes):
            continue

        if edge_mask[i] == 0:
            continue

        G.add_edge(u, v)

    return G

class FidelityAlpha(object):
    r"""
        Idea `robust fidelity metrics` came from https://arxiv.org/pdf/2310.01820v1.pdf.
        Code is based on the paper `MEGA: Explaining Graph Neural Networks via Structure-aware Interaction Index`,
        I just rearrange theirs code.
    """
    def __init__(
        self,
        model:       nn.Module,
        alpha:       float = 0.8,
        num_samples: int = 2000,
        subgraph_building_method: str = 'split'
    ):
        self.model = model
        self.alpha = alpha
        self.num_samples = num_samples
        self.subgraph_building_method = subgraph_building_method

    @staticmethod
    def alpha_sampling(G: nx.Graph, alpha: float) -> nx.Graph:
        r"""
        Random sample a graph under parameter alpha.
        """
        H = nx.Graph()

        for u in G.nodes:
            if np.random.rand() <= alpha:
                H.add_node(u)
        for u, v in G.edges:
            if u in H.nodes and v in H.nodes:
                H.add_edge(u, v)

        return H

    @staticmethod
    def graph_minus(G1: nx.Graph, G2: nx.Graph) -> nx.Graph:
        r"""
        Get graph with explanation graph.
        """
        H = nx.Graph()
        for u in G1.nodes:
            if u not in G2.nodes:
                H.add_node(u)

        for u, v in G1.edges:
            if u in H.nodes and v in H.nodes:
                H.add_edge(u, v)
        return H

    @staticmethod
    def graph_plus(G1: nx.Graph, G2: nx.Graph, edge_list) -> nx.Graph:
        r"""
        Get graph without explanation graph.
        """
        H = nx.Graph()
        for u in G1.nodes:
            H.add_node(u)
        for u in G2.nodes:
            H.add_node(u)

        for u, v in edge_list:
            if u in H.nodes and v in H.nodes:
                H.add_edge(u, v)

        return H
    
    @torch.no_grad()
    def compute_payoffs(self, data: Explanation, masks: List[Tensor], corn_node_id: Optional[Tensor] = None):
        r"""
        Getting prediction.
        """
        masked_dataset = MaskedDataset(data, masks, self.subgraph_building_method)

        # N = max(data_.num_nodes for data_ in masked_dataset)
        # masked_dataset = [pre_transform_mutag(data_, 417) for data_ in masked_dataset]

        masked_dataset = DataLoader(masked_dataset, batch_size=128, shuffle=False)
        masked_payoff_list = []
        for masked_data in masked_dataset:
            masked_data.to(data.x.device)
            if corn_node_id is not None: 
                #preds = self.model(masked_data.x, masked_data.edge_index, batch=masked_data.batch).softmax(-1)[corn_node_id, data.y].unsqueeze(dim=-1)
                preds = self.model(masked_data)[1].softmax(-1)[corn_node_id, data.y].unsqueeze(dim=-1)
            else: 
                #preds = self.model(masked_data.x, masked_data.edge_index, batch=masked_data.batch).softmax(-1)[:, data.y]
                preds = self.model(masked_data)[1].softmax(-1)[:, data.y]
            masked_payoff_list.append(preds)
        return torch.cat(masked_payoff_list).detach().cpu().numpy()

    def fidelity_alpha_plus(self, 
                            data: Explanation, 
                            expl_graph: nx.Graph, 
                            ori_prob: Optional[Tensor] = None, 
                            corn_node_id: Optional[Tensor] = None) -> float:
        graph    = pyg.utils.to_networkx(data, to_undirected=True)
        if ori_prob is None:
            if corn_node_id is None:
                #ori_prob = self.model(data.x, data.edge_index).softmax(dim=-1)[:, data.y]
                ori_prob = self.model(data)[1].softmax(dim=-1)[:, data.y]
            else: 
                #ori_prob = self.model(data.x, data.edge_index).softmax(dim=-1)[corn_node_id][data.y]
                ori_prob = self.model(data)[1].softmax(dim=-1)[corn_node_id][data.y]
            ori_prob = ori_prob.cpu().item()
        mask_lst = []

        for _ in range(self.num_samples):
            expl_graph_plus = self.alpha_sampling(expl_graph, self.alpha)
            non_expl_graph  = self.graph_minus(graph, expl_graph_plus)
            node_mask       = torch.zeros(data.num_nodes)
            node_mask[torch.LongTensor(list(non_expl_graph.nodes))] = 1
            mask_lst.append(node_mask)

        excluded_masks = torch.vstack(mask_lst)
        excluded_probs = self.compute_payoffs(data, excluded_masks, corn_node_id)
        scores = abs(ori_prob - excluded_probs)

        if len(scores) != 0:
            return np.mean(scores)
        else:
            return 0

    def fidelity_alpha_minus(self, 
                             data: Explanation, 
                             expl_graph: nx.Graph, 
                             ori_prob: Optional[Tensor] = None, 
                             corn_node_id: Optional[Tensor] = None) -> float:
        graph = pyg.utils.to_networkx(data, to_undirected=True)
        if ori_prob is None:
            if corn_node_id is None:
                #ori_prob = self.model(data.x, data.edge_index).softmax(dim=-1)[:, data.y]
                ori_prob = self.model(data)[1].softmax(dim=-1)[:, data.y]
            else: 
                #ori_prob = self.model(data.x, data.edge_index).softmax(dim=-1)[corn_node_id][data.y]
                ori_prob = self.model(data)[1].softmax(dim=-1)[corn_node_id][data.y]
            ori_prob = ori_prob.cpu().item()
        mask_lst = []

        for _ in range(self.num_samples):
            non_expl_graph       = self.graph_minus(graph, expl_graph)
            non_expl_graph_minus = self.alpha_sampling(non_expl_graph, 1 - self.alpha)
            expl_graph_minus     = self.graph_plus(non_expl_graph_minus, expl_graph, graph.edges)
            node_mask            = torch.zeros(data.num_nodes)
            node_mask[torch.LongTensor(list(expl_graph_minus.nodes))] = 1
            mask_lst.append(node_mask)

        included_masks = torch.vstack(mask_lst)
        included_probs = self.compute_payoffs(data, included_masks, corn_node_id)
        scores = abs(ori_prob - included_probs)

        if len(scores) != 0:
            return np.mean(scores)
        else:
            return 0

    def __call__(self, data: Explanation, **kwargs) -> Tuple[float, float, float]:
        r"""

        Args:
            data: pyg Data
            **kwargs:
                node_list: List[int]
                max_nodes: int
                max_edges: int
                ori_pred: float
                edge_list: List[Tuple[int, int]]
        """
        assert (kwargs.get('max_nodes') is not None or
                kwargs.get('edge_list') is not None or
                kwargs.get('ground_truth_mask') is not None)
        # if data.ground_truth_mask.sum() == 0: return 0., 0., 0.
        # if kwargs.get('max_nodes') is None:
        #     kwargs['max_nodes'] = (data.edge_index[:, data.ground_truth_mask.bool()]
        #                            .flatten()
        #                            .unique()
        #                            .size(0))

        expl_graph = get_explanation_syn(data, data.edge_mask, **kwargs)
        fid_plus   = self.fidelity_alpha_plus( data, expl_graph, kwargs.get('ori_prob'), kwargs.get('corn_node_id'))
        fid_minus  = self.fidelity_alpha_minus(data, expl_graph, kwargs.get('ori_prob'), kwargs.get('corn_node_id'))
        fid_delta  = fid_plus - fid_minus

        return fid_delta, fid_plus, fid_minus

class ExplanationEvaluator(object):
    r"""
    Collect explanations information, and then compute the metrics.
    """
    def __init__(
        self, 
        gnn_model:       nn.Module,
        alpha:          float = 0.4,
        num_samples:    int = 2000,
        subgraph_building_method: str = 'split'
    ):
        self.gnn_model = gnn_model

        self.alpha_func = FidelityAlpha(gnn_model, alpha, num_samples, subgraph_building_method)
        self.ori_func   = FidelityAlpha(gnn_model, 1, 1, subgraph_building_method)

        # time consume
        self.start_time  = dict()
        self.finish_time = dict()
        # for roc auc
        self.target_masks : List[Tensor] = []
        self.predict_masks: List[Tensor] = []
        self.roc_auc_score: Optional[float] = None
        # for others
        self.related_preds = defaultdict(list)
        self.results_dict  = {}

    def reset(self):
        # time consume
        self.start_time  = dict()
        self.finish_time = dict()
        # for roc auc
        self.target_masks : List[Tensor] = []
        self.predict_masks: List[Tensor] = []
        self.roc_auc_score: Optional[float] = None
        # for others
        self.related_preds = defaultdict(list)
        self.results_dict  = {}

    def fidelity(self, data: Explanation, mode: str = 'alpha', **kwargs):
        if mode == 'alpha':
            results = self.alpha_func(data, **kwargs)
            self.related_preds['fid_delta'].append(results[0])
            self.related_preds['fid_plus'].append(results[1])
            self.related_preds['fid_minus'].append(results[2])
        elif mode == 'ori':
            results = self.ori_func(data, **kwargs)
            self.related_preds['ori_fid_delta'].append(results[0])
            self.related_preds['ori_fid'].append(results[1])
            self.related_preds['ori_fid_inv'].append(results[2])
        else:
            raise NotImplementedError
        
    def get_average(self, metric: str) -> Tuple[float, float]:
        if metric not in self.related_preds.keys():
            raise ValueError(f"there is no metric named: {metric}")

        score = np.array(self.related_preds[metric])
        return score.mean().item(), score.std().item()

    def get_summarized_results(self):
        for metric in self.related_preds.keys():
            self.results_dict[metric] = self.get_average(metric)
    
    def collect(self, data: Explanation, metric: str = 'acc', **kwargs):
        if metric == 'fid':
            self.fidelity(data, mode='alpha', **kwargs)
            self.fidelity(data, mode='ori', **kwargs)
        if self.roc_auc_score is None:
            # remove self loops
            self_loops = data.edge_index[0] != data.edge_index[1]
            if data.get('edge_mask') is not None and data.get('ground_truth_mask') is not None:
                self.predict_masks.append(data.edge_mask[self_loops])
                self.target_masks.append(data.ground_truth_mask[self_loops])

    def run_start(self, mode: str = 'train'):
        self.start_time[mode] = time.time()

    def run_finish(self, mode: str = 'train'):
        self.finish_time[mode] = time.time()

    def runtime(self, mode: str):
        assert (self.start_time.get(mode) is not None and
                self.finish_time.get(mode) is not None)
        return self.finish_time[mode] - self.start_time[mode]

    def accuracy_one_element(self, idx: int) -> float:
        try:
            return roc_auc_score(self.target_masks[idx].cpu(), self.predict_masks[idx].cpu())
        except:
            return 0.

    @property
    def fidelity_plus(self):
        return self.results_dict['fid_plus']

    @property
    def fidelity_minus(self):
        return self.results_dict['fid_minus']

    @property
    def fidelity_delta(self):
        return self.results_dict['fid_delta']

    @property
    def fidelity_ori(self):
        return self.results_dict['ori_fid']

    @property
    def fidelity_ori_inv(self):
        return self.results_dict['ori_fid_inv']

    @property
    def fidelity_ori_delta(self):
        return self.results_dict['ori_fid_delta']
    
    @property
    def accuracy(self) -> float:
        r"""
        Accuracy, reflecting the alignment between explanation results and human understanding.
        """
        if self.roc_auc_score: 
            return self.roc_auc_score

        tgt = torch.cat(self.target_masks, dim=-1)
        src = torch.cat(self.predict_masks, dim=-1)
        assert tgt.size(0) == src.size(0)

        self.roc_auc_score = roc_auc_score(tgt.cpu(), src.cpu())
        return self.roc_auc_score

class GNN(torch.nn.Module): 
    def __init__(self):
        super(GNN, self).__init__()
        self.enc = GCN(5, 20, 2, 20)
        self.cls = Linear(20, 2)

    def forward(self, x, edge_index, batch=None): 
        x = self.enc(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.cls(x)

if __name__ == '__main__': 
    # robust fidelity示例代码
    # 手动创建解释图
    # 这里的解释子图只需要给Data类就行，可以是pyg的Data，或者继承Data类的任意类
    # edge_mask: 得到的解释结果，即每条边的权重
    # exp = Explanation(
    #     x=torch.Tensor([[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],  [0., 1., 0., 0., 0.]]),
    #     edge_index=torch.Tensor([[0, 0, 1, 1, 2, 2, 3], [1, 2, 0, 3, 0, 3, 2]]).long(), 
    #     edge_mask=torch.randn(7), 
    #     ground_truth_mask=torch.Tensor([1., 0., 1., 0., 0., 0., 0.]), 
    #     y=torch.tensor([0]).long())
    exp = Data(
        x=torch.Tensor([[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],  [0., 1., 0., 0., 0.]]),
        edge_index=torch.Tensor([[0, 0, 1, 1, 2, 2, 3], [1, 2, 0, 3, 0, 3, 2]]).long(), 
        edge_mask=torch.randn(7), 
        ground_truth_mask=torch.Tensor([1., 0., 1., 0., 0., 0., 0.]), 
        #y=torch.tensor([0]).long()
        )
    
    expls = [exp]
    # 添加模型
    gnn = GNN()
    
    expl_evaluator = ExplanationEvaluator(gnn, alpha=0.4)
    for exp in expls:
        # 计算r-fidelity前固定随机种子  
        #fix_seed(2025)
        fix_seed(2023)
        # 计算r-fidelity 
        sparsity_infos = {}
        try: 
            # 计算需要保留的最大节点个数，按照ground truth进行选取
            sparsity_infos['max_nodes'] = (exp.edge_index[:, exp.ground_truth_mask == 1]
                                        .flatten()
                                        .unique()
                                        .size(0))
        except: 
            # 若无ground truth，则进行手动控制
            # 这里是选择计算前30%边所在子图的节点个数
            topk_indices = torch.topk(exp.edge_mask, int(exp.num_edges * 0.3))[-1]
            sparsity_infos['max_nodes'] = (exp.edge_index[:, topk_indices]
                                        .flatten()
                                        .unique()
                                        .size(0))
        expl_evaluator.collect(exp, 'fid', **sparsity_infos)
    
    # 汇总最终结果，计算平均值和标准差
    expl_evaluator.get_summarized_results() 
    # 输出r-fidelity
    # 第一位为均值，第二位为标准差
    print(f"fid_delta (Mean): {expl_evaluator.fidelity_delta[0]:.6f} | (STD): {expl_evaluator.fidelity_delta[1]:.6f} | "
          f"fid+: {expl_evaluator.fidelity_plus[0]:.6f} | "
          f"fid-: {expl_evaluator.fidelity_minus[0]:.6f} | ")