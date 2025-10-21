#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
sys.path.append('./codes/forgraph/')
from codes.explain.config import args
from sklearn.metrics import roc_auc_score
from codes.explain.models import GCN2 as GCN
from codes.explain.metrics import *
from codes.explain.Explainer import ExplainerGCPGDisenMask
from codes.explain.utils import *
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch 
import torch.optim
import time
from torch_geometric.data import Data
from codes.Configures import model_args
from codes.GNNmodels import GnnNets
import os.path as osp
from torch_geometric.utils import to_networkx

def main(dataset_name, disen_k, train_topk):

    def acc(sub_adj, sub_edge_label):
        mask = explainer.masked_adj.cpu().detach().numpy()
        real = []
        pred = []
        sub_adj = coo_matrix(sub_adj)
        sub_edge_label = sub_edge_label.todense()
        for r,c in list(zip(sub_adj.row, sub_adj.col)):
            d = sub_edge_label[r,c] + sub_edge_label[c,r]
            if d == 0:
                real.append(0)
            else:
                real.append(1)
            pred.append(mask[r][c]+mask[c][r])

        if len(np.unique(real))==1 or len(np.unique(pred))==1:
            return -1, [], []
        return roc_auc_score(real, pred), real, pred


    def test_fixmask(model):
        f = open(save_map + str(iteration) + "/LOG_"+testmodel_filename+"_test_fixmask.txt", "w")
        if args.dataset == "BA_3Motifs_7class":
            #class_arr = [12, 13, 23, 123, 1, 2, 3]
            y = dataset.data.y[test_indices]
            for l in range(max(y)+1):
                test_indices_temp = torch.tensor(test_indices)[torch.where(y==l)[0]]
                if l ==0:
                    motif_index_arr = [1, 2, 12]
                elif l ==1:
                    motif_index_arr = [1, 3, 13]
                elif l ==2:
                    motif_index_arr = [2, 3, 23]
                elif l ==3:
                    motif_index_arr = [1, 2, 3, 12, 13, 23, 123]
                elif l ==4:
                    motif_index_arr = [1]
                elif l ==5:
                    motif_index_arr = [2]
                elif l ==6:
                    motif_index_arr = [3]
                test_fixmask_onecase(f, model, motif_index_arr, test_indices_temp, l)
        elif args.dataset == "Mutagenicity_full":
            y = dataset.data.y[test_indices]
            for l in range(max(y)+1):
                test_indices_temp = torch.tensor(test_indices)[torch.where(y==l)[0]]
                test_fixmask_mutag(f, model, test_indices_temp, l)
        f.close()


    def test_fixmask_mutag(f, model, test_indices, l):
        plotutils = PlotUtils(dataset_name=args.dataset)
        model.eval()
        test_indices_gt_dic = {"test_indices_gt":[], "gt_count_arr":[]}
        for graphid in test_indices:
            sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
            sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
            sub_edge_label = dataset.data.edge_label[dataset.slices['edge_label'][graphid].item() : dataset.slices['edge_label'][graphid+1].item()]
            data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device), edge_label=sub_edge_label)
            sub_edge_gt = dataset.data.edge_label_gt[dataset.slices['edge_label_gt'][graphid].item() : dataset.slices['edge_label_gt'][graphid+1].item()]
            #graph = to_networkx(data, to_undirected=True)
            mol_x = torch.topk(data.x, 1)[1].squeeze().detach().cpu().tolist()
            edge_index = data.edge_index.T.detach().cpu().tolist()
            edge_label = data.edge_label.detach().cpu().tolist()
            mol = plotutils.graph_to_mol(mol_x, edge_index, edge_label)
            s = Chem.MolToSmiles(mol)
            if 'N(=O)O' in s or 'N([H])[H]' in s:
                print("graphid: ", graphid, ", s: ",s)
                gt_count = len(torch.where(sub_edge_gt == 1)[0])/4
                test_indices_gt_dic["test_indices_gt"].append(graphid)
                test_indices_gt_dic["gt_count_arr"].append(gt_count)
                test_indices_gt_dic[graphid] = gt_count

        max_gt_count = int(max(test_indices_gt_dic["gt_count_arr"]))
        for i in range(1, max_gt_count + 1):
            fix_fidelityplus_complete = []
            fix_fidelityminus_complete = []
            topk=[]
            acc_temp = 0
            for graphid in test_indices_gt_dic["test_indices_gt"]:
                logits, prob, _, sub_embs = model(data)
                origin_pred = prob.squeeze()
                label = dataset.data.y[graphid]
                acc_temp = acc_temp + (label == torch.argmax(origin_pred))

                maskimp_edge_mask = torch.ones(sub_edge_index.shape[1], dtype=torch.float32).to(args.device) 
                masknotimp_edge_mask  = torch.ones(sub_edge_index.shape[1], dtype=torch.float32).to(args.device) 
                selected_impedges_idx = torch.where(sub_edge_gt == 1)[0][:4*i]    #select i gt, i is the count of gt(ground-truth)
                other_notimpedges_idx = torch.where(sub_edge_gt == 0)[0]

                topk.append(len(selected_impedges_idx)/len(maskimp_edge_mask))
                maskimp_edge_mask[selected_impedges_idx] = 0.1#重要的边，权重置为1-mask
                #maskimp_edge_mask[other_notimpedges_idx] = 0.9
                explainer.__clear_masks__()
                explainer.__set_masks__(sub_features, sub_edge_index, maskimp_edge_mask)    
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                _, maskimp_preds, _, embed = model(data)
                maskimp_pred = maskimp_preds.squeeze()
                explainer.__clear_masks__()

                #masknotimp_edge_mask[selected_impedges_idx] = 0.9  
                masknotimp_edge_mask[other_notimpedges_idx] = 0.1      #除了重要的top_k%之外的其他边置为mask
                explainer.__clear_masks__()
                explainer.__set_masks__(sub_features, sub_edge_index, masknotimp_edge_mask)    
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                _, masknotimp_preds, _, embed = model(data)
                masknotimp_pred = masknotimp_preds.squeeze()
                explainer.__clear_masks__()

                fidelity_complete_onenode = sum(abs(origin_pred - maskimp_pred)).item()
                fix_fidelityplus_complete.append(fidelity_complete_onenode)

                fidelityminus_complete_onenode = sum(abs(origin_pred - masknotimp_pred)).item()
                fix_fidelityminus_complete.append(fidelityminus_complete_onenode)

            fix_fidelityplus = np.mean(fix_fidelityplus_complete)
            fix_fidelityminus = np.mean(fix_fidelityminus_complete)
            fix_F_fidelity = 2/ (torch.true_divide(1, fix_fidelityplus) + torch.true_divide(1, torch.true_divide(1, fix_fidelityminus)))
            fix_topk =  np.mean(topk)
            f.write("acc={}".format(acc_temp/len(test_indices)) + "\n")
            if l is not None:
                f.write("fix_topk_c{}_gt{}={}".format(l, i, fix_topk) + "\n")
                f.write("fix_fidelityplus_c{}_gt{}={}".format(l, i, fix_fidelityplus) + "\n")
                f.write("fix_fidelityminus_c{}_gt{}={}".format(l, i, fix_fidelityminus)+"\n")
                f.write("fix_F_fidelity_c{}_gt{}={}".format(l, i, fix_F_fidelity)+"\n")
            else:
                f.write("fix_topk_gt{}={}".format(i, fix_topk) + "\n")
                f.write("fix_fidelityplus_gt{}={}".format(i, fix_fidelityplus) + "\n")
                f.write("fix_fidelityminus_gt{}={}".format(i, fix_fidelityminus)+"\n")
                f.write("fix_F_fidelity_gt{}={}".format(i, fix_F_fidelity)+"\n")



    def test_fixmask_onecase(f, model, motif_index_arr, test_indices, l=None):
        model.eval()
        for motif_index  in motif_index_arr:
            fix_fidelityplus_complete = []
            fix_fidelityminus_complete = []
            topk=[]
            acc_temp = 0
            for graphid in test_indices:
                sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
                sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                logits, prob, _, sub_embs = model(data)
                label = dataset.data.y[graphid]
                sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
                
                x = sub_features.to(args.device)
                origin_pred = prob.squeeze()
                acc_temp = acc_temp + (label == torch.argmax(origin_pred))
                maskimp_edge_mask = torch.ones(sub_edge_index.shape[1], dtype=torch.float32).to(args.device) 
                masknotimp_edge_mask  = torch.ones(sub_edge_index.shape[1], dtype=torch.float32).to(args.device) 
                
                sub_edge_label = dataset.data.edge_label[dataset.slices['edge_label'][graphid].item() : dataset.slices['edge_label'][graphid+1].item()]
                basis_idx_0 = torch.where( sub_edge_label == 0)[0]
                motif_idx_1 = torch.where( sub_edge_label == 1)[0]
                motif_idx_2 = torch.where( sub_edge_label == 2)[0]
                motif_idx_3 = torch.where( sub_edge_label == 3)[0]
                motif_idx_4 = torch.where( sub_edge_label == 4)[0]
                motif_idx_5 = torch.where( sub_edge_label == 5)[0]
                motif_idx_6 = torch.where( sub_edge_label == 6)[0]

                if motif_index == 123456:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_2, motif_idx_3, motif_idx_4, motif_idx_5, motif_idx_6), dim=0)
                    other_notimpedges_idx = basis_idx_0
                elif motif_index == 14:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_4), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_2, motif_idx_3, motif_idx_5, motif_idx_6), dim=0)
                elif motif_index == 25:
                    selected_impedges_idx = torch.cat((motif_idx_2, motif_idx_5), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_1, motif_idx_3, motif_idx_4, motif_idx_6), dim=0)
                elif motif_index == 36:
                    selected_impedges_idx = torch.cat((motif_idx_3, motif_idx_6), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_1, motif_idx_2, motif_idx_4, motif_idx_5), dim=0)
                elif motif_index == 2356:
                    selected_impedges_idx = torch.cat((motif_idx_2, motif_idx_3, motif_idx_5, motif_idx_6), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_1, motif_idx_4), dim=0)
                elif motif_index == 1346:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_3, motif_idx_4, motif_idx_6), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_2, motif_idx_5), dim=0)
                elif motif_index == 1245:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_2, motif_idx_4, motif_idx_5), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_3, motif_idx_6), dim=0)
                elif motif_index == 1:
                    selected_impedges_idx = motif_idx_1
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_2, motif_idx_3), dim=0)
                elif motif_index == 2:
                    selected_impedges_idx = motif_idx_2
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_1, motif_idx_3), dim=0)
                elif motif_index == 3:
                    selected_impedges_idx = motif_idx_3
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_1, motif_idx_2), dim=0)
                elif motif_index == 12:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_2), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_3), dim=0)
                elif motif_index == 13:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_3), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_2), dim=0)
                elif motif_index == 23:
                    selected_impedges_idx = torch.cat((motif_idx_2, motif_idx_3), dim=0)
                    other_notimpedges_idx = torch.cat((basis_idx_0, motif_idx_1), dim=0)
                elif motif_index == 123:
                    selected_impedges_idx = torch.cat((motif_idx_1, motif_idx_2, motif_idx_3), dim=0)
                    other_notimpedges_idx = basis_idx_0

                topk.append(len(selected_impedges_idx)/len(maskimp_edge_mask))
                maskimp_edge_mask[selected_impedges_idx] = 0.1#重要的边，权重置为1-mask
                #maskimp_edge_mask[other_notimpedges_idx] = 0.9#重要的边，权重置为1-mask
                explainer.__clear_masks__()
                explainer.__set_masks__(x, sub_edge_index, maskimp_edge_mask)    
                data = Data(x=x, edge_index=sub_edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
                _, maskimp_preds, _, embed = model(data)
                maskimp_pred = maskimp_preds.squeeze()
                explainer.__clear_masks__()

                #masknotimp_edge_mask[selected_impedges_idx] = 0.9  
                masknotimp_edge_mask[other_notimpedges_idx] = 0.1      #除了重要的top_k%之外的其他边置为mask
                explainer.__clear_masks__()
                explainer.__set_masks__(x, sub_edge_index, masknotimp_edge_mask)    
                data = Data(x=x, edge_index=sub_edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
                _, masknotimp_preds, _, embed = model(data)
                masknotimp_pred = masknotimp_preds.squeeze()
                explainer.__clear_masks__()

                fidelity_complete_onenode = sum(abs(origin_pred - maskimp_pred)).item()
                fix_fidelityplus_complete.append(fidelity_complete_onenode)

                fidelityminus_complete_onenode = sum(abs(origin_pred - masknotimp_pred)).item()
                fix_fidelityminus_complete.append(fidelityminus_complete_onenode)

            fix_fidelityplus = np.mean(fix_fidelityplus_complete)
            fix_fidelityminus = np.mean(fix_fidelityminus_complete)
            fix_F_fidelity = 2/ (torch.true_divide(1, fix_fidelityplus) + torch.true_divide(1, torch.true_divide(1, fix_fidelityminus)))
            fix_topk =  np.mean(topk)
            f.write("acc={}".format(acc_temp/len(test_indices)) + "\n")
            if l is not None:
                f.write("fix_topk_c{}_{}={}".format(l, motif_index, fix_topk) + "\n")
                f.write("fix_fidelityplus_c{}_{}={}".format(l, motif_index, fix_fidelityplus) + "\n")
                f.write("fix_fidelityminus_c{}_{}={}".format(l, motif_index, fix_fidelityminus)+"\n")
                f.write("fix_F_fidelity_c{}_{}={}".format(l, motif_index, fix_F_fidelity)+"\n")
            else:
                f.write("fix_topk_{}={}".format(motif_index, fix_topk) + "\n")
                f.write("fix_fidelityplus_{}={}".format(motif_index, fix_fidelityplus) + "\n")
                f.write("fix_fidelityminus_{}={}".format(motif_index, fix_fidelityminus)+"\n")
                f.write("fix_F_fidelity_{}={}".format(motif_index, fix_F_fidelity)+"\n")


    def plot_one(iteration, data, sub_edge_index, edge_mask, graphid, disen_idx):
        plotutils = PlotUtils(dataset_name=args.dataset)
        if disen_idx is None:
            figurename = f"example_{graphid}_{dataset.data.y[graphid]}.pdf"
        else:
            figurename = f"example_{graphid}_{dataset.data.y[graphid]}_disen{disen_idx}.pdf"
        if not os.path.exists (save_map + str(iteration)+"/case"):
            os.makedirs(save_map + str(iteration)+"/case")
        if args.dataset =="Mutagenicity" or args.dataset =="Mutagenicity_full" or args.dataset =="NCI1":
            visual_imp_edge_count = 8
        else:
            if args.dataset == "BA_3Motifs_7class":
                visual_imp_edge_count = 36
            else:
                visual_imp_edge_count = 12
        #plot(sub_adj, label, graphid, iteration)
        #edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]
        edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
        #important_edges = sub_edge_index[:, edges_idx_desc]
        #important_nodes = list(set(important_edges[0].numpy()) | set(important_edges[1].numpy()))
        important_nodelist = []
        important_edgelist = []
        important_edgeidx= []
        for idx in edges_idx_desc:
            if len(important_edgelist) < visual_imp_edge_count:
                if (sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()) not in important_edgelist:
                    important_nodelist.append(sub_edge_index[0][idx].item())
                    important_nodelist.append(sub_edge_index[1][idx].item())
                    important_edgelist.append((sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()))
                    important_edgelist.append((sub_edge_index[1][idx].item(), sub_edge_index[0][idx].item()))
                    important_edgeidx.append(idx.item())
                    important_edgeidx.append(idx.item())
        important_nodelist = list(set(important_nodelist))
        if args.dataset =="Mutagenicity" or args.dataset =="Mutagenicity_full" or args.dataset =="NCI1":
            plotutils.visualize(data, nodelist=important_nodelist, edgelist=important_edgelist, 
                            figname=os.path.join(save_map + str(iteration)+"/case", figurename))
        else:
            ori_graph = to_networkx(data, to_undirected=True)
            if args.dataset == "PROTEINS" or args.dataset == "BA_3Motifs_7class":
                if disen_idx is None:
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x, graphid = graphid,
                            figname=os.path.join(save_map + str(iteration)+"/case", figurename))
                else:
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x, graphid = graphid,
                            figname=os.path.join(save_map + str(iteration)+"/case", figurename), edge_weight = edge_mask[important_edgeidx])
            else:
                if hasattr(dataset, 'supplement'):
                    words = dataset.supplement['sentence_tokens'][str(graphid)]
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, words=words,
                            figname=os.path.join(save_map + str(iteration)+"/case", figurename))
                else:
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x, 
                            figname=os.path.join(save_map + str(iteration)+"/case", figurename))



    def test(iteration, test_indices, model, explainer, topk_arr, plot_flag=False):
        preds = []
        reals = []
        ndcgs =[]
        exp_dict={}
        pred_label_dict={}
        plotutils = PlotUtils(dataset_name=args.dataset)
        metric = MaskoutMetric(model, args)
        allnode_related_preds_dict = dict()
        allnode_mask_dict = dict()
        for graphid in test_indices:
            sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
            sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
            data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
            logits, prob, _, sub_embs = model(data)
            label = dataset.data.y[graphid]
            sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
            #dise_gnn.eval()
            #disen_features = dise_gnn(sub_features.to(args.device), sub_edge_index.to(args.device))
            explainer.eval()
            #masked_pred = explainer((sub_features, disen_features, sub_adj, 1.0, label))
            masked_pred, _, _ = explainer((sub_features, sub_adj, sub_embs, 1.0, label))
            #insert = 20
            #acc(sub_adj, insert)
            if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
                sub_edge_label = dataset.data.edge_label[dataset.slices['edge_label'][graphid].item() : dataset.slices['edge_label'][graphid+1].item()]
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device), edge_label=sub_edge_label)
                sub_edge_gt = dataset.data.edge_label_gt[dataset.slices['edge_label_gt'][graphid].item() : dataset.slices['edge_label_gt'][graphid+1].item()]
                sub_edge_gt_matrix= coo_matrix((sub_edge_gt,(sub_edge_index[0],sub_edge_index[1])),shape=(sub_features.shape[0],sub_features.shape[0]))
                auc_onegraph, real, pred = acc(sub_adj, sub_edge_gt_matrix)
                reals.extend(real)
                preds.extend(pred)
            origin_pred = prob.squeeze()
            ndcg_onegraph, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
            ndcgs.append(ndcg_onegraph)

            mask = explainer.masked_adj
            pred_mask, related_preds_dict = metric.metric_del_edges_GC(topk_arr, sub_features, mask, sub_edge_index, origin_pred, masked_pred, label)
            allnode_related_preds_dict[graphid] = related_preds_dict
            allnode_mask_dict[graphid] = pred_mask
            
            exp_dict[graphid] = mask.detach()
            origin_label = torch.argmax(origin_pred)
            pred_label_dict[graphid]=origin_label

            if plot_flag:
                edge_mask_max = mask[sub_edge_index[0], sub_edge_index[1]]
                plot_one(iteration, data, sub_edge_index, edge_mask_max, graphid, None)
                disen_mask =  explainer.disen_M
                for i in range(disen_mask.shape[0]):
                    edge_mask = disen_mask[i]
                    plot_one(iteration, data, sub_edge_index, edge_mask, graphid, i)
                    
        if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
            if len(reals) != 0:
                auc = roc_auc_score(reals,preds)
            else:
                auc = -1
        else:
            auc = 0
        ndcg = np.mean(ndcgs)

        return auc, ndcg, allnode_related_preds_dict, allnode_mask_dict, exp_dict, pred_label_dict


    def train(iteration, model):
        tik = time.time()
        epochs = args.eepochs
        t0 = args.coff_t0
        t1 = args.coff_te
        best_auc = 0
        explainer.train()

        best_auc = 0
        best_decline = 0
        best_F_fidelity = 0
        optimizer = torch.optim.Adam(explainer.elayers.parameters(), lr=args.elr)
        clip_value_min = -2.0
        clip_value_max = 2.0
        f = open(save_map + str(iteration) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
        for epoch in range(epochs):
            loss = torch.tensor(0.0)
            pll, sll, cll, moll, fll = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
            for graphid in train_instances:
                sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
                sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                logits, origin_pred, _, sub_embs = model(data)
                label = dataset.data.y[graphid]
                origin_pred = origin_pred.squeeze()
                pred_label = torch.argmax(origin_pred)
                sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
                #disen_features = dise_gnn(sub_features.to(args.device), sub_edge_index.to(args.device))
                #pred = explainer((sub_features, disen_features, sub_adj, tmp, label))
                pred, disen_fidelityminus_complete, disen_fidelityplus_complete = explainer((sub_features, sub_adj, sub_embs, tmp, origin_pred))
                l, pl, sl, cl, mol, fl = explainer.loss(pred, pred_label, disen_fidelityminus_complete, disen_fidelityplus_complete)
                loss = loss + l
                pll =  pll +pl
                sll = sll + sl
                cll = cll +cl
                moll = moll +mol
                fll = fll +fl

            if torch.isnan(loss):
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(explainer.parameters(), clip_value_max)
            optimizer.step()
            print("epoch", epoch, "loss", loss.item(), "pll", pll.item(), "sll", sll.item(), "cll", cll.item(), "moll", moll.item(), "fll", fll.item())
            f.write("epoch,{}".format(epoch) + ",loss,{}".format(loss.item()) + ",pll,{}".format(pll.item()) + ",sll,{}".format(sll.item()) + ",cll,{}".format(cll.item()) + ",moll,{}".format(moll.item()) + ",fll,{}".format(fll.item()) + "\n")
            
            del sub_features
            del sub_edge_index
            del sub_adj
            torch.cuda.empty_cache()

            #eval
            auc, ndcg, allnode_related_preds_dict, allnode_mask_dict, _, _ = test(iteration, eval_indices, model, explainer, [5])

            fidelityplus_complete = []
            fidelityminus_complete = []
            for graphid in eval_indices:
                related_preds = allnode_related_preds_dict[graphid][5]
                maskimp_probs = related_preds[0]["maskimp"]
                origin_pred =related_preds[0]["origin"]
                fidelity_complete_onenode = sum(abs(origin_pred - maskimp_probs)).item()
                fidelityplus_complete.append(fidelity_complete_onenode)
                masknotimp_probs = related_preds[0]["masknotimp"]
                fidelityminus_complete_onenode = sum(abs(origin_pred - masknotimp_probs)).item()
                fidelityminus_complete.append(fidelityminus_complete_onenode)

            fidelityplus = np.mean(fidelityplus_complete)
            fidelityminus = np.mean(fidelityminus_complete)
            decline = torch.sub(fidelityplus, fidelityminus)
            F_fidelity = 2/ (torch.true_divide(1, fidelityplus) + torch.true_divide(1, torch.true_divide(1, fidelityminus)))
        
            if epoch == 0:
                best_decline = decline
                best_F_fidelity = F_fidelity
            if auc >= best_auc:
                print("saving best auc model...")
                f.write("saving best auc model...\n")
                best_auc = auc
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt')

            if F_fidelity >= best_F_fidelity:
                print("saving best F_fidelity model...")
                f.write("saving best F_fidelity model...\n")
                best_F_fidelity = F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_F.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_F.pt')
            
            if decline >= best_decline:
                print("saving best decline model...")
                f.write("saving best decline model...\n")
                best_decline = decline
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_decline.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_decline.pt')
            print("epoch", epoch, "loss", loss, "auc", auc, "ndcg", ndcg, "best_F_fidelity",best_F_fidelity, "2: ", fidelityplus, "1: ", fidelityminus,"decline",decline, "F_fidelity",F_fidelity)
            f.write("epoch,{}".format(epoch) + ",loss,{}".format(loss)  + ",auc,{}".format(auc) + ",ndcg,{}".format(ndcg)+ ",2,{}".format(fidelityplus) + ",1,{}".format(fidelityminus)  + ",decline,{}".format(decline)  +",F_fidelity,{}".format(F_fidelity)+ "\n")
            
            #del fidelityplus_complete
            #del fidelityminus_complete
            #del mask
            #del x_collector
            del allnode_related_preds_dict
            del allnode_mask_dict
            torch.cuda.empty_cache()

        tok = time.time()
        f.write("train time,{}".format(tok - tik) + "\n")
        f.close()






    #args.elr = 0.003
    args.elr = 0.01
    args.coff_t0=5.0
    args.coff_te=2.0
    args.coff_size = 0.05
    args.lamda = 10.0
    #args.coff_ent = 1.0
    #args.concat = True
    #args.bn = True
    args.graph_classification = True
    args.batch_size = 128
    args.random_split_flag = True
    args.data_split_ratio =  [0.8, 0.1, 0.1]  #None
    args.seed = 2023
    #args.eepochs = 30
    args.eepochs = 100
    args.dataset_root = "/mnt/8T/torch_projects/datasets"
    args.dataset = dataset_name    #BA_3Motifs_7class, Mutagenicity, Mutagenicity_full, NCI1, PROTEINS
    
    #Disen parameters
    args.dropout = 0.2
    args.num_layers = 3        #5
    args.init_k = disen_k       #8
    args.delta_k = 0
    args.disen_hid_dim = 32         #64
    args.routit = 6         #7
    args.tau = 1
    #explainer parameters
    args.e_in_channels = 1
    args.exp_hid_dim = 128
    args.train_topk = train_topk
    
    args.topk_arr = list(range(10))+list(range(10,101,5))
    #BA_3Motifs_7class, NCI1, Mutagenicity_full, PROTEINS
    save_map = f"LOGS/{args.dataset.upper()}_seed2023_size{args.coff_size}_elr{args.elr}_epoch{args.eepochs}_DisenMask_routit{args.routit}_k{args.init_k}_orthogonal_3MLP_5loss_topk{args.train_topk}/"
    
    if not os.path.exists(save_map):
        os.makedirs(save_map)
    #train
    args.model_filename = args.dataset
    #test
    test_flag = False
    testmodel_filename = args.dataset + '_BEST_F'
    args.plot_flag = False

    args.fix_exp =12
    args.mask_thresh = 0.5

    if test_flag:
        log_filename = save_map + "log_test.txt"
    else:
        log_filename = save_map + "log.txt"
    f_mean = open(log_filename, "w")
    auc_all = []
    ndcg_all = []
    PN_all = []
    PS_all = []
    FNS_all = []
    size_all = []
    acc_all = []
    pre_all = []
    rec_all = []
    f1_all = []
    simula_arr = []
    simula_origin_arr = []
    simula_complete_arr = []
    fidelity_arr = []
    fidelity_origin_arr = []
    fidelity_complete_arr = []
    fidelityminus_arr = []
    fidelityminus_origin_arr = []
    fidelityminus_complete_arr = []
    finalfidelity_complete_arr = []
    fvaluefidelity_complete_arr = []
    del_fidelity_arr = []
    del_fidelity_origin_arr = []
    del_fidelity_complete_arr = []
    del_fidelityminus_arr = []
    del_fidelityminus_origin_arr = []
    del_fidelityminus_complete_arr = []
    del_finalfidelity_complete_arr = []
    del_fvaluefidelity_complete_arr = []
    sparsity_edges_arr = []
    fidelity_nodes_arr = []
    fidelity_origin_nodes_arr = []
    fidelity_complete_nodes_arr = []
    fidelityminus_nodes_arr = []
    fidelityminus_origin_nodes_arr = []
    fidelityminus_complete_nodes_arr = []
    finalfidelity_complete_nodes_arr = []
    sparsity_nodes_arr = []
    for iteration in range(1):
        print("Starting iteration: {}".format(iteration))
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #args.device = "cpu"
        if not os.path.exists(save_map+str(iteration)):
            os.makedirs(save_map+str(iteration))

        dataset = get_dataset(args.dataset_root, args.dataset)
        dataset.data.x = dataset.data.x.float()
        dataset.data.y = dataset.data.y.squeeze().long()
        #if args.graph_classification:
        dataloader_params = {'batch_size': args.batch_size,
                                'random_split_flag': args.random_split_flag,
                                'data_split_ratio': args.data_split_ratio,
                                'seed': args.seed} 
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices
        train_instances = loader['train'].dataset.indices
        eval_indices = loader['eval'].dataset.indices
        if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
            test_indices = [i for i in test_indices if dataset.data.y[i]==0]
            train_instances = [i for i in train_instances if dataset.data.y[i]==0]
            eval_indices = [i for i in eval_indices if dataset.data.y[i]==0]
        
        GNNmodel_ckpt_path = osp.join('checkpoint', args.dataset+'_'+str(iteration), 'gcn_best.pth') 
        model = GnnNets(input_dim=dataset.num_node_features,  output_dim=dataset.num_classes, model_args=model_args)
        ckpt = torch.load(GNNmodel_ckpt_path)
        model.load_state_dict(ckpt['net'])
        model.to(args.device)
        model.eval()
        
        #Disen parameters
        args.inp_dim = dataset.num_node_features

        explainer = ExplainerGCPGDisenMask(model=model, args=args)
        explainer.to(args.device)
        
        # Training
        if test_flag:
            f = open(save_map + str(iteration) + "/LOG_"+testmodel_filename+"_test.txt", "w")
            explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + testmodel_filename+".pt") )
            #explainer.load_state_dict(torch.load(save_map + str(iteration) + "/BA_6Motifs_2class_BEST_F.pt") )
        else:
            f = open(save_map + str(iteration) + "/" + "LOG.txt", "w")
            train(iteration, model)
            explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + args.model_filename +'_BEST_F.pt'))

        tik = time.time()
        test_fixmask(model)
        auc, ndcg, allnode_related_preds_dict, allnode_mask_dict, exp_dict, pred_label_dict = test(iteration, test_indices, model, explainer, args.topk_arr, plot_flag=args.plot_flag)
        auc_all.append(auc)
        ndcg_all.append(ndcg)

        PN = compute_pn(exp_dict, pred_label_dict, args, model, dataset)
        PS, ave_size = compute_ps(exp_dict, pred_label_dict, args, model, dataset)
        if PN + PS==0:
            FNS=0
        else:
            FNS = 2 * PN * PS / (PN + PS)
        acc_1, pre, rec, f1=0,0,0,0
        if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
            acc_1, pre, rec, f1 = compute_precision_recall(exp_dict, args, dataset)
        PN_all.append(PN)
        PS_all.append(PS)
        FNS_all.append(FNS)
        size_all.append(ave_size)
        acc_all.append(acc_1)
        pre_all.append(pre)
        rec_all.append(rec)
        f1_all.append(f1)

        one_simula_arr = []
        one_simula_origin_arr = []
        one_simula_complete_arr = []
        one_fidelity_arr = []
        one_fidelity_origin_arr = []
        one_fidelity_complete_arr = []
        one_fidelityminus_arr = []
        one_fidelityminus_origin_arr = []
        one_fidelityminus_complete_arr = []
        one_finalfidelity_complete_arr = []
        one_fvaluefidelity_complete_arr = []
        one_del_fidelity_arr = []
        one_del_fidelity_origin_arr = []
        one_del_fidelity_complete_arr = []
        one_del_fidelityminus_arr = []
        one_del_fidelityminus_origin_arr = []
        one_del_fidelityminus_complete_arr = []
        one_del_finalfidelity_complete_arr = []
        one_del_fvaluefidelity_complete_arr = []
        one_sparsity_edges_arr = []
        one_fidelity_nodes_arr = []
        one_fidelity_origin_nodes_arr = []
        one_fidelity_complete_nodes_arr = []
        one_fidelityminus_nodes_arr = []
        one_fidelityminus_origin_nodes_arr = []
        one_fidelityminus_complete_nodes_arr = []
        one_finalfidelity_complete_nodes_arr = []
        one_sparsity_nodes_arr = []
        for top_k in args.topk_arr:
            print("top_k: ", top_k)
            x_collector = XCollector()
            for graphid in test_indices:
                related_preds = allnode_related_preds_dict[graphid][top_k]
                mask = allnode_mask_dict[graphid]
                x_collector.collect_data(mask, related_preds, label=0)
                f.write("graphid,{}\n".format(graphid))
                f.write("mask,{}\n".format(mask))
                f.write("related_preds,{}\n".format(related_preds))

            one_simula_arr.append(round(x_collector.simula, 4))
            one_simula_origin_arr.append(round(x_collector.simula_origin, 4))
            one_simula_complete_arr.append(round(x_collector.simula_complete, 4))
            one_fidelity_arr.append(round(x_collector.fidelity, 4))
            one_fidelity_origin_arr.append(round(x_collector.fidelity_origin, 4))
            one_fidelity_complete_arr.append(round(x_collector.fidelity_complete, 4))
            one_fidelityminus_arr.append(round(x_collector.fidelityminus, 4))
            one_fidelityminus_origin_arr.append(round(x_collector.fidelityminus_origin, 4))
            one_fidelityminus_complete_arr.append(round(x_collector.fidelityminus_complete, 4))
            one_finalfidelity_complete_arr.append(round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
            F_fidelity = 2/(1/x_collector.fidelity_complete +1/(1/x_collector.fidelityminus_complete))
            one_fvaluefidelity_complete_arr.append(round(F_fidelity, 4))
            one_del_fidelity_arr.append(round(x_collector.del_fidelity, 4))
            one_del_fidelity_origin_arr.append(round(x_collector.del_fidelity_origin, 4))
            one_del_fidelity_complete_arr.append(round(x_collector.del_fidelity_complete, 4))
            one_del_fidelityminus_arr.append(round(x_collector.del_fidelityminus, 4))
            one_del_fidelityminus_origin_arr.append(round(x_collector.del_fidelityminus_origin, 4))
            one_del_fidelityminus_complete_arr.append(round(x_collector.del_fidelityminus_complete, 4))
            one_del_finalfidelity_complete_arr.append(round(x_collector.del_fidelity_complete - x_collector.del_fidelityminus_complete, 4))
            del_F_fidelity = 2/(1/x_collector.del_fidelity_complete +1/(1/x_collector.del_fidelityminus_complete))
            one_del_fvaluefidelity_complete_arr.append(round(del_F_fidelity, 4))
            one_sparsity_edges_arr.append(round(x_collector.sparsity_edges, 4))
            one_fidelity_nodes_arr.append(round(x_collector.fidelity_nodes, 4))
            one_fidelity_origin_nodes_arr.append(round(x_collector.fidelity_origin_nodes, 4))
            one_fidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes, 4))
            one_fidelityminus_nodes_arr.append(round(x_collector.fidelityminus_nodes, 4))
            one_fidelityminus_origin_nodes_arr.append(round(x_collector.fidelityminus_origin_nodes, 4))
            one_fidelityminus_complete_nodes_arr.append(round(x_collector.fidelityminus_complete_nodes, 4))
            one_finalfidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes - x_collector.fidelityminus_complete_nodes, 4))
            one_sparsity_nodes_arr.append(round(x_collector.sparsity_nodes, 4))

        print("one_simula_arr =", one_simula_arr)
        print("one_simula_origin_arr =", one_simula_origin_arr)
        print("one_simula_complete_arr =", one_simula_complete_arr)
        print("one_fidelity_arr =", one_fidelity_arr)
        print("one_fidelity_origin_arr =", one_fidelity_origin_arr)
        print("one_fidelity_complete_arr =", one_fidelity_complete_arr)
        print("one_fidelityminus_arr=", one_fidelityminus_arr)
        print("one_fidelityminus_origin_arr=", one_fidelityminus_origin_arr)
        print("one_fidelityminus_complete_arr=", one_fidelityminus_complete_arr)
        print("one_finalfidelity_complete_arr=", one_finalfidelity_complete_arr)
        print("one_fvaluefidelity_complete_arr=", one_fvaluefidelity_complete_arr)
        print("one_del_fidelity_arr =", one_del_fidelity_arr)
        print("one_del_fidelity_origin_arr =", one_del_fidelity_origin_arr)
        print("one_del_fidelity_complete_arr =", one_del_fidelity_complete_arr)
        print("one_del_fidelityminus_arr=", one_del_fidelityminus_arr)
        print("one_del_fidelityminus_origin_arr=", one_del_fidelityminus_origin_arr)
        print("one_del_fidelityminus_complete_arr=", one_del_fidelityminus_complete_arr)
        print("one_del_finalfidelity_complete_arr=", one_del_finalfidelity_complete_arr)
        print("one_del_fvaluefidelity_complete_arr=", one_del_fvaluefidelity_complete_arr)
        print("one_sparsity_edges_arr =", one_sparsity_edges_arr)
        print("one_fidelity_nodes_arr =", one_fidelity_nodes_arr)
        print("one_fidelity_origin_nodes_arr =", one_fidelity_origin_nodes_arr)
        print("one_fidelity_complete_nodes_arr =", one_fidelity_complete_nodes_arr)
        print("one_fidelityminus_nodes_arr=", one_fidelityminus_nodes_arr)
        print("one_fidelityminus_origin_nodes_arr=", one_fidelityminus_origin_nodes_arr)
        print("one_fidelityminus_complete_nodes_arr=", one_fidelityminus_complete_nodes_arr)
        print("one_finalfidelity_complete_nodes_arr=", one_finalfidelity_complete_nodes_arr)
        print("one_sparsity_nodes_arr =", one_sparsity_nodes_arr)

        tok = time.time()
        f.write("one_auc={}".format(auc) + "\n")
        f.write("one_ndcg={}".format(ndcg) + "\n")
        f.write("PN: {}, PS: {}, FNS: {}, size:{}\n".format(PN, PS, FNS, ave_size))
        f.write("acc:{}, pre:{}, rec:{}, f1:{}\n".format(acc_1, pre, rec, f1))
        f.write("one_simula={}".format(one_simula_arr) + "\n")
        f.write("one_simula_orign={}".format(one_simula_origin_arr) + "\n")
        f.write("one_simula_complete={}".format(one_simula_complete_arr) + "\n")
        f.write("one_fidelity={}".format(one_fidelity_arr) + "\n")
        f.write("one_fidelity_orign={}".format(one_fidelity_origin_arr) + "\n")
        f.write("one_fidelity_complete={}".format(one_fidelity_complete_arr) + "\n")
        f.write("one_fidelityminus={}".format(one_fidelityminus_arr)+"\n")
        f.write("one_fidelityminus_origin={}".format(one_fidelityminus_origin_arr)+"\n")
        f.write("one_fidelityminus_complete={}".format(one_fidelityminus_complete_arr)+"\n")
        f.write("one_finalfidelity_complete={}".format(one_finalfidelity_complete_arr)+"\n")
        f.write("one_fvaluefidelity_complete={}".format(one_fvaluefidelity_complete_arr)+"\n")
        f.write("one_del_fidelity={}".format(one_del_fidelity_arr) + "\n")
        f.write("one_del_fidelity_orign={}".format(one_del_fidelity_origin_arr) + "\n")
        f.write("one_del_fidelity_complete={}".format(one_del_fidelity_complete_arr) + "\n")
        f.write("one_del_fidelityminus={}".format(one_del_fidelityminus_arr)+"\n")
        f.write("one_del_fidelityminus_origin={}".format(one_del_fidelityminus_origin_arr)+"\n")
        f.write("one_del_fidelityminus_complete={}".format(one_del_fidelityminus_complete_arr)+"\n")
        f.write("one_del_finalfidelity_complete={}".format(one_del_finalfidelity_complete_arr)+"\n")
        f.write("one_del_fvaluefidelity_complete={}".format(one_del_fvaluefidelity_complete_arr)+"\n")
        f.write("one_sparsity_edges={}".format(one_sparsity_edges_arr) + "\n")
        f.write("one_fidelity_nodes={}".format(one_fidelity_nodes_arr) + "\n")
        f.write("one_fidelity_origin_nodes={}".format(one_fidelity_origin_nodes_arr) + "\n")
        f.write("one_fidelity_complete_nodes={}".format(one_fidelity_complete_nodes_arr) + "\n")
        f.write("one_fidelityminus_nodes={}".format(one_fidelityminus_nodes_arr)+"\n")
        f.write("one_fidelityminus_origin_nodes={}".format(one_fidelityminus_origin_nodes_arr)+"\n")
        f.write("one_fidelityminus_complete_nodes={}".format(one_fidelityminus_complete_nodes_arr)+"\n")
        f.write("one_finalfidelity_complete_nodes={}".format(one_finalfidelity_complete_nodes_arr)+"\n")
        f.write("one_sparsity_nodes={}".format(one_sparsity_nodes_arr) + "\n")
        f.write("test time,{}".format(tok-tik))
        f.close()

        simula_arr.append(one_simula_arr)
        simula_origin_arr.append(one_simula_origin_arr)
        simula_complete_arr.append(one_simula_complete_arr)
        fidelity_arr.append(one_fidelity_arr)
        fidelity_origin_arr.append(one_fidelity_origin_arr)
        fidelity_complete_arr.append(one_fidelity_complete_arr)
        fidelityminus_arr.append(one_fidelityminus_arr)
        fidelityminus_origin_arr.append(one_fidelityminus_origin_arr)
        fidelityminus_complete_arr.append(one_fidelityminus_complete_arr)
        finalfidelity_complete_arr.append(one_finalfidelity_complete_arr)
        fvaluefidelity_complete_arr.append(one_fvaluefidelity_complete_arr)
        del_fidelity_arr.append(one_del_fidelity_arr)
        del_fidelity_origin_arr.append(one_del_fidelity_origin_arr)
        del_fidelity_complete_arr.append(one_del_fidelity_complete_arr)
        del_fidelityminus_arr.append(one_del_fidelityminus_arr)
        del_fidelityminus_origin_arr.append(one_del_fidelityminus_origin_arr)
        del_fidelityminus_complete_arr.append(one_del_fidelityminus_complete_arr)
        del_finalfidelity_complete_arr.append(one_del_finalfidelity_complete_arr)
        del_fvaluefidelity_complete_arr.append(one_del_fvaluefidelity_complete_arr)
        sparsity_edges_arr.append(one_sparsity_edges_arr)
        fidelity_nodes_arr.append(one_fidelity_nodes_arr)
        fidelity_origin_nodes_arr.append(one_fidelity_origin_nodes_arr)
        fidelity_complete_nodes_arr.append(one_fidelity_complete_nodes_arr)
        fidelityminus_nodes_arr.append(one_fidelityminus_nodes_arr)
        fidelityminus_origin_nodes_arr.append(one_fidelityminus_origin_nodes_arr)
        fidelityminus_complete_nodes_arr.append(one_fidelityminus_complete_nodes_arr)
        finalfidelity_complete_nodes_arr.append(one_finalfidelity_complete_nodes_arr)
        sparsity_nodes_arr.append(one_sparsity_nodes_arr)

    print("args.dataset", args.dataset)
    print("Disen_auc_all = ", auc_all)
    print("Disen_ndcg_all = ", ndcg_all)
    print("Disen_simula_arr =", simula_arr)
    print("Disen_simula_origin_arr =", simula_origin_arr)
    print("Disen_simula_complete_arr =", simula_complete_arr)
    print("Disen_fidelity_arr =", fidelity_arr)
    print("Disen_fidelity_origin_arr =", fidelity_origin_arr)
    print("Disen_fidelity_complete_arr =", fidelity_complete_arr)
    print("Disen_fidelityminus_arr=", fidelityminus_arr)
    print("Disen_fidelityminus_origin_arr=", fidelityminus_origin_arr)
    print("Disen_fidelityminus_complete_arr=", fidelityminus_complete_arr)
    print("Disen_finalfidelity_complete_arr", finalfidelity_complete_arr)
    print("Disen_fvaluefidelity_complete_arr", fvaluefidelity_complete_arr)
    print("Disen_del_fidelity_arr =", del_fidelity_arr)
    print("Disen_del_fidelity_origin_arr =", del_fidelity_origin_arr)
    print("Disen_del_fidelity_complete_arr =", del_fidelity_complete_arr)
    print("Disen_del_fidelityminus_arr=", del_fidelityminus_arr)
    print("Disen_del_fidelityminus_origin_arr=", del_fidelityminus_origin_arr)
    print("Disen_del_fidelityminus_complete_arr=", del_fidelityminus_complete_arr)
    print("Disen_del_finalfidelity_complete_arr", del_finalfidelity_complete_arr)
    print("Disen_del_fvaluefidelity_complete_arr", del_fvaluefidelity_complete_arr)
    print("Disen_sparsity_edges_arr =", sparsity_edges_arr)
    print("Disen_fidelity_nodes_arr =", fidelity_nodes_arr)
    print("Disen_fidelity_origin_nodes_arr =", fidelity_origin_nodes_arr)
    print("Disen_fidelity_complete_nodes_arr =", fidelity_complete_nodes_arr)
    print("Disen_fidelityminus_nodes_arr=", fidelityminus_nodes_arr)
    print("Disen_fidelityminus_origin_nodes_arr=", fidelityminus_origin_nodes_arr)
    print("Disen_fidelityminus_complete_nodes_arr=", fidelityminus_complete_nodes_arr)
    print("Disen_finalfidelity_complete_nodes_arr", finalfidelity_complete_nodes_arr)
    print("Disen_sparsity_nodes_arr =", sparsity_nodes_arr)

    f_mean.write("Disen_auc_all={}".format(auc_all) + "\n")
    f_mean.write("Disen_ndcg_all={}".format(ndcg_all) + "\n")
    f_mean.write("Disen_PN_all={}".format(PN_all) + "\n")
    f_mean.write("Disen_PS_all={}".format(PS_all) + "\n")
    f_mean.write("Disen_FNS_all={}".format(FNS_all) + "\n")
    f_mean.write("Disen_size_all={}".format(size_all) + "\n")
    f_mean.write("Disen_acc_all={}".format(acc_all) + "\n")
    f_mean.write("Disen_pre_all={}".format(pre_all) + "\n")
    f_mean.write("Disen_rec_all={}".format(rec_all) + "\n")
    f_mean.write("Disen_f1_all={}".format(f1_all) + "\n")
    f_mean.write("Disen_simula_arr={}".format(simula_arr) + "\n")
    f_mean.write("Disen_simula_origin_arr={}".format(simula_origin_arr) + "\n")
    f_mean.write("Disen_simula_complete_arr={}".format(simula_complete_arr) + "\n")
    f_mean.write("Disen_fidelity_arr={}".format(fidelity_arr) + "\n")
    f_mean.write("Disen_fidelity_origin_arr={}".format(fidelity_origin_arr) + "\n")
    f_mean.write("Disen_fidelity_complete_arr={}".format(fidelity_complete_arr) + "\n")
    f_mean.write("Disen_fidelityminus_arr = {}".format(fidelityminus_arr)+"\n")
    f_mean.write("Disen_fidelityminus_origin_arr = {}".format(fidelityminus_origin_arr)+"\n")
    f_mean.write("Disen_fidelityminus_complete_arr = {}".format(fidelityminus_complete_arr)+"\n")
    f_mean.write("Disen_finalfidelity_complete_arr = {}".format(finalfidelity_complete_arr)+"\n")
    f_mean.write("Disen_fvaluefidelity_complete_arr = {}".format(fvaluefidelity_complete_arr)+"\n")
    f_mean.write("Disen_del_fidelity_arr={}".format(del_fidelity_arr) + "\n")
    f_mean.write("Disen_del_fidelity_origin_arr={}".format(del_fidelity_origin_arr) + "\n")
    f_mean.write("Disen_del_fidelity_complete_arr={}".format(del_fidelity_complete_arr) + "\n")
    f_mean.write("Disen_del_fidelityminus_arr = {}".format(del_fidelityminus_arr)+"\n")
    f_mean.write("Disen_del_fidelityminus_origin_arr = {}".format(del_fidelityminus_origin_arr)+"\n")
    f_mean.write("Disen_del_fidelityminus_complete_arr = {}".format(del_fidelityminus_complete_arr)+"\n")
    f_mean.write("Disen_del_finalfidelity_complete_arr = {}".format(del_finalfidelity_complete_arr)+"\n")
    f_mean.write("Disen_del_fvaluefidelity_complete_arr = {}".format(del_fvaluefidelity_complete_arr)+"\n")
    f_mean.write("Disen_sparsity_edges_arr={}".format(sparsity_edges_arr) + "\n")
    f_mean.write("Disen_fidelity_nodes_arr={}".format(fidelity_nodes_arr) + "\n")
    f_mean.write("Disen_fidelity_origin_nodes_arr={}".format(fidelity_origin_nodes_arr) + "\n")
    f_mean.write("Disen_fidelity_complete_nodes_arr={}".format(fidelity_complete_nodes_arr) + "\n")
    f_mean.write("Disen_fidelityminus_nodes_arr = {}".format(fidelityminus_nodes_arr)+"\n")
    f_mean.write("Disen_fidelityminus_origin_nodes_arr = {}".format(fidelityminus_origin_nodes_arr)+"\n")
    f_mean.write("Disen_fidelityminus_complete_nodes_arr = {}".format(fidelityminus_complete_nodes_arr)+"\n")
    f_mean.write("Disen_finalfidelity_complete_nodes_arr = {}".format(finalfidelity_complete_nodes_arr)+"\n")
    f_mean.write("Disen_sparsity_nodes_arr={}".format(sparsity_nodes_arr) + "\n")

    simula_mean = np.average(np.array(simula_arr), axis=0)
    simula_origin_mean = np.average(np.array(simula_origin_arr), axis=0)
    simula_complete_mean = np.average(np.array(simula_complete_arr),axis=0)
    fidelity_mean = np.average(np.array(fidelity_arr),axis=0)
    fidelity_origin_mean = np.average(np.array(fidelity_origin_arr),axis=0)
    fidelity_complete_mean = np.average(np.array(fidelity_complete_arr),axis=0)
    fidelityminus_mean = np.average(np.array(fidelityminus_arr),axis=0)
    fidelityminus_origin_mean = np.average(np.array(fidelityminus_origin_arr),axis=0)
    fidelityminus_complete_mean = np.average(np.array(fidelityminus_complete_arr),axis=0)
    finalfidelity_complete_mean = np.average(np.array(finalfidelity_complete_arr), axis=0)
    fvaluefidelity_complete_mean = np.average(np.array(fvaluefidelity_complete_arr), axis=0)
    del_fidelity_mean = np.average(np.array(del_fidelity_arr),axis=0)
    del_fidelity_origin_mean = np.average(np.array(del_fidelity_origin_arr),axis=0)
    del_fidelity_complete_mean = np.average(np.array(del_fidelity_complete_arr),axis=0)
    del_fidelityminus_mean = np.average(np.array(del_fidelityminus_arr),axis=0)
    del_fidelityminus_origin_mean = np.average(np.array(del_fidelityminus_origin_arr),axis=0)
    del_fidelityminus_complete_mean = np.average(np.array(del_fidelityminus_complete_arr),axis=0)
    del_finalfidelity_complete_mean = np.average(np.array(del_finalfidelity_complete_arr), axis=0)
    del_fvaluefidelity_complete_mean = np.average(np.array(del_fvaluefidelity_complete_arr), axis=0)
    sparsity_edges_mean = np.average(np.array(sparsity_edges_arr),axis=0)
    fidelity_nodes_mean = np.average(np.array(fidelity_nodes_arr),axis=0)
    fidelity_origin_nodes_mean = np.average(np.array(fidelity_origin_nodes_arr),axis=0)
    fidelity_complete_nodes_mean = np.average(np.array(fidelity_complete_nodes_arr),axis=0)
    fidelityminus_nodes_mean = np.average(np.array(fidelityminus_nodes_arr),axis=0)
    fidelityminus_origin_nodes_mean = np.average(np.array(fidelityminus_origin_nodes_arr),axis=0)
    fidelityminus_complete_nodes_mean = np.average(np.array(fidelityminus_complete_nodes_arr),axis=0)
    finalfidelity_complete_nodes_mean = np.average(np.array(finalfidelity_complete_nodes_arr), axis=0)
    sparsity_nodes_mean = np.average(np.array(sparsity_nodes_arr),axis=0)

    print("Disen_auc_mean =", np.mean(auc_all))
    print("Disen_ndcg_mean =", np.mean(ndcg_all))
    print("Disen_simula_mean =", list(simula_mean))
    print("Disen_simula_origin_mean =", list(simula_origin_mean))
    print("Disen_simula_complete_mean =", list(simula_complete_mean))
    print("Disen_fidelity_mean = ", list(fidelity_mean))
    print("Disen_fidelity_origin_mean =", list(fidelity_origin_mean))
    print("Disen_fidelity_complete_mean =", list(fidelity_complete_mean))
    print("Disen_fidelityminus_mean =", list(fidelityminus_mean))
    print("Disen_fidelityminus_origin_mean =", list(fidelityminus_origin_mean))
    print("Disen_fidelityminus_complete_mean =", list(fidelityminus_complete_mean))
    print("Disen_finalfidelity_complete_mean =", list(finalfidelity_complete_mean))
    print("Disen_fvaluefidelity_complete_mean = ", list(fvaluefidelity_complete_mean))
    print("Disen_del_fidelity_mean = ", list(del_fidelity_mean))
    print("Disen_del_fidelity_origin_mean = ", list(del_fidelity_origin_mean))
    print("Disen_del_fidelity_complete_mean = ", list(del_fidelity_complete_mean))
    print("Disen_del_fidelityminus_mean = ", list(del_fidelityminus_mean))
    print("Disen_del_fidelityminus_origin_mean = ", list(del_fidelityminus_origin_mean))
    print("Disen_del_fidelityminus_complete_mean = ", list(del_fidelityminus_complete_mean))
    print("Disen_del_finalfidelity_complete_mean = ", list(del_finalfidelity_complete_mean))
    print("Disen_del_fvaluefidelity_complete_mean = ", list(del_fvaluefidelity_complete_mean))
    print("Disen_sparsity_edges_mean =", list(sparsity_edges_mean))
    print("Disen_fidelity_nodes_mean =", list(fidelity_nodes_mean))
    print("Disen_fidelity_origin_nodes_mean =", list(fidelity_origin_nodes_mean))
    print("Disen_fidelity_complete_nodes_mean =", list(fidelity_complete_nodes_mean))
    print("Disen_fidelityminus_nodes_mean =", list(fidelityminus_nodes_mean))
    print("Disen_fidelityminus_origin_nodes_mean =", list(fidelityminus_origin_nodes_mean))
    print("Disen_fidelityminus_complete_nodes_mean =", list(fidelityminus_complete_nodes_mean))
    print("Disen_finalfidelity_complete_nodes_mean =", list(finalfidelity_complete_nodes_mean))
    print("Disen_sparsity_nodes_mean =", list(sparsity_nodes_mean))

    f_mean.write("Disen_auc_mean = {}".format(np.mean(auc_all))+ "\n")
    f_mean.write("Disen_ndcg_mean = {}".format(np.mean(ndcg_all))+ "\n")
    f_mean.write("Disen_PN_mean = {}".format(np.mean(PN_all))+ "\n")
    f_mean.write("Disen_PS_mean = {}".format(np.mean(PS_all))+ "\n")
    f_mean.write("Disen_FNS_mean = {}".format(np.mean(FNS_all))+ "\n")
    f_mean.write("Disen_size_mean = {}".format(np.mean(size_all))+ "\n")
    f_mean.write("Disen_acc_mean = {}".format(np.mean(acc_all))+ "\n")
    f_mean.write("Disen_pre_mean = {}".format(np.mean(pre_all))+ "\n")
    f_mean.write("Disen_rec_mean = {}".format(np.mean(rec_all))+ "\n")
    f_mean.write("Disen_f1_mean = {}".format(np.mean(f1_all))+ "\n")
    f_mean.write("Disen_simula_mean = {}".format(list(simula_mean))+ "\n")
    f_mean.write("Disen_simula_origin_mean = {}".format(list(simula_origin_mean))+ "\n")
    f_mean.write("Disen_simula_complete_mean = {}".format(list(simula_complete_mean))+ "\n")
    f_mean.write("Disen_fidelity_mean = {}".format(list(fidelity_mean))+ "\n")
    f_mean.write("Disen_fidelity_origin_mean = {}".format(list(fidelity_origin_mean))+ "\n")
    f_mean.write("Disen_fidelity_complete_mean = {}".format(list(fidelity_complete_mean))+ "\n")
    f_mean.write("Disen_fidelityminus_mean = {}".format(list(fidelityminus_mean))+"\n")
    f_mean.write("Disen_fidelityminus_origin_mean = {}".format(list(fidelityminus_origin_mean))+"\n")
    f_mean.write("Disen_fidelityminus_complete_mean = {}".format(list(fidelityminus_complete_mean))+"\n")
    f_mean.write("Disen_finalfidelity_complete_mean = {}".format(list(finalfidelity_complete_mean))+"\n")
    f_mean.write("Disen_fvaluefidelity_complete_mean = {}".format(list(fvaluefidelity_complete_mean)) + "\n")
    f_mean.write("Disen_del_fidelity_mean = {}".format(list(del_fidelity_mean))+ "\n")
    f_mean.write("Disen_del_fidelity_origin_mean = {}".format(list(del_fidelity_origin_mean))+ "\n")
    f_mean.write("Disen_del_fidelity_complete_mean = {}".format(list(del_fidelity_complete_mean))+ "\n")
    f_mean.write("Disen_del_fidelityminus_mean = {}".format(list(del_fidelityminus_mean))+"\n")
    f_mean.write("Disen_del_fidelityminus_origin_mean = {}".format(list(del_fidelityminus_origin_mean))+"\n")
    f_mean.write("Disen_del_fidelityminus_complete_mean = {}".format(list(del_fidelityminus_complete_mean))+"\n")
    f_mean.write("Disen_del_finalfidelity_complete_mean = {}".format(list(del_finalfidelity_complete_mean))+"\n")
    f_mean.write("Disen_del_fvaluefidelity_complete_mean = {}".format(list(del_fvaluefidelity_complete_mean)) + "\n")
    f_mean.write("Disen_sparsity_edges_mean = {}".format(list(sparsity_edges_mean))+ "\n")
    f_mean.write("Disen_fidelity_nodes_mean = {}".format(list(fidelity_nodes_mean))+ "\n")
    f_mean.write("Disen_fidelity_origin_nodes_mean = {}".format(list(fidelity_origin_nodes_mean))+ "\n")
    f_mean.write("Disen_fidelity_complete_nodes_mean = {}".format(list(fidelity_complete_nodes_mean))+ "\n")
    f_mean.write("Disen_fidelityminus_nodes_mean = {}".format(list(fidelityminus_nodes_mean))+"\n")
    f_mean.write("Disen_fidelityminus_origin_nodes_mean = {}".format(list(fidelityminus_origin_nodes_mean))+"\n")
    f_mean.write("Disen_fidelityminus_complete_nodes_mean = {}".format(list(fidelityminus_complete_nodes_mean))+"\n")
    f_mean.write("Disen_finalfidelity_complete_nodes_mean = {}".format(list(finalfidelity_complete_nodes_mean))+"\n")
    f_mean.write("Disen_sparsity_nodes_mean = {}".format(list(sparsity_nodes_mean))+ "\n")
    f_mean.close()



def test_onegraph(explain_graph_arr, explainModel_ckpt_path):
    starttime = time.time()
    top_k = 15
    GNNmodel_ckpt_path = osp.join('model_weights', args.dataset, 'gcn_3l_best.pth') 

    dataset = get_dataset(args.dataset_root, args.dataset)
    model = GnnNets(input_dim=dataset.num_node_features,  output_dim=dataset.num_classes, model_args=model_args)
    ckpt = torch.load(GNNmodel_ckpt_path)
    model.load_state_dict(ckpt['net'])

    explainer = ExplainerGCPGDisenMask(model=model, args=args)
    explainer.to(args.device)
    explainer.load_state_dict(torch.load(explainModel_ckpt_path) )
    
    for explain_graph in explain_graph_arr:
        sub_features = dataset.data.x[dataset.slices['x'][explain_graph].item():dataset.slices['x'][explain_graph+1].item(), :]
        sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][explain_graph].item():dataset.slices['edge_index'][explain_graph+1].item()]
        data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
        logits, prob, sub_embs = model(data)
        label = dataset.data.y[explain_graph]
        sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
        explainer.eval()
        masked_pred = explainer((sub_features, sub_embs, sub_adj, 1.0, label))
        mask = explainer.masked_adj
    print("get mask time: ",time.time()-starttime)
        
    '''origin_pred = prob.squeeze()
        ndcg_onegraph, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
            
        metric = MaskoutMetric(model, args)
        pred_mask, related_preds_dict = metric.metric_del_edges_GC([top_k], sub_features, mask, sub_edge_index, origin_pred, masked_pred, label)

        x_collector = XCollector()
        x_collector.collect_data(pred_mask, related_preds_dict[top_k], label=0)

        print("explain_graph,{}\n".format(explain_graph) + "ndcg_onenode={}".format(ndcg_onegraph) + "\n")
        print("fidelity_complete=", round(x_collector.fidelity_complete, 4))
        print("fidelityminus_complete=", round(x_collector.fidelityminus_complete, 4))
        print("finalfidelity_complete=", round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
        print("test time: ", time.time()-starttime)'''
    

if __name__ == "__main__":
    #BA_3Motifs_7class, Mutagenicity_full, NCI1, PROTEINS
    main("Mutagenicity_full", 3, 20)

    #experiment of hyper parameter ( channel number disen_k )
    '''for dataset_name in ["NCI1"]:
        for disen_k in [2, 4, 5]:
            main(dataset_name, disen_k, 20)'''
    
    #experiment of hyper parameter train_topk ( selected k in training stage )
    '''for dataset_name in ["NCI1"]:
        for train_topk in [5, 10, 15, 25, 30]:
            main(dataset_name, 3, train_topk)'''
    #explain_graph_arr = [random.randint(301, 600) for p in range(0, 100)]
    #explainModel_ckpt_path ="LOGS/"+args.dataset+"_final/0/"+args.dataset+"_BEST.pt"
    #test_onegraph(explain_graph_arr, explainModel_ckpt_path)