import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_max_pool
from math import sqrt
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn import ModuleList, Linear
from disenconv import  InitDisenLayer, RoutingLayer as RoutingLayerBak
from DisenModel import SparseInputLinear, NeibSampler, RoutingLayer
import networkx as nx


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', Linear(in_channels, hidden_channels)),
                ('act', nn.ReLU()),
                ('lin2', Linear(hidden_channels, out_channels))
                ]))
     
    def forward(self, x):
        return self.mlp(x)



class EdgeMaskNet(torch.nn.Module):

    def __init__(self,
                 inp_dim,
                 e_in_channels,
                 args):
        super(EdgeMaskNet, self).__init__()

        self.init_disen = InitDisenLayer(inp_dim, args.disen_hid_dim, args.init_k)
        
        self.conv_layers = nn.ModuleList()
        self.k = args.init_k
        self.args = args
        for _ in range(args.num_layers):
            self.conv_layers.append(RoutingLayerBak(self.k, args))
            self.k -= args.delta_k  
        
        self.dropout = args.dropout

        if e_in_channels > 1:
            self.edge_lin1 = Linear(2 * args.disen_hid_dim, args.exp_hid_dim)
            self.edge_lin2 = Linear(e_in_channels, args.exp_hid_dim)
        
        self.elayers = nn.Sequential(
                nn.Linear(2 * args.disen_hid_dim, args.exp_hid_dim),
                nn.ReLU(),
                nn.Linear(args.exp_hid_dim, 1)
            ).to(self.args.device)

        self._initialize_weights()
    
    def _dropout(self, X):
        return F.dropout(X, p=self.dropout, training=self.training)
        
    def forward(self, x, edge_index, edge_attr= None):
        Z = self.init_disen(x)
        for disen_conv in self.conv_layers:
            Z = disen_conv(Z, edge_index)
            Z = self._dropout(torch.relu(Z))
        #Z = Z.reshape(len(Z), -1)

        disen_graph_emb = []
        for i in range(self.k):
            batch = torch.zeros(Z.shape[0]).long().to(self.args.device)
            #disen_graph_emb.append(global_mean_pool(Z[:,i,:], batch))
            disen_graph_emb.append(global_max_pool(Z[:,i,:], batch))
        disen_graph_emb = torch.cat(disen_graph_emb)

        e_h_all = []
        for i in range(self.k):
            e_h = torch.cat([Z[edge_index[0, :], i], Z[edge_index[1, :],i]], dim=1)
            e_h = self.elayers(e_h)
            e_h_all.append(e_h.unsqueeze(0))

        e_h_all = torch.cat(e_h_all).squeeze(-1)
        e_h_final =  torch.max(e_h_all, dim=0).values
        return disen_graph_emb, Z, e_h_final, e_h_all

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight) 


class DisenEdgeMaskNet(torch.nn.Module):
    def __init__(self,
                 inp_dim,
                 args):
        super(DisenEdgeMaskNet, self).__init__()

        self.k = args.init_k
        elayers = []
        for i in range(self.k):
            elayers.append(nn.Sequential(
                nn.Linear(inp_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(args.device))
        self.elayers = nn.ModuleList(elayers)

    def forward(self, e_h):
        e_h_all = []
        for i in range(self.k):
            e_h = self.elayers[i](e_h)
            e_h_all.append(e_h.unsqueeze(0))

        e_h_all = torch.cat(e_h_all).squeeze(-1)
        e_h_final =  torch.max(e_h_all, dim=0).values
        return e_h_final, e_h_all



class ExplainerGCPGDisenMask(nn.Module):
    def __init__(self, model, args, **kwargs):
        super(ExplainerGCPGDisenMask, self).__init__(**kwargs)

        self.args = args
        self.device = model.device
        # input dims for the MLP is defined by the concatenation of the hidden layers of the GCN
        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]
        if args.concat:
            input_dim = sum(hiddens) * 2 # or just times 3?
        else:
            input_dim = hiddens[-1] * 2
        '''self.elayers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )'''

        #Disen node embeddings
        '''self.mask_net = EdgeMaskNet(
            inp_dim = self.args.inp_dim,
            e_in_channels=self.args.e_in_channels,
            args = self.args).to(args.device)'''

        #Disen mask by different MLP
        elayers = []
        for i in range(self.args.init_k):
            elayers.append(nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(args.device))
        self.elayers = nn.ModuleList(elayers)
        
        '''self.mask_net = DisenEdgeMaskNet(
            inp_dim = input_dim,
            args = self.args).to(args.device)'''
        
        self.model = model
        self.softmax = nn.Softmax(dim=-1)

        self.mask_act = 'sigmoid'
        #self.mask_act = 'RELU'
        self.init_bias = 0.0


    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        if training:
            debug_var = 0.0
            bias = 0.0
            random_noise = bias + torch.FloatTensor(log_alpha.shape).uniform_(debug_var, 1.0-debug_var)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs
    
    def __set_masks__(self, x, edge_index, edge_mask = None):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #std = torch.nn.init.calculate_gain('sigmoid') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module._edge_mask = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module._edge_mask = None
        #self.node_feat_masks = None
        self.edge_mask = None

        
    def forward(self, inputs, training=None):
        x, adj, embed, tmp, origin_pred = inputs
        self.tmp = tmp
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj)
        edge_index = torch.nonzero(adj).T.to(self.device)
        adj = adj.to(self.device)

        h = torch.cat([embed[edge_index[0]], embed[edge_index[1]]], dim=-1)
        h = h.to(self.device)

        #disen node embedding by disenGCN
        #disen_graph_emb, disen_node_embs, self.mask_logits,  self.disen_M = self.mask_net(x, edge_index)

        #disen mask by different MLP
        e_h_all = []
        for i in range(self.args.init_k):
            e_h = self.elayers[i](h)
            e_h_all.append(e_h.unsqueeze(0))
        e_h_all = torch.cat(e_h_all, dim=2)
        self.mask_logits =  torch.max(e_h_all, dim=2).values.squeeze(0)
        self.disen_M = e_h_all.squeeze(0).T

        values = self.concrete_sample(self.mask_logits, beta=tmp, training=training)
        nodesize = x.shape[0]

        sparsemask = torch.sparse_coo_tensor(
            indices=edge_index,
            values=values,
            size=[nodesize, nodesize]
        ).to(self.device)
        sym_mask = sparsemask.coalesce().to_dense().to(torch.float32)  #FIXME: again a reorder() is omitted, maybe coalesce
        #self.mask = sym_mask
        sym_mask = (sym_mask + sym_mask.T) / 2
        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj

        # modify model
        edge_mask = masked_adj[edge_index[0], edge_index[1]]
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        output, probs, _, _ = self.model(data)
        res = self.softmax(output.squeeze())
        self.__clear_masks__()

        select_k = round(self.args.train_topk/100 * len(edge_index[0]))

        disen_edge_mask_all = []
        disen_fidelityminus_complete = []
        disen_fidelityplus_complete = []
        for i in range(self.disen_M.shape[0]):
            disen_values = self.concrete_sample(self.disen_M[i], beta=tmp, training=training)

            disen_sparsemask = torch.sparse_coo_tensor(
                indices=edge_index,
                values=disen_values,
                size=[nodesize, nodesize]
            ).to(self.device)
            disen_sym_mask = disen_sparsemask.coalesce().to_dense().to(torch.float32)
            disen_sym_mask = (disen_sym_mask + disen_sym_mask.T) / 2
            disen_masked_adj = torch.mul(adj, disen_sym_mask)

            # modify model
            disen_edge_mask = disen_masked_adj[edge_index[0], edge_index[1]]
            disen_edge_mask_all.append(disen_edge_mask.unsqueeze(0))

            selected_impedges_idx = disen_edge_mask.reshape(-1).sort(descending=True).indices[:select_k]#按比例选择top_k%的重要边
            other_notimpedges_idx = disen_edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #按比例选择top_k%的重要边

            self.__clear_masks__()
            #self.__set_masks__(x, edge_index, disen_edge_mask)
            masknotimp_edge_mask = disen_edge_mask.clone()
            masknotimp_edge_mask[selected_impedges_idx] = 1.0
            masknotimp_edge_mask[other_notimpedges_idx] = disen_edge_mask[other_notimpedges_idx]
            self.__set_masks__(x, edge_index, masknotimp_edge_mask)
            data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
            output, disen_masknotimp_probs, _, _ = self.model(data)
            fidelityminus_complete_onenode = sum(abs(origin_pred - disen_masknotimp_probs.squeeze()))
            disen_fidelityminus_complete.append(fidelityminus_complete_onenode)
            self.__clear_masks__()

            #self.__set_masks__(x, edge_index, 1-disen_edge_mask)
            maskimp_edge_mask = disen_edge_mask.clone()
            maskimp_edge_mask[selected_impedges_idx] = 1-disen_edge_mask[selected_impedges_idx]
            maskimp_edge_mask[other_notimpedges_idx] = 1.0
            self.__set_masks__(x, edge_index, maskimp_edge_mask)
            data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
            output, disen_maskimp_probs, _, _ = self.model(data)
            fidelity_complete_onenode = sum(abs(origin_pred - disen_maskimp_probs.squeeze()))
            disen_fidelityplus_complete.append(fidelity_complete_onenode)
            self.__clear_masks__()
        disen_edge_mask_all = torch.cat(disen_edge_mask_all).squeeze(-1)
        self.disen_edge_mask_all = disen_edge_mask_all
        disen_edge_mask_max=  torch.max(disen_edge_mask_all, dim=0).values

        for i in range(self.disen_M.shape[0]):
            for j in range(i+1, self.disen_M.shape[0]):
                disen_edge_mask_max_temp=  torch.max(disen_edge_mask_all[[i,j]], dim=0).values
                self.__clear_masks__()
                #self.__set_masks__(x, edge_index, disen_edge_mask_max_temp)
                masknotimp_edge_mask_temp = disen_edge_mask_max_temp.clone()
                masknotimp_edge_mask_temp[selected_impedges_idx] = 1.0
                masknotimp_edge_mask_temp[other_notimpedges_idx] = disen_edge_mask_max_temp[other_notimpedges_idx]
                self.__set_masks__(x, edge_index, masknotimp_edge_mask_temp)
                data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
                output, disen_masknotimp_probs, _, _ = self.model(data)
                fidelityminus_complete_onenode = sum(abs(origin_pred - disen_masknotimp_probs.squeeze()))
                disen_fidelityminus_complete.append(fidelityminus_complete_onenode)
                self.__clear_masks__()

                #self.__set_masks__(x, edge_index, 1-disen_edge_mask_max_temp)
                maskimp_edge_mask_temp = disen_edge_mask_max_temp.clone()
                maskimp_edge_mask_temp[selected_impedges_idx] = 1-disen_edge_mask_max_temp[selected_impedges_idx]
                maskimp_edge_mask_temp[other_notimpedges_idx] = 1.0
                self.__set_masks__(x, edge_index, maskimp_edge_mask_temp)
                data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
                output, disen_maskimp_probs, _, _ = self.model(data)
                fidelity_complete_onenode = sum(abs(origin_pred - disen_maskimp_probs.squeeze()))
                disen_fidelityplus_complete.append(fidelity_complete_onenode)
                self.__clear_masks__()

        self.__clear_masks__()
        #self.__set_masks__(x, edge_index, disen_edge_mask_max)
        masknotimp_edge_mask_max = disen_edge_mask_max.clone()
        masknotimp_edge_mask_max[selected_impedges_idx] = 1.0
        masknotimp_edge_mask_max[other_notimpedges_idx] = disen_edge_mask_max[other_notimpedges_idx]
        self.__set_masks__(x, edge_index, masknotimp_edge_mask_max)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        output, disen_masknotimp_probs, _, _ = self.model(data)
        fidelityminus_complete_onenode = sum(abs(origin_pred - disen_masknotimp_probs.squeeze()))
        disen_fidelityminus_complete.append(fidelityminus_complete_onenode)
        self.__clear_masks__()
        #self.__set_masks__(x, edge_index, 1-disen_edge_mask_max)
        maskimp_edge_mask_max = disen_edge_mask_max.clone()
        maskimp_edge_mask_max[selected_impedges_idx] = 1-disen_edge_mask_max[selected_impedges_idx]
        maskimp_edge_mask_max[other_notimpedges_idx] = 1.0
        self.__set_masks__(x, edge_index, maskimp_edge_mask_max)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        output, disen_maskimp_probs, _, _ = self.model(data)
        fidelity_complete_onenode = sum(abs(origin_pred - disen_maskimp_probs.squeeze()))
        disen_fidelityplus_complete.append(fidelity_complete_onenode)
        self.__clear_masks__()
        return res, disen_fidelityminus_complete, disen_fidelityplus_complete


    def loss(self, pred, pred_label, disen_fidelityminus_complete, disen_fidelityplus_complete):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #gt_label_node = self.label
        #logit = pred[gt_label_node]
        logit = pred[pred_label]
        pred_loss = -torch.log(logit)
        
        # size
        #mask = self.mask
        mask = self.mask_logits
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(mask)
        size_loss = self.args.coff_size * torch.sum(mask) #len(mask[mask > 0]) #torch.sum(mask)

        for i in range(self.disen_M.shape[0]):
            disen_mask = self.disen_M[i]
            if self.mask_act == "sigmoid":
                disen_mask = torch.sigmoid(disen_mask)
            elif self.mask_act == "ReLU":
                disen_mask = nn.functional.relu(disen_mask)
            size_loss = size_loss + self.args.coff_size * torch.sum(disen_mask)

        #L0Norm
        '''self.gamma = -0.1
        self.zeta = 1.1
        self.eps = 1e-20
        size_loss =  self.args.coff_size * torch.sigmoid(self.mask_logits -  self.tmp*np.log(-self.gamma/self.zeta + self.eps)).sum()'''

        # entropy
        '''mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.args.coff_ent * torch.mean(mask_ent)'''
        mask_ent_loss = 0.0

        #orthogonality
        mask_orthogonal_rep = torch.matmul(self.disen_M, self.disen_M.T)
        mask_orthogonal_loss =  torch.abs(mask_orthogonal_rep-  torch.diag(torch.diag(mask_orthogonal_rep))).sum()
        #mask_orthogonal_loss =  0.01*mask_orthogonal_loss

        '''mask_orthogonal_rep = torch.matmul(self.disen_edge_mask_all, self.disen_edge_mask_all.T)
        mask_orthogonal_loss =  torch.abs(mask_orthogonal_rep-  torch.diag(torch.diag(mask_orthogonal_rep))).sum()'''
        #mask_orthogonal_loss =0
        
        fidelity_loss = 0
        fidelity_loss = fidelity_loss + torch.true_divide(1, disen_fidelityplus_complete[-1]+0.000001) + disen_fidelityminus_complete[-1]
        F_fidelity = 2/ (torch.true_divide(1, disen_fidelityplus_complete[-1]) + torch.true_divide(1, torch.true_divide(1, disen_fidelityminus_complete[-1])))
        for i in range(len(disen_fidelityminus_complete)-1):
            temp_F_fidelity = 2/ (torch.true_divide(1, disen_fidelityplus_complete[i]) + torch.true_divide(1, torch.true_divide(1, disen_fidelityminus_complete[i])))
            fl = temp_F_fidelity - F_fidelity
            fidelity_loss = fidelity_loss + fl
            #fidelity_loss = fidelity_loss + max(0, fl)
            #fidelity_loss = fidelity_loss + fl + self.args.lamda * max(0, fl)
        #fidelity_loss = 1000*fidelity_loss

        loss = pred_loss + size_loss + mask_ent_loss + mask_orthogonal_loss + fidelity_loss
        #loss = pred_loss + size_loss + mask_ent_loss + orthogonal_loss
        return loss, pred_loss, size_loss, mask_ent_loss, mask_orthogonal_loss, fidelity_loss


    def loss_hidden(self, pred, label, new_hidden_embs, hidden_embs):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #gt_label_node = self.label
        #logit = pred[gt_label_node]
        logit = pred[label]
        pred_loss = -torch.log(logit)

        cosine_hidden_loss=0
        for i in range(len(new_hidden_embs)):
            cosine_hidden_loss = cosine_hidden_loss + sum(1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1))

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        size_loss = self.args.coff_size * torch.sum(mask) #len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.args.coff_ent * torch.mean(mask_ent)

        loss = cosine_hidden_loss + pred_loss + size_loss + mask_ent_loss
        return loss
