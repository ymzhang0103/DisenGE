import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn



# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


# noinspection PyUnresolvedReferences
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        #n =  max(graph.nodes()) + 1
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) <= n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
        #for v in graph.nodes():
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


# noinspection PyUnresolvedReferences
class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter):
        #nb = neighbors.view(-1)
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        #n, m = neighbors.size(0), neighbors.size(1)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        #x = fn.normalize(x.view(x.size(0), k, delta_d), dim=2).view(x.size(0), d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        #print("z[neighbors].shape", z[neighbors].shape)
        #print("n, m, k, delta_d", n, m, k, delta_d)
        z = z[neighbors].view(n, m, k, delta_d)
        #z = z[neighbors].view(neighbors.size(0), neighbors.size(1), k, delta_d)
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            #u += x.view(x.size(0), k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)


class CapsuleNet(nn.Module):  # CapsuleNet = DisenGCN
    def __init__(self, nfeat, nclass, hyperpm):
        super(CapsuleNet, self).__init__()
        ncaps, rep_dim = hyperpm.ncaps, hyperpm.nhidden * hyperpm.ncaps
        self.pca = SparseInputLinear(nfeat, rep_dim)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass)
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        nb = nb.view(-1)
        x = fn.relu(self.pca(x))
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        x = self.mlp(x)
        return fn.log_softmax(x, dim=1)

