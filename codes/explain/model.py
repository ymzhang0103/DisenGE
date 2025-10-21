import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from codes.fourwork.disenconv_delnorm import RoutingLayer, InitDisenLayer

class DisenGCN(nn.Module):
    '''
    :param disen_hid_dim: Output embedding dimensions
    :param init_k: Maximum number of capsules 
    :param delta_k: Difference in the number of capsules per layer
    :param routit: Number of iterations when routing
    :param tau: Softmax temperature
    :param dropout: Dropout rate (1 - keep probability)
    :param num_layers: Number of conv layers
    '''
    def __init__(self, 
                 inp_dim,
                 num_classes,
                 args
                 ):
        super(DisenGCN, self).__init__()
        self.init_disen = InitDisenLayer(inp_dim, args.disen_hid_dim, args.init_k)
        
        self.conv_layers = nn.ModuleList()
        k = args.init_k
        for l in range(args.num_layers):
            fac_dim = args.disen_hid_dim // k
            self.conv_layers.append(RoutingLayer(k, args.routit, args.tau))
            inp_dim = fac_dim * k
            k -= args.delta_k   
        
        self.dropout = args.dropout
        self.classifier = nn.Linear(inp_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def _dropout(self, X):
        return F.dropout(X, p=self.dropout, training=self.training)
        
    def forward(self, X, edges):
        Z = self.init_disen(X)
        for disen_conv in self.conv_layers:
            Z = disen_conv(Z, edges)
            Z = self._dropout(torch.relu(Z))
        Z = Z.reshape(len(Z), -1)
        #Z = self.classifier(Z)
        return Z
