import torch
import torch.nn as nn
import models.init as complexinit

from scipy.sparse import csr_matrix
from torch_geometric.utils import add_self_loops, degree
from models.utils import scipy_sparse_mat_to_torch_sparse_tensor
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
import scipy.sparse as sp
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_uniform
from ..utils import one_dim_weighted_add, two_dim_weighted_add
            
class ADPA_Com2LayerGraphConvolution(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, num_nodes, num_layers, dropout=0.5):
        super(ADPA_Com2LayerGraphConvolution, self).__init__()
        self.input_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.query_edges = None
        self.num_nodes = num_nodes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        #attention mechanisms: original, gate, ori_ref, jk, none
        self.use_attention = "original"

        self.linear_o = nn.ModuleList()
        self.linear_a = nn.ModuleList()
        self.linear_at = nn.ModuleList()
        self.linear_aa = nn.ModuleList()
        self.linear_aat = nn.ModuleList()
        self.linear_ata = nn.ModuleList()
        self.linear_atat = nn.ModuleList()

        self.linear_o.append(nn.Linear(feat_dim, hidden_dim))
        self.linear_a.append(nn.Linear(feat_dim, hidden_dim))
        self.linear_at.append(nn.Linear(feat_dim, hidden_dim))
        self.linear_aa.append(nn.Linear(feat_dim, hidden_dim))
        self.linear_aat.append(nn.Linear(feat_dim, hidden_dim))
        self.linear_ata.append(nn.Linear(feat_dim, hidden_dim))
        self.linear_atat.append(nn.Linear(feat_dim, hidden_dim))
        
        if self.use_attention=="original":
            self.o_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
            self.a_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
            self.at_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
            self.aa_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
            self.aat_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
            self.ata_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
            self.atat_learnable_weight = nn.Parameter(torch.randn(size=(1,1)), requires_grad=True)
        
        elif self.use_attention=="gate":
            self.learnable_weight = nn.Linear(hidden_dim, 1)
            
        elif self.use_attention=="ori_ref":
            self.learnable_weight = nn.Linear((1+1)*hidden_dim, 1)
            
        else:
        #jk attention
            self.learnable_weight = nn.Linear(7*hidden_dim, 7)

        for id in range(self.num_layers-1):
            self.linear_o.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_a.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_at.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_aa.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_aat.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_ata.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_atat.append(nn.Linear(hidden_dim, hidden_dim))

        if self.use_attention=="original":
            self.linear_output = nn.Linear(7*hidden_dim, output_dim)
        else:
            self.linear_output = nn.Linear(hidden_dim, output_dim)

        self.adj, self.adj_t, self.adj_aa, self.adj_aat, self.adj_ata, self.adj_atat = None, None, None, None, None, None

    def forward(self, original_feature, a_feature, at_feature, aa_feature, aat_feature, ata_feature, atat_feature):
        #initilize
        self.num_nodes = aat_feature.size(0)

        x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat = original_feature, a_feature, at_feature, aa_feature, aat_feature, ata_feature, atat_feature
        for layer in range(self.num_layers):
            x_o = self.linear_o[layer](x_o)
            x_a = self.linear_a[layer](x_a)
            x_at = self.linear_at[layer](x_at)      
            x_aa = self.linear_aa[layer](x_aa)
            x_aat = self.linear_aat[layer](x_aat)
            x_ata = self.linear_ata[layer](x_ata)
            x_atat = self.linear_atat[layer](x_atat)

            if layer<self.num_layers:
                x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat = \
                    self.relu(x_o), self.relu(x_a), self.relu(x_at), self.relu(x_aa), self.relu(x_aat), self.relu(x_ata), self.relu(x_atat)
                x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat = self.dropout(x_o), self.dropout(x_a), self.dropout(x_at), self.dropout(x_aa), self.dropout(x_aat), self.dropout(x_ata), self.dropout(x_atat)
        
        if self.use_attention=="original":
            x = self.simple_attention(x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat)
            
        elif self.use_attention=="gate":
            x = self.gate_attention(x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat)
            
        elif self.use_attention=="ori_ref":
            x = self.ori_attention(x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat)
            
        elif self.use_attention=="jk":
            x = self.jk_attention(x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat)
            
        else:
            x = x_o+x_a+x_at+x_aa+x_aat+x_ata+x_atat
        
        x = self.linear_output(x)

        return x

    def simple_attention(self, x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat):
        x = torch.cat((self.o_learnable_weight*x_o, self.a_learnable_weight*x_a, self.at_learnable_weight*x_at, self.aa_learnable_weight*x_aa, self.aat_learnable_weight*x_aat, self.ata_learnable_weight*x_ata, self.atat_learnable_weight*x_atat), axis=-1)
        return x
        
    def jk_attention(self, x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat):
        x_list = [x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat]
        x = torch.hstack(x_list)
        x_weight_list = F.softmax(
            torch.sigmoid(self.learnable_weight(x)), dim=1
        )
        x = two_dim_weighted_add(
            x_list, weight_list=x_weight_list)
        return x
      
    def gate_attention(self, x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat):
        x_list = [x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat]  
        adopted_feat_list = torch.vstack(x_list)
        x_weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(7-0, -1).T), dim=1)
        x = two_dim_weighted_add(
                x_list, weight_list=x_weight_list)
        return x
        
    def ori_attention(self, x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat):
        x_list = [x_o, x_a, x_at, x_aa, x_aat, x_ata, x_atat]
        x_list_1 = x_list[0].repeat(7,1)
        adopted_feat_list = torch.hstack(
                (x_list_1, torch.vstack(x_list)))
        x_weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(-1, 7-0)), dim=1)
                
        x = two_dim_weighted_add(
                x_list, weight_list=x_weight_list)       
        return x