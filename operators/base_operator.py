import os
import platform

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.special import comb
from torch import Tensor

from operators.utils import csr_sparse_dense_matmul
from models.utils import scipy_sparse_mat_to_torch_sparse_tensor

class calculator:
    def __init__(self, value, r_step=0, i_step=0):
        self.value = value
        self.r_step = r_step
        self.i_step = i_step

    def prop_step(self):
        return self.r_step + self.i_step
    
    def reversal(self):
        if self.i_step & 1 == 0 and self.i_step != 0:
            self.value = -self.value
    
    def set_variable(self, value, r=False, i=False):
        self.value = value
        if r:   self.r_step += 1
        elif i: self.i_step += 1

class ADPAGraphOp:
    def __init__(self, feat_dim, hidden_dim, prop_steps, r=None):
        self.prop_steps = prop_steps
        self.r = r
        self.adj = None
        self.adj_t = None
        self.adj_aa = None
        self.adj_atat = None
        self.adj_aat = None
        self.adj_ata = None
        
        self.lamda = 1.0
        self.weight_decay = []


    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.adj, self.adj_t, self.adj_aa, self.adj_aat, self.adj_ata, self.adj_atat = self.construct_adj(adj)

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0] or self.adj_t.shape[1] != feature.shape[0] or self.adj_aat.shape[1] != feature.shape[0] or self.adj_ata.shape[1] != feature.shape[0] or self.adj_aa.shape[1] != feature.shape[0] or self.adj_atat.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        self.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.adj)
        self.adj_t = scipy_sparse_mat_to_torch_sparse_tensor(self.adj_t)
        self.adj_aa = scipy_sparse_mat_to_torch_sparse_tensor(self.adj_aa)
        self.adj_aat = scipy_sparse_mat_to_torch_sparse_tensor(self.adj_aat)
        self.adj_ata = scipy_sparse_mat_to_torch_sparse_tensor(self.adj_ata)
        self.adj_atat = scipy_sparse_mat_to_torch_sparse_tensor(self.adj_atat)

        feature = torch.FloatTensor(feature)

        original_feat_list = [feature]
        a_prop_feat_list = [feature]
        at_prop_feat_list = [feature]
        aa_prop_feat_list = [feature]
        atat_prop_feat_list = [feature]
        aat_prop_feat_list = [feature]
        ata_prop_feat_list = [feature]

        for iterate in range(self.prop_steps):
            self.weight_decay.append( np.log(self.lamda / (iterate+1))+1+1e-6 )

        for iterate in range(self.prop_steps):
            a_feat_temp = self.adj @ (self.weight_decay[iterate] * a_prop_feat_list[-1])
            at_feat_temp = self.adj_t @ (self.weight_decay[iterate] * at_prop_feat_list[-1])
            aa_feat_temp = self.adj_aa @ (self.weight_decay[iterate] * aa_prop_feat_list[-1])
            atat_feat_temp = self.adj_atat @ (self.weight_decay[iterate] * atat_prop_feat_list[-1])
            aat_feat_temp = self.adj_aat @ (self.weight_decay[iterate] * aat_prop_feat_list[-1])
            ata_feat_temp = self.adj_ata @ (self.weight_decay[iterate] * ata_prop_feat_list[-1])
            original_feat_temp = original_feat_list[-1]
        
            original_feat_list.append(original_feat_temp)
            a_prop_feat_list.append(a_feat_temp)
            at_prop_feat_list.append(at_feat_temp)
            aa_prop_feat_list.append(aa_feat_temp)
            aat_prop_feat_list.append(aat_feat_temp)
            ata_prop_feat_list.append(ata_feat_temp)
            atat_prop_feat_list.append(atat_feat_temp)

        return [torch.FloatTensor(feat) for feat in original_feat_list], \
            [torch.FloatTensor(feat) for feat in a_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in at_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in aa_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in aat_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in ata_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in atat_prop_feat_list]


def ada_platform_one_step_propagation(adj, x):
    if platform.system() == "Linux":
        one_step_prop_x = csr_sparse_dense_matmul(adj, x)
    else:
        one_step_prop_x = adj.dot(x)
    return one_step_prop_x

class ADPABaseMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(ADPABaseMessageOp, self).__init__()
        self._aggr_type = None
        self.start, self.end = start, end

    @property
    def aggr_type(self):
        return self.aggr_type

    def combine(self, original_feat_list, a_feat_list, at_feat_list, aa_feat_list, aat_feat_list, ata_feat_list, atat_feat_list):
        return NotImplementedError

    def aggregate(self, original_feat_list, a_feat_list, at_feat_list, aa_feat_list, aat_feat_list, ata_feat_list, atat_feat_list):
        if not isinstance(a_feat_list, list) or not isinstance(at_feat_list, list) or not isinstance(aa_feat_list, list) or not isinstance(aat_feat_list, list) or not isinstance(ata_feat_list, list) or not isinstance(atat_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in a_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self.combine(original_feat_list, a_feat_list, at_feat_list, aa_feat_list, aat_feat_list, ata_feat_list, atat_feat_list)