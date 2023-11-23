import torch
import numpy as np
import scipy.sparse as sp
import math
import torch
import scipy
import os.path as osp
import numpy.ctypeslib as ctl

from ctypes import c_int
from torch import Tensor
from torch_sparse import coalesce
from scipy.sparse import csr_matrix
from torch_scatter import scatter_add
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def one_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError("This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[0]:
        raise ValueError("The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 1:
        raise ValueError("The weight list should be a 1d tensor!")

    feat_shape = feat_list[0].shape
    feat_reshape = torch.vstack([feat.view(1, -1).squeeze(0) for feat in feat_list])
    weighted_feat = (feat_reshape * weight_list.view(-1, 1)).sum(dim=0).view(feat_shape)
    return weighted_feat

def two_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError("This function is designed for list(feature) and tensor(weight)!")
    elif len(feat_list) != weight_list.shape[1]:
        raise ValueError("The feature list and the weight list have different lengths!")
    elif len(weight_list.shape) != 2:
        raise ValueError("The weight list should be a 2d tensor!")

    feat_reshape = torch.stack(feat_list, dim=2)
    weight_reshape = weight_list.unsqueeze(dim=2)
    weighted_feat = torch.bmm(feat_reshape, weight_reshape).squeeze(dim=2)
    return weighted_feat
