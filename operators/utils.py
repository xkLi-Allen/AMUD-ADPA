import math
import torch
import scipy
import numpy as np
import os.path as osp
import scipy.sparse as sp
import numpy.ctypeslib as ctl

from ctypes import c_int
from torch import Tensor
from torch_sparse import coalesce
from scipy.sparse import csr_matrix
from torch_scatter import scatter_add
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix


def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten().astype(np.float32)
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("./csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def adj_to_directed_ADPA_norm(adj, r):
    num_nodes = adj.shape[0]
    adj_t = adj.T
    adj_aa = adj @ adj
    adj_atat = adj_t @ adj_t
    adj_aat = adj @ adj_t
    adj_ata = adj_t @ adj

    adj = adj+sp.eye(adj.shape[0])
    adj_t = adj_t+sp.eye(adj_t.shape[0])

    degrees_a = np.array(adj.sum(1))
    degrees_at = np.array(adj_t.sum(1))
    degrees_aa = np.array(adj_aa.sum(1))
    degrees_aat = np.array(adj_aat.sum(1))
    degrees_ata = np.array(adj_ata.sum(1))
    degrees_atat = np.array(adj_atat.sum(1))
    
    r_inv_sqrt_left = np.power(degrees_a, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees_a, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_normalized = r_mat_inv_sqrt_left @ adj @ r_mat_inv_sqrt_right

    r_inv_sqrt_left = np.power(degrees_at, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees_at, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_t_normalized = r_mat_inv_sqrt_left @ adj_t @ r_mat_inv_sqrt_right

    r_inv_sqrt_left = np.power(degrees_aa, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees_aa, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_aa_normalized = r_mat_inv_sqrt_left @ adj_aa @ r_mat_inv_sqrt_right

    r_inv_sqrt_left = np.power(degrees_aat, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees_aat, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_aat_normalized = r_mat_inv_sqrt_left @ adj_aat @ r_mat_inv_sqrt_right

    r_inv_sqrt_left = np.power(degrees_ata, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees_ata, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_ata_normalized = r_mat_inv_sqrt_left @ adj_ata @ r_mat_inv_sqrt_right

    r_inv_sqrt_left = np.power(degrees_atat, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees_atat, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_atat_normalized = r_mat_inv_sqrt_left @ adj_atat @ r_mat_inv_sqrt_right

    adj_normalized, adj_t_normalized, adj_aa_normalized, adj_aat_normalized, adj_ata_normalized, adj_atat_normalized = \
        adj_normalized.tocsr(), adj_t_normalized.tocsr(), adj_aa_normalized.tocsr(), adj_aat_normalized.tocsr(), adj_ata_normalized.tocsr(), adj_atat_normalized.tocsr()

    return adj_normalized, adj_t_normalized, adj_aa_normalized, adj_aat_normalized, adj_ata_normalized, adj_atat_normalized
    
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

def squeeze_first_dimension(feat_list):
    if isinstance(feat_list, Tensor):
        if len(feat_list.shape) == 3:
            feat_list = feat_list[0]
    elif isinstance(feat_list, list):
        if len(feat_list[0].shape) == 3:
            for i in range(len(feat_list)):
                feat_list[i] = feat_list[i].squeeze(dim=0)
    return feat_list

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)