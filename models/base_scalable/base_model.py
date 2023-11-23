import time
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import scipy.sparse as sp
import torch.nn.functional as F

from models.utils import scipy_sparse_mat_to_torch_sparse_tensor

class ADPABaseModel(nn.Module):
    def __init__(self):
        super(ADPABaseModel, self).__init__()
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None
        self.num_nodes = None
        #1-hop and 2-hop neighbors
        self.original_feat_list = None
        self.a_processed_feat_list = None
        self.at_processed_feat_list = None
        self.aa_processed_feat_list = None
        self.atat_processed_feat_list = None
        self.aat_processed_feat_list = None
        self.ata_processed_feat_list = None

        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.naive_graph_op is not None:
            self.original_feat_list, self.a_processed_feat_list, self.at_processed_feat_list, self.aa_processed_feat_list, \
                self.aat_processed_feat_list, self.ata_processed_feat_list, self.atat_processed_feat_list = self.naive_graph_op.propagate(adj, feature)
        else:
            raise ValueError("TwoDirBaseSGModel must predefine the graph structure operator!")

        self.num_nodes = feature.size(0)
        self.pre_msg_learnable = True

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        original_processed_feature = [feat.to(device) for feat in self.original_feat_list]
        a_processed_feature = [feat.to(device) for feat in self.a_processed_feat_list]
        at_processed_feature = [feat.to(device) for feat in self.at_processed_feat_list]
        aa_processed_feature = [feat.to(device) for feat in self.aa_processed_feat_list]
        aat_processed_feature = [feat.to(device) for feat in self.aat_processed_feat_list]
        ata_processed_feature = [feat.to(device) for feat in self.ata_processed_feat_list]
        atat_processed_feature = [feat.to(device) for feat in self.atat_processed_feat_list]
        
        #7-op
        original_processed_feature, a_processed_feature, at_processed_feature, aa_processed_feature, \
            aat_processed_feature, ata_processed_feature, atat_processed_feature= \
            self.pre_msg_op.aggregate(original_processed_feature, a_processed_feature, at_processed_feature, aa_processed_feature, \
                                      aat_processed_feature, ata_processed_feature, atat_processed_feature)
        
    
        if ori is not None: self.base_model.query_edges = ori
        
        output = self.base_model(original_processed_feature, a_processed_feature, at_processed_feature, aa_processed_feature, aat_processed_feature, ata_processed_feature, atat_processed_feature)

        return output[idx] if self.base_model.query_edges is None else output