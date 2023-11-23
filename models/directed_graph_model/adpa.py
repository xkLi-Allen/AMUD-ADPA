import torch
import torch.nn as nn
from models.base_scalable.base_model import ADPABaseModel
from models.base_scalable.complex_models import ADPA_Com2LayerGraphConvolution
from operators.graph_operator.spatial_directed_adpa_operator import DirADPAGraphOp
from operators.message_operator.adpa_message_operator import ADPAMessageOp

class ADPA(ADPABaseModel):
    def __init__(self, r, feat_dim, hidden_dim, output_dim, num_nodes, dropout, prop_steps, num_layers, task_level):
        super(ADPA, self).__init__()
        self.num_nodes = num_nodes

        self.naive_graph_op = DirADPAGraphOp(feat_dim = feat_dim, hidden_dim = hidden_dim, prop_steps=prop_steps, r=r)
        self.pre_msg_op = ADPAMessageOp(0, prop_steps+1, prop_steps, feat_dim, hidden_dim)
        self.base_model = ADPA_Com2LayerGraphConvolution(feat_dim=feat_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_nodes=self.num_nodes, num_layers=num_layers, dropout=dropout)