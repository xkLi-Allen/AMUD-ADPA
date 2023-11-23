import scipy.sparse as sp

from operators.base_operator import ADPAGraphOp
from operators.utils import adj_to_directed_ADPA_norm


class DirADPAGraphOp(ADPAGraphOp):
    def __init__(self, feat_dim, hidden_dim, prop_steps, r=0.5):
        super(DirADPAGraphOp, self).__init__(feat_dim, hidden_dim, prop_steps, r)
        self.r = r

    def construct_adj(self, adj):
        adj = adj.tocoo()
        adj_normalized, adj_t_normalized, adj_aa_normalized, adj_aat_normalized, adj_ata_normalized, adj_atat_normalized = adj_to_directed_ADPA_norm(adj, self.r)
        return adj_normalized, adj_t_normalized, adj_aa_normalized, adj_aat_normalized, adj_ata_normalized, adj_atat_normalized