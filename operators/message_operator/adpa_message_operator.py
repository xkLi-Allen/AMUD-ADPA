import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, Linear

from ..base_operator import ADPABaseMessageOp
from ..utils import one_dim_weighted_add, two_dim_weighted_add

class ADPAMessageOp(ADPABaseMessageOp):
    def __init__(self, start, end, *args):
        super(ADPAMessageOp, self).__init__(start, end)
        self._aggr_type = "learnable_weighted"
        #attention mechanisms: original, simple, gate, ori_ref, none
        self.use_attention="original"
        self.learnable_weight = None

        if len(args) != 3:
            raise ValueError(
                "Invalid parameter numbers for the learnable weighted aggregator!")
        prop_steps, feat_dim, hidden_dim = args[0], args[1], args[2]
        
        if self.use_attention=="simple":
            self.a_learnable_weight = Parameter(torch.FloatTensor(1, prop_steps+1).view(-1))
            self.at_learnable_weight = Parameter(torch.FloatTensor(1, prop_steps+1).view(-1))
            self.aa_learnable_weight = Parameter(torch.FloatTensor(1, prop_steps+1).view(-1))
            self.aat_learnable_weight = Parameter(torch.FloatTensor(1, prop_steps+1).view(-1))
            self.ata_learnable_weight = Parameter(torch.FloatTensor(1, prop_steps+1).view(-1))
            self.atat_learnable_weight = Parameter(torch.FloatTensor(1, prop_steps+1).view(-1))
        
        elif self.use_attention=="gate":
            self.a_learnable_weight = Linear(feat_dim, 1)
            self.at_learnable_weight = Linear(feat_dim, 1)
            self.aa_learnable_weight = Linear(feat_dim, 1)
            self.aat_learnable_weight = Linear(feat_dim, 1)
            self.ata_learnable_weight = Linear(feat_dim, 1)
            self.atat_learnable_weight = Linear(feat_dim, 1)
            
        elif self.use_attention=="ori_ref":
            self.a_learnable_weight = Linear(feat_dim + feat_dim, 1)
            self.at_learnable_weight = Linear(feat_dim + feat_dim, 1)
            self.aa_learnable_weight = Linear(feat_dim + feat_dim, 1)
            self.aat_learnable_weight = Linear(feat_dim + feat_dim, 1)
            self.ata_learnable_weight = Linear(feat_dim + feat_dim, 1)
            self.atat_learnable_weight = Linear(feat_dim + feat_dim, 1)
            
        else:
            self.a_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
            self.at_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
            self.aa_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
            self.aat_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
            self.ata_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
            self.atat_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
            self.apat_learnable_weight = Linear(
            feat_dim+ (prop_steps + 1) * feat_dim, 1)
        

    def original_attention(self, feat_list, learnable_weight):
    
        reference_feat = torch.hstack(feat_list).repeat(
                self.end - self.start, 1)
        adopted_feat_list = torch.hstack(
            (reference_feat, torch.vstack(feat_list[self.start:self.end])))
        weight_list = F.softmax(
            torch.sigmoid(learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)
        weighted_feat = two_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        
        return weighted_feat
        
    def simple_attention(self, feat_list, learnable_weight):
        weight_list = F.softmax(torch.sigmoid(
                learnable_weight[self.start:self.end]), dim=0)
        weighted_feat = one_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
                
        return weighted_feat
                
    def gate_attention(self, feat_list, learnable_weight):
        adopted_feat_list = torch.vstack(feat_list[self.start:self.end])
        weight_list = F.softmax(
                torch.sigmoid(learnable_weight(adopted_feat_list).view(self.end - self.start, -1).T), dim=1)
        weighted_feat = two_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        
        return weighted_feat
        
    def ori_attention(self, feat_list, learnable_weight):
        reference_feat = feat_list[0].repeat(self.end - self.start, 1)
        adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self.start:self.end])))
        weight_list = F.softmax(
                torch.sigmoid(learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)
                
        weighted_feat = two_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        
        return weighted_feat
        
    def none_attention(self, feat_list, learnable_weight):
        weighted_feat = feat_list[0]
        for id,feat in enumerate(feat_list):
            if id>0:
                weighted_feat += feat_list[id]
        
        return weighted_feat

    def combine(self, original_feat_list, a_feat_list, at_feat_list, aa_feat_list, aat_feat_list, ata_feat_list, atat_feat_list):
        weight_list = None
        original_weighted_feat = original_feat_list[0]

        if self.use_attention=="simple":
            a_weighted_feat = self.simple_attention(a_feat_list, self.a_learnable_weight)
            at_weighted_feat = self.simple_attention(at_feat_list, self.at_learnable_weight)
            aa_weighted_feat = self.simple_attention(aa_feat_list, self.aa_learnable_weight)
            aat_weighted_feat = self.simple_attention(aat_feat_list, self.aat_learnable_weight)
            ata_weighted_feat = self.simple_attention(ata_feat_list, self.ata_learnable_weight)
            atat_weighted_feat = self.simple_attention(atat_feat_list, self.atat_learnable_weight)
            
        elif self.use_attention=="gate":
            a_weighted_feat = self.gate_attention(a_feat_list, self.a_learnable_weight)
            at_weighted_feat = self.gate_attention(at_feat_list, self.at_learnable_weight)
            aa_weighted_feat = self.gate_attention(aa_feat_list, self.aa_learnable_weight)
            aat_weighted_feat = self.gate_attention(aat_feat_list, self.aat_learnable_weight)
            ata_weighted_feat = self.gate_attention(ata_feat_list, self.ata_learnable_weight)
            atat_weighted_feat = self.gate_attention(atat_feat_list, self.atat_learnable_weight)
            
        elif self.use_attention=="ori_ref":
            a_weighted_feat = self.ori_attention(a_feat_list, self.a_learnable_weight)
            at_weighted_feat = self.ori_attention(at_feat_list, self.at_learnable_weight)
            aa_weighted_feat = self.ori_attention(aa_feat_list, self.aa_learnable_weight)
            aat_weighted_feat = self.ori_attention(aat_feat_list, self.aat_learnable_weight)
            ata_weighted_feat = self.ori_attention(ata_feat_list, self.ata_learnable_weight)
            atat_weighted_feat = self.ori_attention(atat_feat_list, self.atat_learnable_weight)
            
        elif self.use_attention=="original":
            a_weighted_feat = self.original_attention(a_feat_list, self.a_learnable_weight)
            at_weighted_feat = self.original_attention(at_feat_list, self.at_learnable_weight)
            aa_weighted_feat = self.original_attention(aa_feat_list, self.aa_learnable_weight)
            aat_weighted_feat = self.original_attention(aat_feat_list, self.aat_learnable_weight)
            ata_weighted_feat = self.original_attention(ata_feat_list, self.ata_learnable_weight)
            atat_weighted_feat = self.original_attention(atat_feat_list, self.atat_learnable_weight)
            
        else:
            a_weighted_feat = self.none_attention(a_feat_list, self.a_learnable_weight)
            at_weighted_feat = self.none_attention(at_feat_list, self.at_learnable_weight)
            aa_weighted_feat = self.none_attention(aa_feat_list, self.aa_learnable_weight)
            aat_weighted_feat = self.none_attention(aat_feat_list, self.aat_learnable_weight)
            ata_weighted_feat = self.none_attention(ata_feat_list, self.ata_learnable_weight)
            atat_weighted_feat = self.none_attention(atat_feat_list, self.atat_learnable_weight)
        
        return original_weighted_feat, a_weighted_feat, at_weighted_feat, aa_weighted_feat, aat_weighted_feat, ata_weighted_feat, atat_weighted_feat
