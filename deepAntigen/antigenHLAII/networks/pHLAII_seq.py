import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_geometric.nn.dense.linear as pyg_linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import global_add_pool, global_max_pool
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from .top_k_pooling_seq import TopKPooling

class TGCN(MessagePassing):
    def __init__(self, hidden_channels, K, aggr='add'):
        super(TGCN, self).__init__(aggr=aggr)
        self.k_head = K
        self.message_w = pyg_linear.Linear(hidden_channels+11, hidden_channels, weight_initializer='kaiming_uniform')
        self.update_w = pyg_linear.Linear(2*hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.vatt_w = nn.ModuleList([pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
                                     for i in range(self.k_head)])
        self.satt_w = nn.ModuleList([pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
                                     for i in range(self.k_head)])
        self.att_w = nn.ModuleList([pyg_linear.Linear(hidden_channels, 1,  weight_initializer='kaiming_uniform')
                                    for i in range(self.k_head)])
        self.x_to_s_w = pyg_linear.Linear(self.k_head*hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.s_w = pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.s_to_x_w = pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.gate11 = pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.gate12 = pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.gate21 = pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.gate22 = pyg_linear.Linear(hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')

    def forward(self, x_0, s_0, edge_index, edge_attr, ibatch):
        x_u = self.propagate(edge_index, x=x_0, edge_attr=edge_attr, size=None)
        s_0_expan = s_0[ibatch, :]
        s_u = torch.tanh(self.s_w(s_0))
        s_to_x_u = torch.tanh(self.s_to_x_w(s_0))
        s_to_x_u_expan = s_to_x_u[ibatch, :]
        x_to_s_list = []
        for k in range(self.k_head):
            b = torch.tanh(self.vatt_w[k](x_0)) * torch.tanh(self.satt_w[k](s_0_expan))
            att = self.att_w[k](b)
            softmax_att = torch_scatter.scatter_softmax(att, ibatch, dim=0)
            x_to_s = global_add_pool(x_0*softmax_att, ibatch)
            x_to_s_list.append(x_to_s)
        x_to_s_u = torch.tanh(self.x_to_s_w(torch.cat(x_to_s_list, dim=1)))
        x_to_s_g = torch.sigmoid(self.gate11(x_to_s_u)+self.gate12(s_u))
        s_l = x_to_s_g*x_to_s_u+(1-x_to_s_g)*s_u
        s_to_x_g = torch.sigmoid(self.gate21(s_to_x_u_expan)+self.gate22(x_u))
        x_l = s_to_x_g*s_to_x_u_expan+(1-s_to_x_g*x_u)
        return x_l, s_l

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        T = self.message_w(torch.cat((x_j, edge_attr), dim=1))
        return F.leaky_relu(T, 0.1)

    def update(self, inputs: Tensor, x) -> Tensor:
        output = self.update_w(torch.cat((inputs, x), dim=1))
        return F.leaky_relu(output, 0.1)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.W_MHC = nn.Linear(self.hidden_size, self.hidden_size * self.n_heads)
        self.W_Peptide = nn.Linear(self.hidden_size, self.hidden_size * self.n_heads)
        self.reset_param()

    def reset_param(self):
        nn.init.xavier_uniform_(self.W_MHC.weight)
        nn.init.xavier_uniform_(self.W_Peptide.weight)
    
    def forward(self, peptide, mhc):
        batch_size = peptide.size(0)

        mhc_s = self.W_MHC(mhc).view(batch_size, -1, self.n_heads, self.hidden_size).transpose(1, 2)
        peptide_s = self.W_Peptide(peptide).view(batch_size, -1, self.n_heads, self.hidden_size).transpose(1, 2)

        scores = torch.matmul(peptide_s, mhc_s.transpose(-1, -2)) / self.hidden_size
        scores = torch.mean(scores, dim=1)
        scores_reshape = scores.view(scores.shape[0],-1)
        att = torch.softmax(scores_reshape, dim=1)
        att = att.view(scores.shape[0],scores.shape[1],scores.shape[2])
        att = att.unsqueeze(-1)
        intermap = peptide.unsqueeze(-3) + mhc.unsqueeze(-2)
        output = torch.sum(intermap * att, dim=(1, 2))
        return output
        
class DeepGCN(MessagePassing):
    def __init__(self, args, aggr='add'):
        super(DeepGCN, self).__init__(aggr=aggr)
        self.depth = args['depth']
        self.init_pep = pyg_linear.Linear(25, args['hidden_size'], weight_initializer='kaiming_uniform')
        self.init_pse = pyg_linear.Linear(25, args['hidden_size'], weight_initializer='kaiming_uniform')
        self.gcn_pep = nn.ModuleList([TGCN(args['hidden_size'],args['heads']) for i in range(args['depth'])])
        self.bn_pep = nn.ModuleList([BatchNorm(args['hidden_size']) for i in range(args['depth'])])
        self.bn_pep_s = nn.ModuleList([BatchNorm(args['hidden_size']) for i in range(args['depth'])])
        self.gcn_pse = nn.ModuleList([TGCN(args['hidden_size'],args['heads']) for i in range(args['depth'])])
        self.bn_pse = nn.ModuleList([BatchNorm(args['hidden_size']) for i in range(args['depth'])])
        self.bn_pse_s = nn.ModuleList([BatchNorm(args['hidden_size']) for i in range(args['depth'])])
        self.pep_to_pse_s = nn.ModuleList([pyg_linear.Linear(args['hidden_size'], args['hidden_size'], weight_initializer='kaiming_uniform') for i in range(args['depth'])])
        self.pse_to_pep_s = nn.ModuleList([pyg_linear.Linear(args['hidden_size'], args['hidden_size'], weight_initializer='kaiming_uniform') for i in range(args['depth'])])
        self.gate_pep1 = nn.ModuleList([pyg_linear.Linear(args['hidden_size'], args['hidden_size'], weight_initializer='kaiming_uniform') for i in range(args['depth'])])
        self.gate_pep2 = nn.ModuleList([pyg_linear.Linear(args['hidden_size'], args['hidden_size'], weight_initializer='kaiming_uniform') for i in range(args['depth'])])
        self.gate_pse1 = nn.ModuleList([pyg_linear.Linear(args['hidden_size'], args['hidden_size'], weight_initializer='kaiming_uniform') for i in range(args['depth'])])
        self.gate_pse2 = nn.ModuleList([pyg_linear.Linear(args['hidden_size'], args['hidden_size'], weight_initializer='kaiming_uniform') for i in range(args['depth'])])
        self.gru_pep = nn.ModuleList([nn.GRUCell(args['hidden_size'], args['hidden_size']) for i in range(args['depth'])])
        self.gru_pep_s = nn.ModuleList([nn.GRUCell(args['hidden_size'], args['hidden_size']) for i in range(args['depth'])])
        self.gru_pse = nn.ModuleList([nn.GRUCell(args['hidden_size'], args['hidden_size']) for i in range(args['depth'])])
        self.gru_pse_s = nn.ModuleList([nn.GRUCell(args['hidden_size'], args['hidden_size']) for i in range(args['depth'])])
        self.top_K_pep = TopKPooling(args['hidden_size'], ratio=args['k'])
        self.top_K_pse = TopKPooling(args['hidden_size'], ratio=args['k'])
        self.peptide_pseudo_att = CrossAttention(args['hidden_size'], args['heads'])
        self.dropout = nn.Dropout(p=0.2)
        self.projector = pyg_linear.Linear(args['hidden_size'], int(0.5*args['hidden_size']), weight_initializer='kaiming_uniform')
        self.classier = pyg_linear.Linear(int(0.5*args['hidden_size']), 2, weight_initializer='kaiming_uniform')

    def forward(self, peptide_graphs, pseudo_graphs):
        x_pep, edge_index_pep, edge_attr_pep, ibatch_pep = peptide_graphs.x, peptide_graphs.edge_index, peptide_graphs.edge_attr, peptide_graphs.batch
        x_pse, edge_index_pse, edge_attr_pse, ibatch_pse = pseudo_graphs.x, pseudo_graphs.edge_index, pseudo_graphs.edge_attr, pseudo_graphs.batch
        x_pep_l = F.leaky_relu(self.init_pep(x_pep), 0.1)
        s_pep_l = global_add_pool(x_pep_l, ibatch_pep)
        x_pse_l = F.leaky_relu(self.init_pse(x_pse), 0.1)
        s_pse_l = global_add_pool(x_pse_l, ibatch_pse)
        for l in range(self.depth):
            x_pep_u, s_pep_u = self.gcn_pep[l](x_pep_l, s_pep_l, edge_index_pep, edge_attr_pep, ibatch_pep)
            x_pse_u, s_pse_u = self.gcn_pse[l](x_pse_l, s_pse_l, edge_index_pse, edge_attr_pse, ibatch_pse)
            pep_to_pse = self.pep_to_pse_s[l](s_pep_u)
            pse_to_pep = self.pse_to_pep_s[l](s_pse_u)
            pse_to_pep_g = torch.sigmoid(self.gate_pep1[l](pse_to_pep)+self.gate_pep2[l](s_pep_u))
            s_pep_u = pse_to_pep_g*pse_to_pep+(1-pse_to_pep_g)*s_pep_u
            pep_to_pse_g = torch.sigmoid(self.gate_pse1[l](pep_to_pse)+self.gate_pse2[l](s_pse_u))
            s_pse_u = pep_to_pse_g*pep_to_pse+(1-pep_to_pse_g)*s_pse_u
            x_pep_l = self.gru_pep[l](x_pep_l, x_pep_u)
            x_pse_l = self.gru_pse[l](x_pse_l, x_pse_u)
            s_pep_l = self.gru_pep_s[l](s_pep_l, s_pep_u)
            s_pse_l = self.gru_pse_s[l](s_pse_l, s_pse_u)
            x_pep_l = self.bn_pep[l](x_pep_l)
            x_pse_l = self.bn_pse[l](x_pse_l)
            s_pep_l = self.bn_pep_s[l](s_pep_l)
            s_pse_l = self.bn_pse_s[l](s_pse_l)
        peptide_fs, _ = self.top_K_pep(x_pep_l,batch=ibatch_pep)
        # peptide_global = global_max_pool(x_pep_l, ibatch_pep)
        # peptide_fs = peptide_global.unsqueeze(1)
        pseudo_fs, _ = self.top_K_pse(x_pse_l,batch=ibatch_pse)
        # pseudo_global = global_max_pool(x_pse_l, ibatch_pse)
        # pseudo_fs = pseudo_global.unsqueeze(1)
        peptide_pseudo_intermap = self.peptide_pseudo_att(peptide_fs, pseudo_fs)
        proj = F.relu(self.dropout(self.projector(peptide_pseudo_intermap)))
        logits = self.classier(proj)
        return torch.softmax(logits, dim=1)