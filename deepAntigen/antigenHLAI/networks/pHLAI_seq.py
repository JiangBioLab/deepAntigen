import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.dense.linear as pyg_linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
import torch_scatter
from .top_k_pooling_seq import TopKPooling

class TGCN(MessagePassing):
    def __init__(self, hidden_channels, aggr='add'):
        super(TGCN, self).__init__(aggr=aggr)
        self.message_w = pyg_linear.Linear(hidden_channels+11, hidden_channels, weight_initializer='kaiming_uniform')
        self.update_w = pyg_linear.Linear(2*hidden_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.GRU_x = nn.GRUCell(hidden_channels, hidden_channels)

    def forward(self, x_0, edge_index, edge_attr, ibatch):
        x_u = self.propagate(edge_index, x=x_0, edge_attr=edge_attr, size=None)
        x_out = self.GRU_x(x_0, x_u)
        return x_out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        T = self.message_w(torch.cat((x_j, edge_attr), dim=1))
        return F.leaky_relu(T, 0.1)

    def update(self, inputs: Tensor, x) -> Tensor:
        output = self.update_w(torch.cat((inputs, x), dim=1))
        return F.leaky_relu(output, 0.1)
        
class Encoder(MessagePassing):
    def __init__(self, in_channels, hidden_channels, depth, k, aggr='add'):
        super(Encoder, self).__init__(aggr=aggr)
        self.init_w = pyg_linear.Linear(in_channels, hidden_channels, weight_initializer='kaiming_uniform')
        self.GCN_Depth = depth
        self.gcn = nn.ModuleList([TGCN(hidden_channels) for i in range(self.GCN_Depth)])
        self.top_K_pooling = nn.ModuleList([TopKPooling(hidden_channels, ratio=k) for i in range(self.GCN_Depth)])
        self.bn_x = nn.ModuleList([BatchNorm(hidden_channels) for i in range(self.GCN_Depth)])

    def forward(self, graphs):
        x, edge_index, edge_attr, ibatch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        x_l = F.leaky_relu(self.init_w(x), 0.1)
        for i in range(self.GCN_Depth):
            x_l = self.gcn[i](x_l, edge_index, edge_attr, ibatch)
            x_l = self.bn_x[i](x_l)
            if i==self.GCN_Depth-1:
                fs, perm = self.top_K_pooling[i](x_l,batch=ibatch)
        return fs

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()
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
        self.peptide_encoder = Encoder(25, args['hidden_size'], args['depth'], args['k'])
        self.pseudo_encoder = Encoder(25, args['hidden_size'], args['depth'], args['k'])
        self.peptide_pseudo_att = MultiHeadAttention(args['hidden_size'], args['heads'])
        self.dropout = nn.Dropout(p=0.2)
        self.projector = pyg_linear.Linear(args['hidden_size'], int(0.5*args['hidden_size']), weight_initializer='kaiming_uniform')
        self.classier = pyg_linear.Linear(int(0.5*args['hidden_size']), 2, weight_initializer='kaiming_uniform')

    def forward(self, peptide_graphs, pseudo_graphs):
        peptide_fs = self.peptide_encoder(peptide_graphs)
        pseudo_fs = self.pseudo_encoder(pseudo_graphs)
        peptide_pseudo_intermap = self.peptide_pseudo_att(peptide_fs, pseudo_fs)
        proj = F.relu(self.dropout(self.projector(peptide_pseudo_intermap)))
        logits = self.classier(proj)
        return torch.softmax(logits, dim=1)