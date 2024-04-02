from typing import Callable, Optional, Union
import math
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot
import torch_geometric.nn.dense.linear as pyg_linear
import torch.nn.functional as F
from itertools import accumulate

def generate_O_N(chems,max_num_nodes):
    index_parallel = []
    index = []
    num_nodes = []
    cum_atom_num=0
    for i,chem in enumerate(chems):
        # print([atom.GetSymbol() for atom in chem.GetAtoms()])
        atom_index_parallel = [idx+i*max_num_nodes for idx,atom in enumerate(chem.GetAtoms()) if atom.GetSymbol() in ['N','O']]
        index_parallel.extend(atom_index_parallel)
        atom_index = [idx+cum_atom_num for idx,atom in enumerate(chem.GetAtoms()) if atom.GetSymbol() in ['N','O']]
        num_nodes.append(len(atom_index))
        index.extend(atom_index)
        # print(len(chem.GetAtoms()))
        cum_atom_num+=len(chem.GetAtoms())
    return index_parallel,num_nodes,index
    
def topk(x, ratio, chems, batch):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    rich_num_nodes = torch.Tensor(batch_size*[max_num_nodes]).to(x.device)
    cum_rich_num_nodes=torch.cat(
        [rich_num_nodes.new_zeros(1),
         rich_num_nodes.cumsum(dim=0)[:-1]], dim=0)
    void_num_nodes = max_num_nodes-num_nodes
    cum_void_num_nodes=torch.cat(
        [void_num_nodes.new_zeros(1),
         void_num_nodes.cumsum(dim=0)[:-1]], dim=0)
    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ),
                         torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_rich_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    on_index_parallel, on_num, on_index = generate_O_N(chems, max_num_nodes)
    on_index_parallel = torch.LongTensor(on_index_parallel).to(x.device)

    on_index = torch.LongTensor(on_index).to(x.device)
    offset=list(accumulate(on_num))
    offset=[0]+offset[:-1]

    indices=torch.where(perm == on_index_parallel[:, None])[1] 
    indices,_=indices.sort()
    k = num_nodes.new_full((num_nodes.size(0), ), ratio)
    on_num = torch.Tensor(on_num).to(x.device)
    offset = torch.Tensor(offset).to(x.device)
    k = torch.min(k, on_num)
    pre_mask=[
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        offset[i] for i in range(batch_size)
    ]
    pre_mask = torch.cat(pre_mask, dim=0)
    mask=indices[pre_mask.long()]

    perm = perm.view(batch_size,max_num_nodes)
    perm = perm - cum_void_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    perm = perm[mask]
    
    return perm.long(),on_index

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, in_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2).float() * (-math.log(10000.0) / in_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, ibatch):
        num_nodes = scatter_add(torch.ones_like(ibatch), ibatch, dim=0)
        index = torch.cat([torch.arange(num) for num in num_nodes])
        x = x.unsqueeze(1)
        x = x + self.pe[index, :]
        x = self.dropout(x)
        x = x.squeeze(1)
        return x

class TopKPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: int = 1, nonlinearity: Callable = torch.tanh):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        
        self.pos_emb = PositionalEncoding(in_channels)

        self.layer_atom = pyg_linear.Linear(in_channels, 256, weight_initializer='kaiming_uniform')
        self.layer_atom2 = pyg_linear.Linear(256, 128, weight_initializer='kaiming_uniform')
        self.layer_atom3 = pyg_linear.Linear(128, 64, weight_initializer='kaiming_uniform')
        self.weight_atom = nn.Parameter(torch.Tensor(1, 64))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_atom)


    def forward(self, x, chems, batch):
        x = self.pos_emb(x, batch)
        xx = x.unsqueeze(-1) if x.dim() == 1 else x
        xx_t = F.leaky_relu(self.layer_atom(xx),0.1)
        xx_t = F.leaky_relu(self.layer_atom2(xx_t),0.1)
        xx_t = F.leaky_relu(self.layer_atom3(xx_t),0.1)
        score = (xx_t * self.weight_atom).sum(dim=-1)
        score = self.nonlinearity(score / self.weight_atom.norm(p=2, dim=-1))
        perm, on_index = topk(score, self.ratio, chems, batch)
        x_top = xx[perm] * score[perm].view(-1, 1)
        bz = batch.max().item()+1
        x_top = x_top.view(bz, self.ratio, -1)
        return x_top, perm, score[on_index], on_index

    def __repr__(self) -> str:
        ratio = f'ratio={self.ratio}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio},'
                f'multiplier={self.multiplier})')
