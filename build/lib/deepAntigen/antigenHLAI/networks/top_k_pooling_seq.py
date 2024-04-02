from typing import Callable, Optional, Union
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot


def topk(x, ratio, batch):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ),
                         torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    k = num_nodes.new_full((num_nodes.size(0), ), ratio)
    k = torch.min(k, num_nodes)

    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)
    ]
    mask = torch.cat(mask, dim=0)

    perm = perm[mask]
    return perm


class TopKPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: int = 1, nonlinearity: Callable = torch.tanh):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)


    def forward(self, x, batch):        
        xx = x.unsqueeze(-1) if x.dim() == 1 else x
        score = (xx * self.weight).sum(dim=-1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))

        perm = topk(score, self.ratio, batch)
        x_top = xx[perm] * score[perm].view(-1, 1)
        bz = batch.max().item()+1
        x_top = x_top.view(bz, self.ratio, -1)
        return x_top, perm

    def __repr__(self) -> str:
        ratio = f'ratio={self.ratio}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')
