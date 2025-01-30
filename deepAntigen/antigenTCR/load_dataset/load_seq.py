import os
import copy
from .featurizer import MolGraphConvFeaturizer
from rdkit import Chem
from torch_geometric.utils.subgraph import subgraph
from torch_geometric import data as DATA
import torch
import pandas as pd
import pickle

class pTCR_DataSet(DATA.InMemoryDataset):
    def __init__(self, path, aug=False, test=True):
        super(pTCR_DataSet, self).__init__()
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        self.aug = aug
        self.test = test
        self.rawdata = pd.read_csv(path, header=0)
        pep_counts = self.rawdata['peptide'].value_counts()
        tcr_counts = self.rawdata['binding_TCR'].value_counts()

        self.high_freq_pep = list(pep_counts[pep_counts > 10].index)
        self.high_freq_tcr = list(tcr_counts[tcr_counts > 10].index)
        self.peptide_graph = {}
        self.cdr3_graph = {}

    def check(self, seq):
        i = 0
        for aa in seq:
            if aa not in self.AAstringList:
                break
            else:
                i += 1
        if i == len(seq):
            return False
        else:
            return True

    def generateGraph(self, seq):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        seq_chem = Chem.MolFromSequence(seq)
        seq_feature = featurizer._featurize(seq_chem)
        feature, edge_index, edge_feature = seq_feature.node_features, seq_feature.edge_index, seq_feature.edge_features
        graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))
        return graph

    def __len__(self):
        return len(self.rawdata)

    def __getitem__(self, idx):
        row = self.rawdata.loc[idx]
        peptide = row['peptide']
        cdr3 = row['binding_TCR']
        if self.test:
            if 'label' in self.rawdata.columns:
                label = row['label']
            else:
                label = -1
        else:
            label = row['label']
        if self.check(peptide):
            print("peptide:"+peptide+' is skipped.')
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
        if self.check(cdr3):
            print("cdr3:"+cdr3+' is skipped.')
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
        if self.test:
            if peptide in self.high_freq_pep:
                if peptide in self.peptide_graph:
                    peptide_graph =copy.deepcopy(self.peptide_graph[peptide])
                else:
                    peptide_graph = self.generateGraph(peptide)
                    self.peptide_graph[peptide]=peptide_graph
            else:
                peptide_graph = self.generateGraph(peptide)
            if cdr3 in self.high_freq_tcr:
                if cdr3 in self.cdr3_graph:
                    cdr3_graph = copy.deepcopy(self.cdr3_graph[cdr3])
                else:
                    cdr3_graph = self.generateGraph(cdr3)
                    self.cdr3_graph[cdr3]=cdr3_graph
            else:
                cdr3_graph = self.generateGraph(cdr3)
        else:
            if peptide in self.peptide_graph:
                peptide_graph =copy.deepcopy(self.peptide_graph[peptide])
            else:
                peptide_graph = self.generateGraph(peptide)
                self.peptide_graph[peptide]=peptide_graph
            if cdr3 in self.cdr3_graph:
                cdr3_graph = copy.deepcopy(self.cdr3_graph[cdr3])
            else:
                cdr3_graph = self.generateGraph(cdr3)
                self.cdr3_graph[cdr3]=cdr3_graph
        if self.aug:
            peptide_graph = self.augmentation(peptide_graph)
            cdr3_graph = self.augmentation(cdr3_graph)
        peptide_graph = pickle.dumps(peptide_graph)
        cdr3_graph = pickle.dumps(cdr3_graph)

        return (idx, peptide, cdr3, label, peptide_graph, cdr3_graph)

    def augmentation(self,graph):
        aug_graph = copy.deepcopy(graph)
        prob = torch.rand(aug_graph.num_nodes)
        mask = prob > 0.05
        edge_index, edge_attr = subgraph(mask, aug_graph.edge_index, aug_graph.edge_attr, relabel_nodes=True)
        aug_graph.x = aug_graph.x[mask, :]
        aug_graph.edge_index = edge_index
        aug_graph.edge_attr = edge_attr
        return aug_graph
    
def collate(batch):
    idxs = [item[0] for item in batch]
    peptides = [item[1] for item in batch]
    cdr3s = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    peptide_graphs = [pickle.loads(item[4]) for item in batch]
    cdr3_graphs = [pickle.loads(item[5]) for item in batch]
    return idxs, peptides, cdr3s, torch.LongTensor(labels), peptide_graphs, cdr3_graphs