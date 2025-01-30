import os
import copy
from .featurizer import MolGraphConvFeaturizer
from rdkit import Chem
from torch_geometric.utils.subgraph import subgraph
from torch_geometric import data as DATA
import torch
import pandas as pd
import pickle

class pMHC_DataSet(DATA.InMemoryDataset):
    def __init__(self, path, num_process=8, aug=False, test=True):
        super(pMHC_DataSet, self).__init__()
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        self.aug = aug
        self.test = test
        abpath = os.path.abspath(__file__)
        folder = os.path.dirname(abpath)
        self.hla_pseudo = pd.read_csv(os.path.join(folder, 'hlaI_pseudo_seq.csv'))
        self.rawdata = pd.read_csv(path, header=0)
        pep_counts = self.rawdata['pep'].value_counts()
        MHC_counts = self.rawdata['HLA'].value_counts()

        self.high_freq_pep = list(pep_counts[pep_counts > 10].index)
        self.high_freq_mhc = list(MHC_counts[MHC_counts > 10].index)
        self.peptide_graph = {}
        self.pseudo_graph = {}

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

    def __len__(self):
        return len(self.rawdata)

    def generateGraph(self, seq):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        seq_chem = Chem.MolFromSequence(seq)
        seq_feature = featurizer._featurize(seq_chem)
        feature, edge_index, edge_feature = seq_feature.node_features, seq_feature.edge_index, seq_feature.edge_features
        graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))
        return graph

    def __getitem__(self, idx):
        row = self.rawdata.loc[idx]
        peptide=row['pep']
        allele=row['HLA']
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
        pseudo = self.hla_pseudo.loc[self.hla_pseudo['allele']==allele, 'sequence'].iloc[0]
        if self.test:
            if peptide in self.high_freq_pep:
                if peptide in self.peptide_graph:
                    peptide_graph =copy.deepcopy(self.peptide_graph[peptide])
                else:
                    peptide_graph = self.generateGraph(peptide)
                    self.peptide_graph[peptide]=peptide_graph
            else:
                peptide_graph = self.generateGraph(peptide)
            if allele in self.high_freq_mhc:
                if pseudo in self.pseudo_graph:
                    pseudo_graph = copy.deepcopy(self.pseudo_graph[pseudo])
                else:
                    pseudo_graph = self.generateGraph(pseudo)
                    self.pseudo_graph[pseudo]=pseudo_graph
            else:
                pseudo_graph = self.generateGraph(pseudo)
        else:
            if peptide in self.peptide_graph:
                peptide_graph =copy.deepcopy(self.peptide_graph[peptide])
            else:
                peptide_graph = self.generateGraph(peptide)
                self.peptide_graph[peptide]=peptide_graph
            if pseudo in self.pseudo_graph:
                pseudo_graph = copy.deepcopy(self.pseudo_graph[pseudo])
            else:
                pseudo_graph = self.generateGraph(pseudo)
                self.pseudo_graph[pseudo]=pseudo_graph
        if self.aug:
            peptide_graph = self.augmentation(peptide_graph)
            pseudo_graph = self.augmentation(pseudo_graph)
        peptide_graph = pickle.dumps(peptide_graph)
        pseudo_graph = pickle.dumps(pseudo_graph)

        return (idx, peptide, allele, label, peptide_graph, pseudo_graph)

    def augmentation(self, graph):
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
    alleles = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    peptide_graphs = [pickle.loads(item[4]) for item in batch]
    pseudo_graphs = [pickle.loads(item[5]) for item in batch]
    return idxs, peptides, alleles, torch.LongTensor(labels), peptide_graphs, pseudo_graphs

