import os
import copy
from .featurizer import MolGraphConvFeaturizer
from rdkit import Chem
from torch_geometric.utils.subgraph import subgraph
from torch_geometric import data as DATA
import torch
import pandas as pd
from multiprocessing import Process
import torch.multiprocessing as mp
import pickle

class pMHC_DataSet(DATA.InMemoryDataset):
    def __init__(self, path, num_process=8, aug=False, test=True):
        super(pMHC_DataSet, self).__init__()
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        self.aug = aug
        self.test = test
        abpath = os.path.abspath(__file__)
        folder = os.path.dirname(abpath)
        self.hla_pseudo = pd.read_csv(os.path.join(folder, 'hlaII_pseudo_seq.csv'))
        rawdata = pd.read_csv(path, header=0)

        peptides = []
        alpha_ns = []
        beta_ns = []
        pseudos = []
        labels = []

        for _, row in rawdata.iterrows():
            peptide = row['peptide']
            alpha_n = row['alpha_n']
            beta_n = row['beta_n']
            label = row['label']
            if self.test:
                if 'label' in rawdata.columns:
                    label = row['label']
                else:
                    label = -1
            else:
                label = row['label']
            pseudo = self.get_pseudo(alpha_n, beta_n)


            if self.check(peptide):
                print("peptide:"+peptide)
                continue

            if self.check(pseudo):
                print("pseudo:"+pseudo)
                continue
            peptides.append(peptide)
            alpha_ns.append(alpha_n)
            beta_ns.append(beta_n)
            pseudos.append(pseudo)
            labels.append(label)

        peptide_set = set(peptides)
        pseudo_set = set(pseudos)
        
        save_dir = path.rstrip(path.split('/')[-1])
        filename = path.split('/')[-1].split('.')[0]

        if os.path.exists(save_dir+filename+'_'+'peptide_graph.pt'):
            self.peptide_graph = torch.load(save_dir + filename + '_' + 'peptide_graph.pt')
        else:
            self.peptide_graph = generateGraph(peptide_set, num_process)
            torch.save(self.peptide_graph, '%s%s_peptide_graph.pt' % (save_dir,filename))

        if os.path.exists(save_dir+filename+'_'+'pseudo_graph.pt'):
            self.pseudo_graph = torch.load(save_dir + filename + '_' + 'pseudo_graph.pt')
        else:
            self.pseudo_graph = generateGraph(pseudo_set, num_process)
            torch.save(self.pseudo_graph, '%s%s_pseudo_graph.pt' % (save_dir,filename))   

        samples = list(zip(peptides, alpha_ns, beta_ns, pseudos, labels))
        self.samples = samples

    def check(self, cdr3):
        i = 0
        for aa in cdr3:
            if aa not in self.AAstringList:
                break
            else:
                i += 1
        if i == len(cdr3):
            return False
        else:
            return True

    def get_pseudo(self, alpha_n, beta_n):
        alpha_pseudo_seq = self.hla_pseudo.loc[self.hla_pseudo['hla_name_1'] == alpha_n, "pseudo_seq"].iloc[0]
        beta_pseudo_seq = self.hla_pseudo.loc[self.hla_pseudo['hla_name_1'] == beta_n, "pseudo_seq"].iloc[0]
        return alpha_pseudo_seq+beta_pseudo_seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        peptide, alpha_n, beta_n, pseudo, label = self.samples[idx]
        if self.aug:
            peptide_graph = self.augmentation(self.peptide_graph[peptide])
            pseudo_graph = self.augmentation(self.pseudo_graph[pseudo])
        else:
            peptide_graph = copy.deepcopy(self.peptide_graph[peptide])
            pseudo_graph = copy.deepcopy(self.pseudo_graph[pseudo])
        return (idx, peptide, alpha_n, beta_n, label, peptide_graph, pseudo_graph)
    
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
    alpha_ns = [item[2] for item in batch]
    beta_ns = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    peptide_graphs = [item[5] for item in batch]
    pseudo_graphs = [item[6] for item in batch]
    return idxs, peptides, alpha_ns, beta_ns, torch.LongTensor(labels), peptide_graphs, pseudo_graphs

def generateGraph(seqs,threading_num):
    seq_set = set(seqs)
    seq_graph = {}
    threading_num = min(threading_num, len(seq_set))
    seq_manager = mp.Manager()
    seq_queue = seq_manager.list([])
    processes = []
    chunked = chunk_set(seq_set,threading_num)
    for i in range(threading_num):
        process = Process(target=generateGraph_subprocess, args=(chunked[i], seq_queue))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    for graph_dict in seq_queue:
        seq_graph.update(pickle.loads(graph_dict))
    return seq_graph

def generateGraph_subprocess(seqs, queue):
    manager = mp.Manager()
    graphs={}
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    for i,seq in enumerate(seqs):
        seq_chem = Chem.MolFromSequence(seq)
        seq_feature = featurizer._featurize(seq_chem)
        feature, edge_index, edge_feature = seq_feature.node_features, seq_feature.edge_index, seq_feature.edge_features
        GCNData = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))
        graphs[seq]=GCNData
    graph_serialized = pickle.dumps(graphs)
    queue.append(graph_serialized)

def chunk_set(seq_set, n):
    seq_list = list(seq_set)
    avg_chunk_size = len(seq_list) // n
    remainder = len(seq_list) % n

    chunked = []
    start = 0
    for i in range(n):
        if i!=n-1:
            end = start + avg_chunk_size
            chunked.append(seq_list[start:end])
            start = end
        else:
            chunked.append(seq_list[start:])
    return chunked