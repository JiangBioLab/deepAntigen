import os
import copy
from .featurizer import MolGraphConvFeaturizer
from rdkit import Chem
from torch_geometric import data as DATA
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle

class pTCR_DataSet(DATA.InMemoryDataset):
    def __init__(self, path):
        super(pTCR_DataSet).__init__()
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        meta_data = pd.read_csv(path, header=0)
        pdbinfos = meta_data['pdbid']
        self.pdbids = [pdbinfo.split('_')[0] for pdbinfo in pdbinfos]
        data_dir = path.rstrip(path.split('/')[-1])
        self.data_dir = data_dir
        self.peptide_chems = {}
        self.cdr3_chems = {}
        for pdbid in self.pdbids:
            pep_file = data_dir+'pdb_Extracted/'+pdbid+'_peptide.pdb'
            pep_con = data_dir+'pdb_Extracted/'+pdbid+'_pep.pkl'
            peptide_chem = Chem.MolFromPDBFile(pep_file)
            peptide_chem = self.check_impossible_connection(peptide_chem)
            peptide_chem = self.add_CON(pep_con, peptide_chem)
            self.peptide_chems[pdbid]=peptide_chem
            cdr3_file = data_dir+'pdb_Extracted/'+pdbid+'_cdr3.pdb'
            cdr3_con = data_dir+'pdb_Extracted/'+pdbid+'_cdr3.pkl'
            cdr3_chem = Chem.MolFromPDBFile(cdr3_file)
            cdr3_chem = self.check_impossible_connection(cdr3_chem)
            cdr3_chem = self.add_CON(cdr3_con, cdr3_chem)
            self.cdr3_chems[pdbid]=cdr3_chem
        if os.path.exists(data_dir+'PDB_peptide_graph.pt'):
            self.peptide_graph = torch.load(data_dir+'PDB_peptide_graph.pt')
        else:
            self.peptide_graph = {}
            featurizer = MolGraphConvFeaturizer(use_edges=True)
            for pdbid in self.pdbids:
                peptide_chem=self.peptide_chems[pdbid]
                peptide_feature = featurizer._featurize(peptide_chem)
                feature, edge_index, edge_feature = peptide_feature.node_features, peptide_feature.edge_index, peptide_feature.edge_features
                peptide_graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))
                self.peptide_graph[pdbid]=peptide_graph
        torch.save(self.peptide_graph, '%sPDB_peptide_graph.pt' % (data_dir))
        if os.path.exists(data_dir+'PDB_cdr3_graph.pt'):
            self.cdr3_graph = torch.load(data_dir+'PDB_cdr3_graph.pt')
        else:
            self.cdr3_graph = {}
            featurizer = MolGraphConvFeaturizer(use_edges=True)
            for pdbid in self.pdbids:
                cdr3_chem=self.cdr3_chems[pdbid]
                cdr3_feature = featurizer._featurize(cdr3_chem)
                feature, edge_index, edge_feature = cdr3_feature.node_features, cdr3_feature.edge_index, cdr3_feature.edge_features
                cdr3_graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))
                self.cdr3_graph[pdbid]=cdr3_graph  
        torch.save(self.cdr3_graph, '%sPDB_cdr3_graph.pt' % (data_dir))
    def __len__(self):
        return len(self.pdbids)

    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]
        distance_matrix = np.load(self.data_dir+'distance_matrix/'+pdbid+'.npy')
        distance_matrix = np.where(distance_matrix>30, 30, distance_matrix)
        peptide_chem = self.peptide_chems[pdbid]
        peptide_graph = self.peptide_graph[pdbid]
        cdr3_chem = self.cdr3_chems[pdbid]
        cdr3_graph = self.cdr3_graph[pdbid]
        return (pdbid, peptide_chem, peptide_graph, cdr3_chem, cdr3_graph, distance_matrix)

    def check_impossible_connection(self, molecule):
        new_molecule = Chem.RWMol(molecule)
        for atom in molecule.GetAtoms():
            for neighbor_atom in atom.GetNeighbors():
                neighbor_residue_id = neighbor_atom.GetPDBResidueInfo().GetResidueNumber()
                current_residue_id = atom.GetPDBResidueInfo().GetResidueNumber()
                if neighbor_residue_id != current_residue_id:
                    new_molecule.RemoveBond(atom.GetIdx(), neighbor_atom.GetIdx())
        chem = new_molecule.GetMol()
        return chem

    def add_CON(self, con, molecule):
        editable_mol = Chem.EditableMol(molecule)
        with open(con,'rb') as tf:
            connect = pickle.load(tf)
        for atomid1, atomid2 in connect.items():
            atom1 = molecule.GetAtomWithIdx(atomid1)
            atom2 = molecule.GetAtomWithIdx(atomid2)
            bond = molecule.GetBondBetweenAtoms(atomid1, atomid2)
            if bond is not None:
                pass
            else:
                editable_mol.AddBond(atomid1, atomid2, order=Chem.rdchem.BondType.SINGLE)
        new_molecule = editable_mol.GetMol()
        new_molecule = Chem.RemoveHs(new_molecule)
        return new_molecule

def collate(batch):
    idxs = [item[0] for item in batch]
    peptide_chems= [item[1] for item in batch]
    peptide_graphs = [item[2] for item in batch]
    cdr3_chems = [item[3] for item in batch]
    cdr3_graphs = [item[4] for item in batch]
    distance_matrixs = [item[5] for item in batch]
    return idxs, peptide_chems, peptide_graphs, cdr3_chems, cdr3_graphs, distance_matrixs