import os
import copy
from .featurizer import MolGraphConvFeaturizer
from rdkit import Chem
from torch_geometric import data as DATA
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pickle

class pMHC_DataSet(DATA.InMemoryDataset):
    def __init__(self, path):
        super(pMHC_DataSet, self).__init__()
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        meta_data = pd.read_csv(path, header=0)
        pdbinfos = meta_data['pdbid']
        self.pdbids = [pdbinfo.split('_')[0] for pdbinfo in pdbinfos]
        data_dir = path.rstrip(path.split('/')[-1])
        self.data_dir = data_dir
        self.peptide_chems = {}
        self.mhc_chems = {}
        for pdbid in self.pdbids:
            pep_file = data_dir+'pdb_Extracted/'+pdbid+'_peptide.pdb'
            pep_con = data_dir+'pdb_Extracted/'+pdbid+'_pep.pkl'
            peptide_chem = Chem.MolFromPDBFile(pep_file)
            peptide_chem = self.check_impossible_connection(peptide_chem)
            peptide_chem = self.add_CON(pep_con, peptide_chem)
            self.peptide_chems[pdbid]=peptide_chem
            mhc_file = data_dir+'pdb_Extracted/'+pdbid+'_mhc.pdb'
            mhc_con = data_dir+'pdb_Extracted/'+pdbid+'_mhc.pkl'
            mhc_chem = Chem.MolFromPDBFile(mhc_file)
            mhc_chem = self.check_impossible_connection(mhc_chem)
            mhc_chem = self.add_CON(mhc_con, mhc_chem)
            self.mhc_chems[pdbid]=mhc_chem
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
        if os.path.exists(data_dir+'PDB_mhc_graph.pt'):
            self.mhc_graph = torch.load(data_dir+'PDB_mhc_graph.pt')
        else:
            self.mhc_graph = {}
            featurizer = MolGraphConvFeaturizer(use_edges=True)
            for pdbid in self.pdbids:
                mhc_chem=self.mhc_chems[pdbid]
                mhc_feature = featurizer._featurize(mhc_chem)
                feature, edge_index, edge_feature = mhc_feature.node_features, mhc_feature.edge_index, mhc_feature.edge_features
                mhc_graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))
                self.mhc_graph[pdbid]=mhc_graph  
        torch.save(self.mhc_graph, '%sPDB_mhc_graph.pt' % (data_dir))
        
    def __len__(self):
        return len(self.pdbids)

    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]
        distance_matrix = np.load(self.data_dir+'distance_matrix/'+pdbid+'.npy')
        distance_matrix = np.where(distance_matrix>30, 30, distance_matrix)
        peptide_chem = self.peptide_chems[pdbid]
        peptide_graph = self.peptide_graph[pdbid]
        mhc_chem = self.mhc_chems[pdbid]
        mhc_graph = self.mhc_graph[pdbid]
        return (pdbid, peptide_chem, peptide_graph, mhc_chem, mhc_graph, distance_matrix)

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
    peptide_chems = [item[1] for item in batch]
    peptide_graphs = [item[2] for item in batch]
    mhc_chems = [item[3] for item in batch]
    pseudo_graphs = [item[4] for item in batch]
    distance_matrixs = [item[5] for item in batch]
    return idxs, peptide_chems, peptide_graphs, mhc_chems, pseudo_graphs, distance_matrixs