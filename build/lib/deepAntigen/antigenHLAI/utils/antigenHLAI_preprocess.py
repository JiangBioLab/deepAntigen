import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import pickle
import math
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold

def split_data(filepath, fold_num):
    df = pd.read_csv(filepath,header=0)
    save_dir = filepath.rstrip(filepath.split('/')[-1])+'k_fold_dataset/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cv_split = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=666)
    for fold_i, (train_index, val_index) in enumerate(cv_split.split(X=df, y=df['label'])):
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        train_df.to_csv(f'{save_dir}train_fold{fold_i+1}.csv',index=False)
        val_df.to_csv(f'{save_dir}val_fold{fold_i+1}.csv',index=False)
    print('Splited datasets have been saved to'+save_dir)

def process_pdb(pdb_dir, meta_file):
    p_path = os.path.abspath(os.path.join(pdb_dir, ".."))
    save_dir = os.path.join(p_path,'pdb_Extracted')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    summary = pd.read_csv(meta_file,header=0)
    pdbs = list(summary['pdbid'])
    pseudo_seq_pos=[7,9,24,45,59,62,63,66,67,69,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
    for pdb in pdbs:
        splited = pdb.split('_')
        print(splited[0])
        antigen_chain_id = splited[1]
        mhc_alpha_chain_id = splited[2]
        alpha_seq = summary.loc[summary['pdbid']==pdb,'alpha_seq'].iloc[0]
        alpha_cut = summary.loc[summary['pdbid']==pdb,'alpha_cut'].iloc[0]

        pdb_file = os.path.join(pdb_dir, splited[0]+'.pdb')
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        parser = PDBParser()
        structure = parser.get_structure("PDB", pdb_file)
        mhc_connect = {}
        peptide_connect = {}
        peptide_flagN = True
        mhc_flagN = True
        count_peptide_atom = -1
        count_mhc_atom = -1
        for model in structure:
            for chain in model:
                if chain.id == antigen_chain_id:
                    pep_file = os.path.join(save_dir,splited[0]+'_peptide.pdb')
                    pf = open(pep_file, 'w')
                    for residue in chain:
                        for atom in residue:
                            if atom.get_name().startswith('H'):
                                continue
                            record = ''
                            atom_serial = atom.get_serial_number()
                            for line in lines:
                                serial = line[6:11].strip()
                                if serial==str(atom_serial):
                                    record = line[0:5]
                                    break
                            if not record.startswith('ATOM'):
                                continue
                            count_peptide_atom+=1
                            atom_name = atom.get_id()
                            if atom_name=='N':
                                if peptide_flagN:
                                    peptide_flagN=False
                                else:
                                    peptide_connect[lastid]=count_peptide_atom
                                    lastid = count_peptide_atom
                            if atom_name=='C':
                               lastid = count_peptide_atom
                            for line in lines:
                                if line.startswith("ATOM"):
                                    serial = line[6:11].strip()
                                    if serial==str(atom_serial):
                                        pf.write(line)
                                        break
                    pf.close()
                elif chain.id == mhc_alpha_chain_id:
                    alpha_X = list(alpha_seq).count('X')
                    alpha_add_index = alpha_cut-alpha_X
                    alpha_pos = pseudo_seq_pos+alpha_add_index
                    index = 0 
                    mhc_file = os.path.join(save_dir,splited[0]+'_mhc.pdb')
                    mf = open(mhc_file, 'w')
                    for residue in chain:
                        index+=1
                        if index in alpha_pos:
                            for atom in residue:
                                if atom.get_name().startswith('H'):
                                    continue
                                record = ''
                                atom_serial = atom.get_serial_number()
                                for line in lines:
                                    serial = line[6:11].strip()
                                    if serial==str(atom_serial):
                                        record = line[0:5]
                                        break
                                if not record.startswith('ATOM'):
                                    continue
                                count_mhc_atom+=1
                                atom_name = atom.get_id()
                                if atom_name=='N':
                                    if mhc_flagN:
                                        mhc_flagN=False
                                    else:
                                        mhc_connect[lastid]=count_mhc_atom
                                        lastid = count_mhc_atom
                                if atom_name=='C':
                                   lastid = count_mhc_atom
                                
                                for line in lines:
                                    if line.startswith("ATOM"):
                                        serial = line[6:11].strip()
                                        if serial==str(atom_serial):
                                            mf.write(line)
                                            break
                        else:
                            continue
                    mf.close()
        with open(os.path.join(save_dir,splited[0]+'_pep.pkl'),'wb') as tf:
            pickle.dump(peptide_connect,tf)
        with open(os.path.join(save_dir,splited[0]+'_mhc.pkl'),'wb') as tf:
            pickle.dump(mhc_connect,tf)
    print('Processed pdb files have been saved to'+save_dir)

def check_impossible_connection(molecule):
    new_molecule = Chem.RWMol(molecule)
    for atom in molecule.GetAtoms():
        for neighbor_atom in atom.GetNeighbors():
            neighbor_residue_id = neighbor_atom.GetPDBResidueInfo().GetResidueNumber()
            current_residue_id = atom.GetPDBResidueInfo().GetResidueNumber()
            if neighbor_residue_id != current_residue_id:
                new_molecule.RemoveBond(atom.GetIdx(), neighbor_atom.GetIdx())
    chem = new_molecule.GetMol()
    return chem

def add_CON(con, molecule):
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

def calculate_distance(pdb_extracted_dir):
    p_path = os.path.abspath(os.path.join(pdb_extracted_dir, ".."))
    save_dir = os.path.join(p_path,'distance_matrix')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = os.listdir(pdb_extracted_dir)
    for file in files:
        if file.endswith('peptide.pdb'):
            pdbid = file.split('_')[0]
            print(pdbid)
            pep_file = os.path.join(pdb_extracted_dir,pdbid+'_peptide.pdb')
            pep_con = os.path.join(pdb_extracted_dir,pdbid+'_pep.pkl')
            mhc_file = os.path.join(pdb_extracted_dir,pdbid+'_mhc.pdb')
            mhc_con = os.path.join(pdb_extracted_dir,pdbid+'_mhc.pkl')
            peptide_chem = Chem.MolFromPDBFile(pep_file)
            peptide_chem = check_impossible_connection(peptide_chem)
            peptide_chem = add_CON(pep_con, peptide_chem)
            mhc_chem = Chem.MolFromPDBFile(mhc_file)
            mhc_chem = check_impossible_connection(mhc_chem)
            mhc_chem = add_CON(mhc_con, mhc_chem)
            peptide_atoms = peptide_chem.GetAtoms()
            mhc_atoms = mhc_chem.GetAtoms()
            peptide_conformer = peptide_chem.GetConformer()
            peptide_atom_positions = peptide_conformer.GetPositions()
            mhc_conformer = mhc_chem.GetConformer()
            mhc_atom_positions = mhc_conformer.GetPositions()
            dist = np.zeros((len(peptide_atoms), len(mhc_atoms)))
            for i in range(len(peptide_atoms)):
                p_atom_coord = peptide_atom_positions[i]
                for j in range(len(mhc_atoms)):
                    m_atom_coord = mhc_atom_positions[j]
                    d = math.sqrt(np.sum(np.power(p_atom_coord-m_atom_coord, 2)))
                    if d>30:
                        dist[i][j] = 30
                    else:
                        dist[i][j] = d
            save_file = os.path.join(save_dir,pdbid+'.npy')
            np.save(save_file, dist)
    print('Distance matrixs have been saved to '+save_dir)