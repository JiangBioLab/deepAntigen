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

aa_dict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}

def aa_3to1(aa_3):
    aa_1 = aa_dict.get(aa_3)
    if aa_1 is None:
        aa_1 = 'X'
    return aa_1

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

def get_pseudo_absolute(alpha_n, alpha_seq, alpha_chain, alpha_site, alpha_cut, beta_n, beta_seq, beta_chain, beta_site,beta_cut, lines, count_mhc_atom, mhc_flagN, mhc_connect,save_dir,pdbid):
    alpha_pseudo_site = [9,11,22,24,31,52,53,58,59,61,65,66,68,72,73]
    beta_pseudo_site = [9,11,13,26,28,30,47,57,67,70,71,74,77,78,81,85,86,89,90]
    
    alpha_X = list(alpha_seq).count('X')
    beta_X = list(beta_seq).count('X')
    
    alpha_add_index = alpha_site-alpha_X+alpha_cut
    beta_add_index = beta_site-beta_X+beta_cut

    alpha_pos = alpha_pseudo_site+alpha_add_index
    beta_pos = beta_pseudo_site+beta_add_index
    mhc_file = os.path.join(save_dir,pdbid+'_mhc.pdb')
    mf = open(mhc_file, 'w')
    for position in alpha_pos:
        index=0
        
        for residue in alpha_chain:
            s_aa =aa_3to1(residue.get_resname())
            if s_aa=='X':
                continue
            index+=1
            if index == position:
                for atom in residue:
                    atom_name = atom.get_id()
                    if atom.get_name().startswith('H'):
                        continue
                    record = ''
                    atom_serial = atom.get_serial_number()
                    for line in lines:
                        serial = line[6:11].strip(' ')
                        a_n = line[13:16].strip(' ')
                        if serial==str(atom_serial) and a_n==str(atom_name):
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
                            serial = line[6:11].strip(' ')
                            if serial==str(atom_serial):
                                mf.write(line)
                                break
            else:
                continue
    for position in beta_pos:
        index=0
        
        for residue in beta_chain:
            s_aa =aa_3to1(residue.get_resname())
            if s_aa=='X':
                continue
            index+=1
            if index == position:
                for atom in residue:
                    atom_name = atom.get_id()
                    if atom.get_name().startswith('H'):
                        continue
                    record = ''
                    atom_serial = atom.get_serial_number()
                    for line in lines:
                        serial = line[6:11].strip(' ')
                        a_n = line[13:16].strip(' ')
                        if serial==str(atom_serial) and a_n==str(atom_name):
                            record = line[0:5]
                            break
                    if not record.startswith('ATOM'):
                        continue
                    count_mhc_atom+=1
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
                            serial = line[6:11].strip(' ')
                            if serial==str(atom_serial):
                                mf.write(line)
                                break
            else:
                continue
    mf.close()

def get_pseudo_relative(alpha_n, alpha_seq, alpha_chain, alpha_site, alpha_cut, beta_n, beta_seq, beta_chain, beta_site,beta_cut, lines, count_mhc_atom, mhc_flagN, mhc_connect, save_dir,pdbid):
    alpha_pseudo_site = [9,11,22,24,31,52,53,58,59,61,65,66,68,72,73]
    beta_pseudo_site = [9,11,13,26,28,30,47,57,67,70,71,74,77,78,81,85,86,89,90]
    alpha_X = list(alpha_seq).count('X')
    beta_X = list(beta_seq).count('X')
    alpha_add_index = alpha_site-alpha_X+alpha_cut
    beta_add_index = beta_site-beta_X+beta_cut

    alpha_pos = alpha_pseudo_site+alpha_add_index
    beta_pos = beta_pseudo_site+beta_add_index

    count_residue = -1
    mhc_file = os.path.join(save_dir,pdbid+'_mhc.pdb')
    mf = open(mhc_file, 'w')
    for position in alpha_pos:
        index=0
        for residue in alpha_chain:
            s_aa =aa_3to1(residue.get_resname())
            if s_aa=='X':
                continue
            index+=1
            if index == position:
                count_residue+=1
                for atom in residue:
                    atom_name = atom.get_id()
                    if atom.get_name().startswith('H'):
                        continue
                    record = ''
                    atom_serial = atom.get_serial_number()
                    for line in lines:
                        serial = line[6:11].strip(' ')
                        a_n = line[13:16].strip(' ')
                        if serial==str(atom_serial) and a_n==str(atom_name):
                            record = line[0:5]
                            break
                    if not record.startswith('ATOM'):
                        continue
                    count_mhc_atom+=1
                    
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
                                new_serial = "{:5}".format(count_mhc_atom+1)
                                new_res_num = "{:3}".format(count_residue+1)
                                new_line = line[:6]+new_serial+line[11:]
                                new_line = new_line[:21]+'A'+new_line[22:]
                                new_line = new_line[:23]+new_res_num+line[26:]
                                mf.write(new_line)
                                break
            else:
                continue
    for position in beta_pos:
        index=0
        for residue in beta_chain:
            s_aa =aa_3to1(residue.get_resname())
            if s_aa=='X':
                continue
            index+=1
            if index == position:
                count_residue+=1
                for atom in residue:
                    atom_name = atom.get_id()
                    if atom.get_name().startswith('H'):
                        continue
                    record = ''
                    atom_serial = atom.get_serial_number()
                    for line in lines:
                        serial = line[6:11].strip(' ')
                        a_n = line[13:16].strip(' ')
                        if serial==str(atom_serial) and a_n==str(atom_name):
                            record = line[0:5]
                            break
                    if not record.startswith('ATOM'):
                        continue
                    count_mhc_atom+=1
                    
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
                                new_serial = "{:5}".format(count_mhc_atom+1)
                                new_res_num = "{:3}".format(count_residue+1)
                                new_line = line[:6]+new_serial+line[11:]
                                new_line = new_line[:21]+'A'+new_line[22:]
                                new_line = new_line[:23]+new_res_num+line[26:]
                                mf.write(new_line)
                                break
            else:
                continue
    mf.close()

def process_pdb_absolute(pdb_dir, meta_file):
    p_path = os.path.abspath(os.path.join(pdb_dir, ".."))
    save_dir = os.path.join(p_path,'pdb_Extracted_absolute')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    summary = pd.read_csv(meta_file,header=0)
    pdbs = list(summary['pdbid'])
    for pdb in pdbs:
        splited = pdb.split('_')
        print(splited[0])
        alpha_n = summary.loc[summary['pdbid']==pdb,'alpha_n'].iloc[0]
        alpha_seq = summary.loc[summary['pdbid']==pdb,'alpha_seq'].iloc[0]
        alpha_site = summary.loc[summary['pdbid']==pdb,'alpha_site'].iloc[0]
        alpha_cut = summary.loc[summary['pdbid']==pdb,'alpha_cut'].iloc[0]
        beta_n = summary.loc[summary['pdbid']==pdb,'beta_n'].iloc[0]
        beta_seq = summary.loc[summary['pdbid']==pdb,'beta_seq'].iloc[0]
        beta_site = summary.loc[summary['pdbid']==pdb,'beta_site'].iloc[0]
        beta_cut = summary.loc[summary['pdbid']==pdb,'beta_cut'].iloc[0]

        antigen_chain_id = splited[1]
        mhc_alpha_chain_id = splited[2]
        mhc_beta_chain_id = splited[3]

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
                    mhc_alpha_chain = chain
                elif chain.id == mhc_beta_chain_id:
                    mhc_beta_chain = chain
            get_pseudo_absolute(alpha_n, alpha_seq, mhc_alpha_chain, alpha_site,alpha_cut, beta_n, beta_seq, mhc_beta_chain, beta_site,beta_cut, lines, count_mhc_atom, mhc_flagN, mhc_connect, save_dir, splited[0])
        with open(os.path.join(save_dir,splited[0]+'_pep.pkl'),'wb') as tf:
            pickle.dump(peptide_connect,tf)
        with open(os.path.join(save_dir,splited[0]+'_mhc.pkl'),'wb') as tf:
            pickle.dump(mhc_connect,tf)
    print('Processed pdb files have been saved to'+save_dir)

def process_pdb_relative(pdb_dir, meta_file):
    p_path = os.path.abspath(os.path.join(pdb_dir, ".."))
    save_dir = os.path.join(p_path,'pdb_Extracted_relative')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    summary = pd.read_csv(meta_file,header=0)
    pdbs = list(summary['pdbid'])
    for pdb in pdbs:
        splited = pdb.split('_')
        print(splited[0])
        alpha_n = summary.loc[summary['pdbid']==pdb,'alpha_n'].iloc[0]
        alpha_seq = summary.loc[summary['pdbid']==pdb,'alpha_seq'].iloc[0]
        alpha_site = summary.loc[summary['pdbid']==pdb,'alpha_site'].iloc[0]
        alpha_cut = summary.loc[summary['pdbid']==pdb,'alpha_cut'].iloc[0]
        beta_n = summary.loc[summary['pdbid']==pdb,'beta_n'].iloc[0]
        beta_seq = summary.loc[summary['pdbid']==pdb,'beta_seq'].iloc[0]
        beta_site = summary.loc[summary['pdbid']==pdb,'beta_site'].iloc[0]
        beta_cut = summary.loc[summary['pdbid']==pdb,'beta_cut'].iloc[0]

        antigen_chain_id = splited[1]
        mhc_alpha_chain_id = splited[2]
        mhc_beta_chain_id = splited[3]

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
                    mhc_alpha_chain = chain
                elif chain.id == mhc_beta_chain_id:
                    mhc_beta_chain = chain
            get_pseudo_relative(alpha_n, alpha_seq, mhc_alpha_chain, alpha_site,alpha_cut, beta_n, beta_seq, mhc_beta_chain, beta_site,beta_cut, lines, count_mhc_atom, mhc_flagN, mhc_connect, save_dir, splited[0])
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