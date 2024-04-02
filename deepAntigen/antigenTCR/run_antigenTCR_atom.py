import os
from .networks.pTCR_atom import DeepGCN
from .load_dataset.load_structure import pTCR_DataSet, collate
from .load_dataset.featurizer import MolGraphConvFeaturizer
from .utils.model_utils import *
import torch
from torch_geometric.data import Batch
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch_geometric import data as DATA
from sklearn.model_selection import KFold
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
from configparser import ConfigParser

def read_config(path):
    conf = ConfigParser()
    conf.read(path)
    args = dict(conf['settings'])
    for key,value in args.items():
        if key in ['hidden_size','depth','k','heads','batchsize','epochs', 'print_freq', 'save_freq']:
            args[key] = int(args[key])
        elif key in ['lr', 'lr_decay_rate', 'weight_decay']:
            args[key] = float(args[key])
        elif key=='cosine':
            args[key] = bool(int(args[key]))
        else:
            pass
    iterations = args['lr_decay_epochs'].split(',')
    args['lr_decay_epochs'] = list([])
    for it in iterations:
        args['lr_decay_epochs'].append(int(it))
    return args


def set_model(args, pretrain_state_dict,device):
    model = DeepGCN(args)
    if pretrain_state_dict is not None:
        model.load_state_dict(pretrain_state_dict)
    if torch.cuda.is_available():
        model = model.to(device)   
        cudnn.benchmark = True
    return model

def check(seq):
    AAstringList=list('ACDEFGHIKLMNPQRSTVWY')
    i = 0
    for aa in seq:
        if aa not in AAstringList:
            break
        else:
            i += 1
    if i == len(seq):
        return False
    else:
        return True

def test(model, peptide_chems, peptide_graphs, cdr3_chems, cdr3_graphs, device):
    with torch.no_grad():
        model.eval()
        peptide_graphs = Batch.from_data_list(peptide_graphs)
        peptide_graphs = peptide_graphs.to(device)
        cdr3_graphs = Batch.from_data_list(cdr3_graphs)
        cdr3_graphs = cdr3_graphs.to(device)
        p_perm, p_scores, p_on_indexs, c_perm, c_scores, c_on_indexs, intermap_logits = model(peptide_graphs, cdr3_graphs,peptide_chems,cdr3_chems)
        preds = torch.argmax(intermap_logits, dim=-1)
        intermap = intermap_logits[0][:,:,-1]
    return p_perm, c_perm, intermap.detach().cpu().numpy()

def Inference(file_path, model_path=''):
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    if len(model_path)==0:
        abpath = os.path.abspath(__file__)
        folder = os.path.dirname(abpath)
        state = torch.load(os.path.join(folder,'Weights','atom-level_parameters.pt'))
    else:
        state = torch.load(model_path)
    pretrain_state_dict = state['model']
    args = state['opt']
    df = pd.read_csv(file_path)

    model = set_model(args, pretrain_state_dict,device)
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    contact_maps = []
    p_atoms = []
    c_atoms = []
    for _,row in df.iterrows():
        sample_id = row['sample_id']
        peptide=row['pep_seq']
        cdr3=row['cdr3_seq']

        peptide_chem = Chem.MolFromSequence(peptide)

        peptide_feature = featurizer._featurize(peptide_chem)
        feature, edge_index, edge_feature = peptide_feature.node_features, peptide_feature.edge_index, peptide_feature.edge_features
        peptide_graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))

        cdr3_chem = Chem.MolFromSequence(cdr3)
        cdr3_feature = featurizer._featurize(cdr3_chem)
        feature, edge_index, edge_feature = cdr3_feature.node_features, cdr3_feature.edge_index, cdr3_feature.edge_features
        cdr3_graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index), edge_attr=torch.Tensor(edge_feature))

        p_perm, c_perm, intermap=test(model,[peptide_chem], [peptide_graph], [cdr3_chem], [cdr3_graph],device)

        peptide_atoms = peptide_chem.GetAtoms()
        cdr3_atoms = cdr3_chem.GetAtoms()

        pas = []
        p_residue_names = []
        p_residue_nums = []

        for i in range(len(p_perm)):
            p_atom = peptide_atoms[int(p_perm[i])]
            p_idx = p_atom.GetIdx()
            pa=p_atom.GetSymbol()
            p_residue_num = p_atom.GetPDBResidueInfo().GetResidueNumber()
            p_residue_name = p_atom.GetPDBResidueInfo().GetResidueName()
            pas.append(pa)
            p_residue_names.append(p_residue_name)
            p_residue_nums.append(str(p_residue_num))
            cas = []
            c_residue_names = []
            c_residue_nums = []
            for j in range(len(c_perm)):
                c_atom = cdr3_atoms[int(c_perm[j])]
                c_idx = c_atom.GetIdx()
                ca = c_atom.GetSymbol()
                c_residue_num = c_atom.GetPDBResidueInfo().GetResidueNumber()
                c_residue_name = c_atom.GetPDBResidueInfo().GetResidueName()
                cas.append(ca)
                c_residue_names.append(c_residue_name)
                c_residue_nums.append(str(c_residue_num))
        data=pd.DataFrame(intermap)
        ylabels = []
        for num, res_name, atom_name in zip(p_residue_nums,p_residue_names, pas):
            label = num+'\n'+res_name+'\n'+atom_name
            ylabels.append(label)
        xlabels = []
        for num, res_name,atom_name in zip(c_residue_nums,c_residue_names,cas):
            label = atom_name+'\n'+res_name+'\n'+num
            xlabels.append(label)
        plt.figure(figsize=(10,8))
        ax=sns.heatmap(data,cmap="Reds_r", cbar=True, xticklabels=xlabels, yticklabels=ylabels)
        plt.xlabel('CDR3')
        plt.ylabel('Peptide')
        plt.title('Sample:'+sample_id)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Contact probability')
        if not os.path.exists(args['output']):
            os.makedirs(args['output'])
        plt.savefig(args['output']+sample_id+'_contact.png',dpi=600)
        contact_maps.append(data)
        p_atoms.append(ylabels)
        c_atoms.append(xlabels)
    print('Prediction results have been saved to'+args['output'])
    return p_atoms, c_atoms, contact_maps

def generate_label(args, p_perm, c_perm, distance_matrixs, device):
    labels = []
    for i in range(len(distance_matrixs)):
        p_index = p_perm[i*args['k']:(i+1)*args['k']]
        c_index = c_perm[i*args['k']:(i+1)*args['k']]
        distance_matrix = distance_matrixs[i]
        temp = distance_matrix[np.ix_(p_index, c_index)]
        temp_label = np.where(temp<5,1,0)
        temp_label = np.expand_dims(temp_label,axis=0)
        labels.append(temp_label)
    label = np.concatenate(labels, axis=0)
    return torch.LongTensor(label).to(device)

def generate_mask(distance_matrixs,p_on_indexs,c_on_indexs, device):
    total_row=0
    total_col=0
    for distance_matrix in distance_matrixs:
        nrow, ncol = distance_matrix.shape
        total_row+=nrow
        total_col+=ncol
    distances = torch.zeros((total_row,total_col))
    current_row = 0
    current_col = 0
    for distance_matrix in distance_matrixs:
        nrow, ncol = distance_matrix.shape
        distances[current_row:current_row+nrow, current_col:current_col+ncol] = torch.LongTensor(distance_matrix)
        current_row+=nrow
        current_col+=ncol
    distances = distances[np.ix_(p_on_indexs, c_on_indexs)]
    mask=torch.where(distances>0,1,0)
    return distances.to(device),mask.to(device)

def finetune_topk_layer(train_loader, model, optimizer, device, pearson_criterion):
    """one epoch fine-tuning"""
    model.train()
    model.apply(fix_bn)
    losses = AverageMeter()

    for _,(_, peptide_chems, peptide_graphs, cdr3_chems, cdr3_graphs, distance_matrixs) in enumerate(train_loader):
        peptide_graphs = Batch.from_data_list(peptide_graphs)
        peptide_graphs = peptide_graphs.to(device)
        cdr3_graphs = Batch.from_data_list(cdr3_graphs)
        cdr3_graphs = cdr3_graphs.to(device)
        p_perm, p_scores, p_on_indexs, c_perm, c_scores, c_on_indexs, intermap_logits = model(peptide_graphs, cdr3_graphs, peptide_chems, cdr3_chems)
        p_scores = torch.exp(p_scores)
        c_scores = torch.exp(c_scores)
        joint_scores = torch.mm(p_scores.unsqueeze(0).T, c_scores.unsqueeze(0))
        distances, mask = generate_mask(distance_matrixs,p_on_indexs,c_on_indexs, device)
        loss2= pearson_criterion(joint_scores, distances, mask)
        bsz = len(distance_matrixs)
        losses.update(loss2.item(), bsz)
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
    return losses.avg

def finetune_classifier_layer(args, train_loader, model, optimizer, device, classify_criterion):
    """one epoch fine-tuning"""
    model.train()
    model.apply(fix_bn)
    losses = AverageMeter()

    for i,(pdbids, peptide_chems, peptide_graphs, cdr3_chems, cdr3_graphs, distance_matrixs) in enumerate(train_loader):
        peptide_graphs = Batch.from_data_list(peptide_graphs)
        peptide_graphs = peptide_graphs.to(device)
        cdr3_graphs = Batch.from_data_list(cdr3_graphs)
        cdr3_graphs = cdr3_graphs.to(device)
        p_perm, p_scores, p_on_indexs, c_perm, c_scores, c_on_indexs, intermap_logits = model(peptide_graphs, cdr3_graphs, peptide_chems, cdr3_chems)
        labels = generate_label(args, p_perm, c_perm, distance_matrixs, device)
        labels = labels.view(-1)
        intermap_logits = intermap_logits.view(-1,2)
        loss1 = classify_criterion(intermap_logits, labels)
        bsz = len(distance_matrixs)
        losses.update(loss1.item(), bsz)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
    return losses.avg

def validation(args, val_loader,model,device):
    with torch.no_grad():
        model.eval()
        p_chems = []
        c_chems = []
        p_perms = []
        c_perms = []
        intermaps = []
        for i,(pdbids, peptide_chems, peptide_graphs, cdr3_chems, cdr3_graphs, distance_matrixs) in enumerate(val_loader):
            p_chems.extend(peptide_chems)
            c_chems.extend(cdr3_chems)
            peptide_graphs = Batch.from_data_list(peptide_graphs)
            peptide_graphs = peptide_graphs.to(device)
            cdr3_graphs = Batch.from_data_list(cdr3_graphs)
            cdr3_graphs = cdr3_graphs.to(device)
            p_perm, p_scores, p_on_indexs, c_perm, c_scores, c_on_indexs, intermap_logits = model(peptide_graphs, cdr3_graphs, peptide_chems, cdr3_chems)
            p_perms.extend(p_perm)
            c_perms.extend(c_perm)
            intermap = intermap_logits[0][:,:,-1]
            intermaps.append(intermap.detach().cpu().numpy())
            p_scores = torch.exp(p_scores)
            c_scores = torch.exp(c_scores)
            joint_scores = torch.mm(p_scores.unsqueeze(0).T, c_scores.unsqueeze(0))
            distances, mask = generate_mask(distance_matrixs,p_on_indexs,c_on_indexs,device)
            joint_scores = joint_scores.view(-1)
            distances = distances.view(-1)
            mean1 = torch.mean(joint_scores)
            mean2 = torch.mean(distances)
            covariance = torch.mean((joint_scores - mean1) * (distances - mean2))
            std1 = torch.std(joint_scores)
            std2 = torch.std(distances)
            pearson_corr = covariance / (std1 * std2)
            print("NPCC:{:.4f}".format(-pearson_corr.item()))
            preds = torch.argmax(intermap_logits, dim=-1)
            labels = generate_label(args, p_perm, c_perm, distance_matrixs,device)
            labels = labels.view(-1).detach().cpu().numpy()
            preds = preds.view(-1).detach().cpu().numpy()
            scores = intermap_logits[:,:,:,1].view(-1).detach().cpu().numpy()
            acc, auroc, f1_score, precision, recall, auc_prc = compute_metrics(labels,preds,scores)
            print("ACC:{:.4f} AUROC:{:.4f} Precision:{:.4f} Recall:{:.4f} F1:{:.4f} AUPR:{:.4f}".format(
                acc, auroc, precision, recall, f1_score, auc_prc))
        return p_chems, p_perms, c_chems, c_perms, intermaps

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def Train(file_path, config_path=''):
    if len(config_path)==0:
        abpath = os.path.abspath(__file__)
        folder = os.path.dirname(abpath)
        args = read_config(os.path.join(folder,'config_atom.ini'))
    else:
        args = read_config(config_path)
    data_dir = file_path.rstrip(file_path.split('/')[-1])
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    dataset = pTCR_DataSet(file_path)
    kf = KFold(n_splits=len(dataset), shuffle=True, random_state=0)
    for train_index, val_index in kf.split(dataset):
        pdb = dataset.pdbids[val_index[0]]
        print(pdb)
        # if pdb!='2bnq':
            # continue
        min_loss = float('inf')
        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)

        pretrain_state_dict = None
        if os.path.exists('./antigenTCR_training_log/seq-level_parameters.pt'):
            state = torch.load('./antigenTCR_training_log/seq-level_parameters.pt', map_location=device)
            pretrain_state_dict = state['model']
        else:
            print('Please pre-train deepAntigen using seqence-level binding data.')
            return

        pearson_criterion = NegativePearsonCorrelationLossWithMask().to(device)
        classify_criterion = WeightedFocalLoss(reduction='sum').to(device)
        model = DeepGCN(args)
        model.load_state_dict(pretrain_state_dict, strict=False)
        model = model.to(device)
        model.peptide_cdr3_att.reset_param()
        model.frozen_encoder_layers()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args['lr'],
                               weight_decay=args['weight_decay'])

        print('Start finetuning topk layer')
        for epoch in range(args['epochs']):
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args['batchsize'],collate_fn=collate, pin_memory=True, drop_last=True)
            adjust_learning_rate(args, optimizer, epoch)
            loss = finetune_topk_layer(train_loader, model, optimizer, device, pearson_criterion)
            print('Epoch:{},Loss:{:.4f}'.format(epoch,loss))
            # if loss<min_loss:
                # min_loss = loss
                # if not os.path.exists(args.save_dir+dataset.pdbids[val_index[0]]):
                    # os.makedirs(args.save_dir+dataset.pdbids[val_index[0]])
                # save_file = args.save_dir+dataset.pdbids[val_index[0]]+'/best_model.pt'
                # save_model(model, optimizer, args, epoch, min_loss, save_file)

        model.frozen_topk_layers()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args['lr'],
                               weight_decay=args['weight_decay'])
        print('Start finetuning classifier layer')
        min_loss = float('inf')
        for epoch in range(args['epochs']):
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args['batchsize'],collate_fn=collate, pin_memory=True, drop_last=True)
            adjust_learning_rate(args, optimizer, epoch)
            loss = finetune_classifier_layer(args, train_loader, model, optimizer, device, classify_criterion)
            print('Epoch:{},Loss:{:.4f}'.format(epoch,loss))
            if loss<min_loss:
                min_loss = loss
                if not os.path.exists(args['save_dir']+pdb):
                    os.makedirs(args['save_dir']+pdb)
                save_file = args['save_dir']+pdb+'/atom-level_parameters.pt'
                save_model(model, optimizer, args, epoch, min_loss, save_file)
        print('Parameters of fine-tuned model have been saved to'+args['save_dir']+pdb)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args['batchsize'],collate_fn=collate, pin_memory=True)
        p_chems, p_perms, c_chems, c_perms, intermaps = validation(args, val_loader,model,device)
        pep_file = data_dir+'pdb_Extracted/'+pdb+'_peptide.pdb'
        cdr3_file = data_dir+'pdb_Extracted/'+pdb+'_cdr3.pdb'
        with open(pep_file, 'r') as pf:
            pep_lines = pf.readlines()
        with open(cdr3_file, 'r') as cf:
            cdr3_lines = cf.readlines()
        peptide_atoms = p_chems[0].GetAtoms()
        cdr3_atoms = c_chems[0].GetAtoms()
        peptide_conformer = p_chems[0].GetConformer()
        peptide_atom_positions = peptide_conformer.GetPositions()
        cdr3_conformer = c_chems[0].GetConformer()
        cdr3_atom_positions = cdr3_conformer.GetPositions()
        pas = []
        p_residue_names = []
        p_residue_nums = []
        dist = np.zeros((len(p_perms), len(c_perms)))
        for i in range(len(p_perms)):
            p_atom = peptide_atoms[int(p_perms[i])]
            p_idx = p_atom.GetIdx()
            pa = pep_lines[p_idx][12:16]
            pa = pa.strip(' ')
            p_residue_num = p_atom.GetPDBResidueInfo().GetResidueNumber()
            p_residue_name = p_atom.GetPDBResidueInfo().GetResidueName()
            pas.append(pa)
            p_residue_names.append(p_residue_name)
            p_residue_nums.append(str(p_residue_num))
            p_atom_coord = peptide_atom_positions[p_perms[i]]
            cas = []
            c_residue_names = []
            c_residue_nums = []
            for j in range(len(c_perms)):
                c_atom = cdr3_atoms[int(c_perms[j])]
                c_idx = c_atom.GetIdx()
                ca =cdr3_lines[c_idx][12:16]
                ca = ca.strip(' ')
                c_residue_num = c_atom.GetPDBResidueInfo().GetResidueNumber()
                c_residue_name = c_atom.GetPDBResidueInfo().GetResidueName()
                cas.append(ca)
                c_residue_names.append(c_residue_name)
                c_residue_nums.append(str(c_residue_num))
                c_atom_coord = cdr3_atom_positions[c_perms[j]]
                dist[i][j] = math.sqrt(np.sum(np.power(p_atom_coord-c_atom_coord, 2)))

        data1=pd.DataFrame(dist)
        data2=pd.DataFrame(intermaps[0])
        ylabels = []
        for num, res_name, atom_name in zip(p_residue_nums,p_residue_names,pas):
            label = num+'\n'+res_name+'\n'+atom_name
            ylabels.append(label)
        xlabels = []
        for num, res_name, atom_name in zip(c_residue_nums,c_residue_names,cas):
            label = atom_name+'\n'+res_name+'\n'+num
            xlabels.append(label)
        if not os.path.exists(args['output']):
            os.makedirs(args['output'])
        plt.figure(figsize=(10,8))
        ax=sns.heatmap(data1,cmap="Reds_r", cbar=True, xticklabels=xlabels, yticklabels=ylabels)
        plt.xlabel('CDR3')
        plt.ylabel('Peptide')
        plt.title('Sample:'+pdb)
        cbar = ax.collections[0].colorbar
        cbar.set_label('True distance')
        plt.savefig(args['output']+pdb+'_distance.png',dpi=600)
        plt.figure(figsize=(10,8))
        ax=sns.heatmap(data2,cmap="Reds_r", cbar=True, xticklabels=xlabels, yticklabels=ylabels)
        plt.xlabel('CDR3')
        plt.ylabel('Peptide')
        plt.title('Sample:'+pdb)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Contact probability')
        plt.savefig(args['output']+pdb+'_contact.png',dpi=600)
        print('Validation results have been saved to'+args['output']+pdb)