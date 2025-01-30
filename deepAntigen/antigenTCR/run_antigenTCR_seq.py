import os
from .load_dataset.load_seq import pTCR_DataSet, collate
from .networks.pTCR_seq import DeepGCN
from .utils.model_utils import *
from torch_geometric.data import Batch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
from configparser import ConfigParser

def read_config(path):
    conf = ConfigParser()
    conf.read(path)
    args = dict(conf['settings'])
    for key,value in args.items():
        if key in ['num_process','hidden_size','depth','k','heads','batchsize','epochs', 'print_freq', 'save_freq', 'fold']:
            args[key] = int(args[key])
        elif key in ['lr', 'lr_decay_rate', 'weight_decay', 'momentum']:
            args[key] = float(args[key])
        elif key=='cosine':
            args[key] = bool(int(args[key]))
        else:
            pass
    # warm-up for large-batch training,
    if args['batchsize'] >= 256:
        args['warm'] = True
    else:
        args['warm'] = False

    if args['warm']:
        args['warmup_from'] = 0.1*args['lr']
        args['warm_epochs'] = 10
        args['warmup_to'] = args['lr']
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

def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch, device):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    train_preds = []
    train_trues = []
    train_scores = []
    for idx, (_, _, _, labels, peptide_graphs, cdr3_graphs) in enumerate(train_loader):
        peptide_graphs = Batch.from_data_list(peptide_graphs)
        peptide_graphs = peptide_graphs.to(device)
        cdr3_graphs = Batch.from_data_list(cdr3_graphs)
        cdr3_graphs = cdr3_graphs.to(device)
        labels = labels.to(device)
        logits = model(peptide_graphs, cdr3_graphs)
        loss = criterion(logits, labels)
        bsz = labels.shape[0]
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        scores = logits[:,1]
        train_preds.extend(preds.detach().cpu().numpy())
        train_scores.extend(scores.detach().cpu().numpy())
        train_trues.extend(labels.detach().cpu().numpy())    
    acc, auroc, f1_score, precision, recall, auc_prc = compute_metrics(train_trues, train_preds, train_scores)
    print("Epoch:{} Train_loss:{:.4f} ACC:{:.4f} AUROC:{:.4f} Precision:{:.4f} Recall:{:.4f} F1:{:.4f} AUPR:{:.4f}".format(
        epoch, losses.avg, acc, auroc, precision, recall, f1_score, auc_prc))

    return auroc

def valid(valid_loader, model, criterion, epoch, device):
    valid_losses = AverageMeter()
    val_preds = []
    val_trues = []
    val_scores = []
    with torch.no_grad():
        model.eval()
        for idx, (_, _, _, labels, peptide_graphs, cdr3_graphs) in enumerate(valid_loader):
            peptide_graphs = Batch.from_data_list(peptide_graphs)
            peptide_graphs = peptide_graphs.to(device)
            cdr3_graphs = Batch.from_data_list(cdr3_graphs)
            cdr3_graphs = cdr3_graphs.to(device)
            labels = labels.to(device)
            logits = model(peptide_graphs, cdr3_graphs)
            loss = criterion(logits, labels)
            bsz = labels.shape[0]
            valid_losses.update(loss.item(), bsz)
            preds = logits.argmax(dim=1)
            scores = logits[:,1]
            val_preds.extend(preds.detach().cpu().numpy())
            val_scores.extend(scores.detach().cpu().numpy())
            val_trues.extend(labels.detach().cpu().numpy())
    acc, auroc, f1_score, precision, recall, auc_prc = compute_metrics(val_trues, val_preds, val_scores)
    print("Epoch:{} Val_loss:{:.4f} ACC:{:.4f} AUROC:{:.4f} Precision:{:.4f} Recall:{:.4f} F1:{:.4f} AUPR:{:.4f}".format(
        epoch, valid_losses.avg, acc, auroc, precision, recall, f1_score, auc_prc))
    return auroc

def test(test_loader, model, device):
    model.eval()
    all_preds = []
    all_trues = []
    all_scores = []
    all_peptides = []
    all_cdr3s = []
    with torch.no_grad():
        for idx, (_, peptides, cdr3s, labels, peptide_graphs, cdr3_graphs) in enumerate(test_loader):
            peptide_graphs = Batch.from_data_list(peptide_graphs)
            peptide_graphs = peptide_graphs.to(device)
            cdr3_graphs = Batch.from_data_list(cdr3_graphs)
            cdr3_graphs = cdr3_graphs.to(device)
            labels = labels.to(device)
            logits = model(peptide_graphs, cdr3_graphs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            scores = logits[:,1]
            all_scores.extend(scores.detach().cpu().numpy())
            all_trues.extend(labels.detach().cpu().numpy())
            all_peptides.extend(peptides)
            all_cdr3s.extend(cdr3s)
    # acc, auroc, f1_score, precision, recall, auc_prc = compute_metrics(all_trues, all_preds, all_scores)
    # print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
        # acc, auroc, f1_score, precision, recall, auc_prc))
    return all_peptides, all_cdr3s, all_scores, all_trues


def Inference(test_file_path,model_path='', multi_process=8):
    headline = pd.read_csv(test_file_path, nrows=0)
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    if len(model_path)==0:
        abpath = os.path.abspath(__file__)
        folder = os.path.dirname(abpath)
        state = torch.load(os.path.join(folder,'Weights','seq-level_parameters.pt'))
    else:
        state = torch.load(model_path)

    pretrain_state_dict = state['model']
    args = state['opt']
    dataset = pTCR_DataSet(test_file_path)
    dataloader = DataLoader(dataset, batch_size=args['batchsize'], collate_fn=collate, shuffle=False, pin_memory=True, num_workers=args['num_process'])
    model= set_model(args, pretrain_state_dict,device)
    all_peptides, all_cdr3s, all_scores, all_labels = test(dataloader, model,device)
    if 'label' in headline.columns:
        df = pd.DataFrame({"peptide":all_peptides,"binding_TCR":all_cdr3s,"score":all_scores,"label":all_labels},index=list(range(len(all_peptides))))
    else:
        df = pd.DataFrame({"peptide":all_peptides,"binding_TCR":all_cdr3s,"score":all_scores},index=list(range(len(all_peptides))))
    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
    df.to_csv(args['output']+'pTCR_predictions.csv',index=False)
    print('Prediction results have been saved to'+args['output']+'pTCR_predictions.csv')
    return df

def Train(k_fold_dir, config_path=''):
    if len(config_path)==0:
        abpath = os.path.abspath(__file__)
        folder = os.path.dirname(abpath)
        args = read_config(os.path.join(folder,'config_seq.ini'))
    else:
        args = read_config(config_path)
    max_val_auroc = 0
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    train_path=os.path.join(k_fold_dir,'train_fold'+str(args['fold'])+'.csv')
    val_path=os.path.join(k_fold_dir,'val_fold'+str(args['fold'])+'.csv')
    train_dataset = pTCR_DataSet(train_path,aug=True,test=False)
    val_dataset = pTCR_DataSet(val_path,aug=False,test=False)

    start_epoch = 0
    model = DeepGCN(args)
    criterion = FocalLoss(reduction='sum')
    model = model.to(device)
    criterion = criterion.to(device)
    if torch.cuda.is_available():
        cudnn.benchmark = True
    optimizer = set_optimizer(model,args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batchsize'],
                              collate_fn=collate, pin_memory=True, num_workers=args['num_process'])
    valid_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args['batchsize'],
                          collate_fn=collate, pin_memory=True, num_workers=args['num_process'])
    for epoch in range(start_epoch+1, args['epochs']+1):
        adjust_learning_rate(args, optimizer, epoch)
        train_auroc = train_one_epoch(args, train_dataloader, model, criterion, optimizer, epoch, device)
        if epoch % args['print_freq'] == 0:
            val_auroc = valid(valid_dataloader, model, criterion, epoch, device)
            if val_auroc > max_val_auroc:
                max_val_auroc = val_auroc
                if not os.path.exists(args['save_dir']):
                    os.makedirs(args['save_dir'])
                save_file = args['save_dir']+'seq-level_parameters.pt'
                save_model(model, optimizer, args, epoch, max_val_auroc, save_file)
        if epoch % args['save_freq'] == 0:
            if not os.path.exists(args['save_dir']):
                os.makedirs(args['save_dir'])
            save_file = args['save_dir']+'epoch'+str(epoch)+'.pt'
            save_model(model, optimizer, args, epoch, max_val_auroc, save_file)
    print('Parameters of pre-trained model have been save to' + args['save_dir'])