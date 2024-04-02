import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        weights = (1-pt)**self.gamma
        fl = weights*ce_loss
        if self.reduction == 'sum':
            fl = fl.sum()
        elif self.reduction == 'mean':
            fl = fl.mean()
        else:
            raise ValueError(f"reduction '{reduction}' is not valid")
        return fl

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        alpha_factor = torch.where(labels == 1, 1., 1 - self.alpha)
        weighted_ce = alpha_factor*ce_loss
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        weights = (1-pt)**self.gamma
        fl = weights*weighted_ce
        if self.reduction == 'sum':
            fl = fl.sum()
        elif self.reduction == 'mean':
            fl = fl.mean()
        else:
            raise ValueError(f"reduction '{reduction}' is not valid")
        return fl


class NegativePearsonCorrelationLossWithMask(torch.nn.Module):
    def __init__(self):
        super(NegativePearsonCorrelationLossWithMask, self).__init__()

    def forward(self, A, B, mask):
        A_flat = A.view(-1)
        B_flat = B.view(-1)
        mask_flat = mask.view(-1)

        A_flat_masked = A_flat[mask_flat != 0]
        B_flat_masked = B_flat[mask_flat != 0]

        if A_flat_masked.numel() == 0:
            return torch.tensor(0.0)
        
        mean_A = torch.mean(A_flat_masked)
        mean_B = torch.mean(B_flat_masked)
        
        cov_AB = torch.mean((A_flat_masked - mean_A) * (B_flat_masked - mean_B))
        
        std_A = torch.std(A_flat_masked)
        std_B = torch.std(B_flat_masked)
        
        correlation_coefficient = cov_AB / (std_A * std_B)
        
        loss = 1 + correlation_coefficient
        
        return loss

def adjust_learning_rate(args, optimizer, epoch):
    lr = args['lr']
    if args['cosine']:
        eta_min = lr * (args['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args['epochs'])) / 2
    else:
        steps = np.sum(epoch > np.asarray(args['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (args['lr_decay_rate'] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args['warm'] and epoch <= args['warm_epochs']:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args['warm_epochs'] * total_batches)
        lr = args['warmup_from'] + p * (args['warmup_to'] - args['warmup_from'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(model,args):
    if args['optim']=='SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args['lr'],
                              momentum=args['momentum'],
                              weight_decay=args['weight_decay'])
    elif args['optim']=='Adam':
        optimizer = optim.Adam(model.parameters(), 
                           lr=args['lr'],
                           betas=(0.9, 0.98), 
                           eps=1e-09,
                           weight_decay=args['weight_decay'])
    else:
        raise ValueError
    return optimizer

def compute_metrics(target, pred, score):
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, score)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, score)
    auc_prc = metrics.auc(r, p)

    return acc, auroc, f1_score, precision, recall, auc_prc

def cal_confusion_matrix(target, pred):
    cm = metrics.confusion_matrix(target, pred)
    return cm

def save_model(model, optimizer, opt, epoch, max_auroc, save_file):
    print('==> Saving model...')
    state = {
        'opt': opt,
        'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'max_auroc':max_auroc
    }
    torch.save(state, save_file)
    del state