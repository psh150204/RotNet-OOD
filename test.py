import argparse
import random 
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader as dataloader
import torchvision.datasets as datasets

from sklearn.metrics import roc_auc_score

from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
from RotDataset import RotDataset
from utils import * 

def arg_parser():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--method', type=str, default='rot', help='rot, msp')
    parser.add_argument('--ood_dataset', type=str, default='cifar100', help='cifar100 | svhn')
    parser.add_argument('--num_workers', type=int, default=8)

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--rot-loss-weight', type=float, default=0.5, help='Multiplicative factor on the rot losses')

    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')


    args = parser.parse_args()

    return args 


def main():
    # arg parser 
    args = arg_parser()
    
    # set seed 
    set_seed(args.seed)  
    
    # dataset 
    id_testdata = datasets.CIFAR10('./data/', train=False, download=True)
    id_testdata = RotDataset(id_testdata, train_mode=False)

    if args.ood_dataset == 'cifar100':
        ood_testdata = datasets.CIFAR100('./data/', train=False, download=True)
    elif args.ood_dataset == 'svhn':
        ood_testdata = datasets.SVHN('./data/', split='test', download=True)
    else:
        raise ValueError(args.ood_dataset)
    ood_testdata = RotDataset(ood_testdata, train_mode=False)
    
    # data loader  
    id_test_loader = dataloader(id_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    ood_test_loader = dataloader(ood_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # load model
    num_classes = 10
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    model.rot_head = nn.Linear(128, 4)
    model = model.cuda()
    model.load_state_dict(torch.load('./models/trained_model_{}.pth'.format(args.method)))

    with torch.no_grad():
        # 1. calculate ood score by two methods(MSP, Rot)
        ood_scores = []

        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, _ in id_test_loader:
            if args.method == 'msp':
                ood_scores.append(-1 * torch.max(F.softmax(model(x_tf_0.cuda())[0], dim=-1), dim=-1).values)
            elif args.method == 'rot':
                ood_scores.append(rot_ood_score(
                    x_tf_0, x_tf_90, x_tf_180, x_tf_270, model))
            else :
                raise ValueError(args.method)

        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, _ in ood_test_loader:
            if args.method == 'msp':
                ood_scores.append(-1 * torch.max(F.softmax(model(x_tf_0.cuda())[0], dim=-1), dim=-1).values)
            else:
                ood_scores.append(rot_ood_score(
                    x_tf_0, x_tf_90, x_tf_180, x_tf_270, model))

    # 2. calculate AUROC by using ood scores 
    ood_scores = torch.cat(ood_scores, dim = 0).cpu()

    ground_truth = torch.zeros(len(id_test_loader.dataset))
    ground_truth = torch.cat([ground_truth, torch.ones(len(ood_test_loader.dataset))], dim = 0)
    
    print('AUROC of ' + args.ood_dataset + ' with ' + args.method + ' is %.4f'%(roc_auc_score(ground_truth, ood_scores)))


def rot_ood_score(x_tf_0, x_tf_90, x_tf_180, x_tf_270, model):
    batch_size = x_tf_0.size(0)
    batch_x = torch.cat(
        [x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).cuda()

    batch_rot_y = torch.cat((
        torch.zeros(batch_size),
        torch.ones(batch_size),
        2 * torch.ones(batch_size),
        3 * torch.ones(batch_size)
        ), 0).long().cuda()

    logits, pen = model(batch_x)

    classification_logits = logits[:batch_size]
    prob = F.softmax(classification_logits, dim=-1)
    uniform_dist = torch.zeros_like(prob).fill_(0.1)
    KLdiv = (prob * (prob/uniform_dist).log()).sum(dim=-1)

    rot_logits = model.rot_head(pen)
    rot_loss = F.cross_entropy(
        rot_logits, batch_rot_y, reduction='none')
    rot_losses = list(rot_loss.split(batch_size))
    for loss in rot_losses:
        loss.unsqueeze_(0)

    rot_loss = torch.cat(rot_losses, dim=0).mean(dim=0)

    return -1 * KLdiv + rot_loss


if __name__ == "__main__":
    main()
