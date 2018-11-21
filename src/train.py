import argparse
import torch
import os

from nn import MyNet
from solver import Solver
from dataHandler import DataHandler
from utils import writeArgsFile

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet training script')
# --------------- General options ---------------
parser.add_argument('-x', '--expID', type=str, default='', metavar='S',
                    help='Experiment ID')
parser.add_argument('--train-dir', type=str, default='../datasets/train/', metavar='S',
                    help='folder for training files')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
# --------------- Training options ---------------
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--val-size', type=float, default=0.2, metavar='F',
                    help='val/(train+val) set size ratio (default: 0.2)')
parser.add_argument('--save-interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before saving (default: 5)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging (default: 10)')
parser.add_argument('--visdom', action='store_true', default=False,
                    help='enables VISDOM')
# --------------- Optimization options ---------------
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, metavar='F',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--beta1', type=float, default=0.9, metavar='F',
                    help='first momentum coefficient (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='F',
                    help='second momentum coefficient (default: 0.999)')
parser.add_argument('--epsilon', type=float, default=1e-8, metavar='F',
                    help='for numerical stability (default: 1e-8)')
# --------------- Model options ---------------
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='use previously saved model')
parser.add_argument('--truncation', type=float, default=3.0, metavar='N',
                    help='truncation value for distance field (default: 3)')
parser.add_argument('--log-transform', type=bool, default=True, metavar='B',
                    help='use log tranformation')
parser.add_argument('--mask', type=bool, default=True, metavar='B',
                    help='mask out known values')

## SETUP
print('SETUP')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
args.saveDir = os.path.join('../models/', args.expID)
writeArgsFile(args,args.saveDir)

torch.manual_seed(args.seed)
kwargs = {}
print('Seed: {:d}'.format(args.seed))

if use_cuda:
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  kwargs = {'num_workers': 4, 'pin_memory': True}
print('Cuda: {}'.format(use_cuda))

## LOAD DATASETS
print('\nLOADING DATASET & SAMPLER.')

train_data = DataHandler(args.train_dir, truncation=args.truncation)
print('Dataset truncation at: {:.1f}'.format(args.truncation))

train_sampler, val_sampler = train_data.subdivide_dataset(args.val_size,
                                                         shuffle=True,
                                                         seed=args.seed)
print('Dataset length: {:d} ({:d}/{:d})'.format(len(train_data), len(train_sampler),
                                                len(val_sampler)))
print('LOADED.')

## LOAD MODEL & SOLVER
print('\nLOADING NETWORK & SOLVER.')

model = MyNet(log_transform=args.log_transform)
checkpoint = {}
if args.model:
  checkpoint.update(torch.load(args.model, map_location=args.device))
  model.load_state_dict(checkpoint['model'])
print('Network params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

solver_args = {k: vars(args)[k] for k in ['saveDir', 'visdom', 'mask']}
optim_args = {'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.epsilon}
solver = Solver(optim_args=optim_args, args=solver_args)
print('Solver learning rate: {:.2e}'.format(args.lr))
print('Solver masked loss: {}'.format(args.mask))
print('LOADED.')

## TRAIN
print('\nTRAINING (batch size {:d}).'.format(args.batch_size))

train_loader = torch.utils.data.DataLoader(train_data,
                                          sampler=train_sampler,
                                          batch_size=args.batch_size,
                                          **kwargs)
val_loader = torch.utils.data.DataLoader(train_data,
                                        sampler=val_sampler,
                                        batch_size=args.batch_size,
                                        **kwargs)
solver.train(model, train_loader, val_loader, log_nth=args.log_interval,
            save_nth=args.save_interval, num_epochs=args.epochs, checkpoint=checkpoint)
print('FINISH.')

print('\nTHE END.')