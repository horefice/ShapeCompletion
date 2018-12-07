#!/usr/bin/env python3
import argparse
import torch
import os

from nn import MyNet
from solver import Solver
from dataHandler import DataHandler
from utils import writeArgsFile

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet evaluation script')
# --------------- General options ---------------
parser.add_argument('-x', '--expID', type=str, default='', metavar='S', 
                    help='experiment ID')
parser.add_argument('--dir', type=str, default='../datasets/test/', metavar='S', 
                    help='folder for test files')
parser.add_argument('--num-workers', type=int, default=4, metavar='N', 
                    help='number of workers for the dataloader')
parser.add_argument('--benchmark', type=bool, default=True, metavar='B', 
                    help='uses CUDNN benchmark')
parser.add_argument('--no-cuda', action='store_true', 
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N', 
                    help='random seed (default: 1)')
# --------------- Evaluation options ---------------
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)')
# --------------- Model options ---------------
parser.add_argument('--model', type=str, default='../models/checkpoint.pth', metavar='S', 
                    help='uses previously saved model')
parser.add_argument('--n-features', type=int, default=80, metavar='N', 
                    help='number of features for unet model')
parser.add_argument('--truncation', type=float, default=2.5, metavar='F', 
                    help='truncation value for distance field (default: 3)')
parser.add_argument('--log-transform', type=bool, default=True, metavar='B', 
                    help='uses log tranformation')
parser.add_argument('--mask', type=bool, default=True, metavar='B', 
                    help='mask out known values')

## SETUP
print('SETUP')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
saveDir = os.path.join('../models/', args.expID)
writeArgsFile(args,saveDir)

torch.manual_seed(args.seed)
kwargs = {}
print('Seed: {:d}'.format(args.seed))

print('Device: {}'.format(args.device))
if use_cuda:
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = args.cudnn
  kwargs = {'num_workers': args.workers, 'pin_memory': True}
  print('Workers: {:d}'.format(args.workers))
  print('Benchmark: {}'.format(args.benchmark))

## LOAD DATASETS
print('\nLOADING DATASET.')

test_data = DataHandler(args.dir, truncation=args.truncation)
print('Dataset truncation at: {:.1f}'.format(args.truncation))
print('Dataset length: {:d}'.format(len(test_data)))
print('Batch size: {:d} x {}'.format(args.batch_size, list(test_data[0][0].size())))
print('LOADED.')

## LOAD MODEL & SOLVER
print('\nLOADING NETWORK & SOLVER.')

model = MyNet(n_features=args.n_features, log_transform=args.log_transform)
checkpoint = torch.load(args.model, map_location=args.device)
model.load_state_dict(checkpoint['model'])
model.to(args.device)
print('Network parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

solver = Solver(loss_func=torch.nn.L1Loss(reduction='sum'), args={'saveDir': saveDir, 'mask': args.mask})
print('Solver masked loss: {}'.format(args.mask))
print('LOADED.')

## TEST
print('\nTESTING.')

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                          shuffle=False, **kwargs)

test_err = solver.eval(model, test_loader, progress_bar=True)
print('Test error: {:.3e}'.format(test_err))
print('FINISH.')

print('\nTHE END.')