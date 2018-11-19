import numpy as np
import argparse
import torch
import datetime
import os

from nn import MyNet
from solver import Solver
from dataHandler import DataHandler
from utils import writeArgsFile

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet evaluation script')
# --------------- General options ---------------
parser.add_argument('-x', '--expID', type=str, default='', metavar='S',
                    help='Experiment ID')
parser.add_argument('--test-dir', type=str, default='../datasets/test/', metavar='S',
                    help='folder for test files')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
# --------------- Evaluation options ---------------
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
# --------------- Model options ---------------
parser.add_argument('--model', type=str, default='../models/checkpoint.pth', metavar='S',
                    help='use previously saved model')
parser.add_argument('--truncation', type=int, default=3, metavar='N',
                    help='truncation value for distance field (default: 3)')
parser.add_argument('--log-transform', type=bool, default=True, metavar='B',
                    help='use log tranformation')
parser.add_argument('--mask', type=bool, default=True, metavar='B',
                    help='mask out known values')

## SETUP
print('SETUP')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
args.saveDir = os.path.join('../models/', args.expID)
writeArgsFile(args,args.saveDir)

torch.manual_seed(args.seed)
kwargs = {}
print('Seed: {:d}'.format(args.seed))

if args.cuda:
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  kwargs = {'num_workers': 0, 'pin_memory': True}
print('Cuda: {}'.format(args.cuda))

## LOAD DATASETS
print('\nLOADING DATASET.')

test_data = DataHandler(args.test_dir, truncation=args.truncation)
print('Dataset truncation at: {:.1f}'.format(args.truncation))
print('Dataset length: {:d}'.format(len(test_data)))
print('LOADED.')

## LOAD MODEL & SOLVER
print('\nLOADING NETWORK & SOLVER.')

model = MyNet(log_transform=args.log_transform)
checkpoint = torch.load(args.model, map_location=args.device)
model.load_state_dict(checkpoint['model'])
print('Network params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

solver_args = {k: vars(args)[k] for k in ['saveDir', 'mask']}
solver = Solver(args=solver_args)
print('Solver masked loss: {}'.format(args.mask))
print('LOADED.')

## TEST
print('\nTESTING (batch size {:d}).'.format(args.batch_size))

test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=args.batch_size,
                                          shuffle=False, **kwargs)
test_acc, test_loss = solver.test(model, test_loader)

print('Test accuracy: {:.2%}'.format(test_acc))
print('Test loss: {:.2e}'.format(test_loss))
print('FINISH.')

print('\nTHE END.')