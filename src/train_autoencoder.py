#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import os

from nn import AENet
from dataHandler import MyDataset, MyDataParallel
from utils import writeArgsFile, AverageMeter, improveSDF


# SETTINGS
parser = argparse.ArgumentParser(description='MyDeepSDF training script')
# --------------- General options ---------------
parser.add_argument('-x', '--expID', type=str, default='', metavar='S',
                    help='experiment ID')
parser.add_argument('--dir', type=str, default='../datasets/train/',
                    metavar='S', help='directory for training files')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='number of workers for the dataloader per GPU')
parser.add_argument('--benchmark', type=bool, default=True, metavar='B',
                    help='uses CUDNN benchmark')
parser.add_argument('--no-cuda', action='store_true',
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
# --------------- Training options ---------------
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-s', '--sub-epochs', type=int, default=0, metavar='N',
                    help='number of sub-epochs for validation (default: 0)')
parser.add_argument('--val-size', type=float, default=0.125, metavar='F',
                    help='val/(train+val) set size ratio (default: 0.125)')
parser.add_argument('--save-interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before saving (default: 5)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches before logging (default: 10)')
parser.add_argument('--visdom', action='store_true',
                    help='enables VISDOM')
# --------------- Optimization options ---------------
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3,
                    metavar='F', help='learning rate (default: 1e-3)')
parser.add_argument('--beta1', type=float, default=0.9, metavar='F',
                    help='first momentum coefficient (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='F',
                    help='second momentum coefficient (default: 0.999)')
parser.add_argument('--epsilon', type=float, default=1e-8, metavar='F',
                    help='for numerical stability (default: 1e-8)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='F',
                    help='L2 penalty/regularization (default: 1e-4)')
# --------------- Model options ---------------
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='uses previously saved model')
parser.add_argument('--n-features', type=int, default=128, metavar='N',
                    help='number of features for unet model (default: 128)')
parser.add_argument('--truncation', type=float, default=3.0, metavar='F',
                    help='truncation value for distance field (default: 3)')
parser.add_argument('--log-transform', type=bool, default=True, metavar='B',
                    help='uses log tranformation (default: True)')
parser.add_argument('--mask', type=bool, default=True, metavar='B',
                    help='mask out known values (default: True)')
parser.add_argument('--colored', type=bool, default=True, metavar='B',
                    help='uses model with color information (default: True)')
# SETUP
print('SETUP')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
saveDir = os.path.join('../models/', args.expID)
writeArgsFile(args, saveDir)

torch.manual_seed(args.seed)
kwargs = {}
print('Seed: {:d}'.format(args.seed))

print('Device: {}'.format(args.device))

if use_cuda:
    print('\nCUDA')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = args.benchmark
    num_gpus = torch.cuda.device_count()
    kwargs = {'num_workers': num_gpus * args.workers, 'pin_memory': True}
    print("Number of GPUs: {:d}".format(num_gpus))
    print('Workers/GPU: {:d}'.format(args.workers))
    print('Benchmark: {}'.format(args.benchmark))

# LOAD DATASETS
print('\nLOADING DATASET & SAMPLER.')

train_data = MyDataset(args.dir, truncation=args.truncation)
print('Dataset truncation at: {:.1f}'.format(args.truncation))

train_sampler, val_sampler = train_data.subdivide_dataset(args.val_size,
                                                          shuffle=True,
                                                          seed=args.seed)
print('Dataset length: {:d} ({:d}/{:d})'.format(len(train_data),
                                                len(train_sampler),
                                                len(val_sampler)))
print('Batch size: {:d} x {}'.format(args.batch_size,
                                     list(train_data[0][0].size())))
print('LOADED.')

# LOAD MODEL & SOLVER
print('\nLOADING NETWORK.')

model = AENet(n_features=int(args.n_features/8))
checkpoint = {}
if args.model:
    checkpoint.update(torch.load(args.model, map_location=args.device))
    model.load_state_dict(checkpoint['model'])
if use_cuda and num_gpus > 1:
    model = MyDataParallel(model)
model.to(args.device)
print('Network parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

print('LOADED.')

# TRAIN
print('\nTRAINING.')

train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler,
                                           batch_size=args.batch_size, **kwargs)
val_loader = torch.utils.data.DataLoader(train_data, sampler=val_sampler,
                                         batch_size=args.batch_size, **kwargs)

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
loss_func = torch.nn.MSELoss(reduction='mean')
iter_per_epoch = len(train_loader)
train_loss_history = []
val_loss_history = []
start_epoch = 0

if len(checkpoint) > 0:
    start_epoch = checkpoint['epoch']
    optim.load_state_dict(checkpoint['optimizer'])
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']
    print("=> Loaded checkpoint (epoch {:d})".format(checkpoint['epoch']))

device = torch.device("cuda:0" if model.is_cuda else "cpu")
for epoch in range(start_epoch, args.epochs):
    for i, (_, inputs) in enumerate(train_loader, 1):
        # Prepare data
        inputs = inputs.float().to(device)

        # Forward pass
        model.train()
        optim.zero_grad()
        outputs, mu, logvar = model(inputs)

        # Compute loss and backward pass
        BCE = loss_func(outputs, inputs)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        loss.backward()
        optim.step()

        # Update progress
        batch_loss = float(loss)
        train_loss_history.append(batch_loss)

        # Logging iteration
        if args.log_interval and i % args.log_interval == 0:
            mean_nth_loss = np.mean(train_loss_history[-args.log_interval:])
            print('[Iteration {:d}/{:d}] TRAIN loss: {:.2e}'.format(i + epoch * iter_per_epoch,
                  iter_per_epoch * args.epochs, mean_nth_loss))

        # Validation
        if i % (iter_per_epoch / (args.sub_epochs + 1)) < 1:
            sub_val_loss = AverageMeter()

            model.log_transform = False
            model.eval()
            with torch.no_grad():
                for (_, inputs) in val_loader:
                    # Prepare data
                    inputs = inputs.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    batch_loss = 0#float(loss_func(outputs, inputs))
                    sub_val_loss.update(batch_loss)

            val_loss_history.append(sub_val_loss.avg)

            if args.log_interval:
                print('[Iteration {:d}/{:d}] VAL   loss: {:.2e}'.format(i + epoch * iter_per_epoch,
                                                                        iter_per_epoch * args.epochs, val_loss_history[-1]))

    # Epoch logging
    print('[Epoch {:d}/{:d}] TRAIN loss: {:.2e}'.format(epoch + 1, args.epochs, np.mean(train_loss_history[-iter_per_epoch:])))
    print('[Epoch {:d}/{:d}] VAL   loss: {:.2e}'.format(epoch + 1, args.epochs, val_loss_history[-1]))

# Plot
if False:
    if inputs is not None and outputs is not None:
        from demo import plot_3d
        import matplotlib.pyplot as plt
        plot_3d(inputs.data.numpy()[0], improveSDF(outputs.data.numpy()[0]), i=0)
        plt.show()

# SAVE
path = os.path.join(saveDir, 'AENet.pth')
state = {'epoch': args.epochs,
         'model': model.state_dict(),
         'optimizer': optim.state_dict(),
         'train_loss_history': train_loss_history,
         'val_loss_history': val_loss_history}
torch.save(state, path)
print('FINISH.')

print('\nTHE END.')
