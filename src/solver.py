import numpy as np
import sys
import os
import shutil

import torch
from utils import AverageMeter
from viz import Viz

class Solver(object):
  default_adam_args = {"lr": 1e-4,
                       "betas": (0.9, 0.999),
                       "eps": 1e-8,
                       "weight_decay": 0.0}

  def __init__(self, optim=torch.optim.Adam, optim_args={},
               loss_func=torch.nn.L1Loss(), saveDir='../models/', vis=False):
    optim_args_merged = self.default_adam_args.copy()
    optim_args_merged.update(optim_args)
    self.optim_args = optim_args_merged
    self.optim = optim
    self.loss_func = loss_func
    self.saveDir = saveDir
    self.visdom = Viz() if vis else False
    self._reset_history()

  def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0,
            checkpoint={}):
    """
    Train a given model with the provided data.

    Inputs:
    - model: model object initialized from a torch.nn.Module
    - train_loader: train data in torch.utils.data.DataLoader
    - val_loader: val data in torch.utils.data.DataLoader
    - num_epochs: total number of training epochs
    - log_nth: log training accuracy and loss every nth iteration
    - checkpoint: object used to resume training from a checkpoint
    """
    optim = self.optim(filter(lambda p: p.requires_grad,model.parameters()),
                       **self.optim_args)
    scheduler = False # torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
    
    iter_per_epoch = len(train_loader)
    start_epoch = 0
    best_val_acc = 0.0
    is_best = False

    if len(checkpoint) > 0:
      start_epoch = checkpoint['epoch']
      best_val_acc = checkpoint['best_val_acc']
      optim.load_state_dict(checkpoint['optimizer'])
      self._load_history()
      print("=> Loaded checkpoint (epoch {:d})".format(checkpoint['epoch']))

    device = torch.device("cuda:0" if model.is_cuda else "cpu")

    if self.visdom:
      iter_plot = self.visdom.create_plot('Epoch', 'Loss', 'Train Loss',
                                          {'ytype':'log'})

    ########################################################################
    # The log should like something like:                                  #
    #   ...                                                                #
    #   [Iteration 700/4800] TRAIN loss: 1.452                             #
    #   [Iteration 800/4800] TRAIN loss: 1.409                             #
    #   [Iteration 900/4800] TRAIN loss: 1.374                             #
    #   [Epoch 1/5] TRAIN   loss: 0.560/1.374                              #
    #   [Epoch 1/5] VAL acc/loss: 53.90%/1.310                              #
    #   ...                                                                #
    ########################################################################

    for epoch in range(start_epoch, num_epochs):
      # TRAINING
      model.train()
      train_loss = 0

      for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = self.loss_func(outputs, targets)
        loss.backward()
        optim.step()

        self.train_loss_history.append(loss.data.cpu().numpy())
        if log_nth and i % log_nth == 0:
          last_log_nth_losses = self.train_loss_history[-log_nth:]
          train_loss = np.mean(last_log_nth_losses)
          print('[Iteration {:d}/{:d}] TRAIN loss: {:.2f}'
                .format(i + epoch * iter_per_epoch,
                  iter_per_epoch * num_epochs,
                  train_loss))

          if self.visdom:
            self.visdom.update_plot(x=epoch + i / iter_per_epoch,
                                    y=train_loss,
                                    window=iter_plot,
                                    type_upd="append")

      if log_nth:
        print('[Epoch {:d}/{:d}] TRAIN   loss: {:.2f}'.format(epoch + 1,
                                                              num_epochs,
                                                              train_loss))

      # VALIDATION
      if len(val_loader):
        val_acc, val_loss = self.test(model, val_loader)
        self.val_acc_history.append(val_acc)
        self.val_loss_history.append(val_loss)

        # Set best model to the one with highest validation set accuracy
        is_best = val_acc >= best_val_acc
        best_val_acc = max(val_acc,best_val_acc)

        # Reduce LR progressively
        if scheduler:
          scheduler.step(val_acc)

        if log_nth:
          print('[Epoch {:d}/{:d}] VAL acc/loss: {:.2%}/{:.2f}'.format(epoch + 1,
                                                                      num_epochs,
                                                                      val_acc,
                                                                      val_loss))
      
      self._save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'optimizer' : optim.state_dict(),
      }, is_best)

  def test(self, model, test_loader):
    """
    Test a given model with the provided data.

    Inputs:
    - model: model object initialized from a torch.nn.Module
    - test_loader: test data in torch.utils.data.DataLoader
    """
    test_acc = AverageMeter()
    test_loss = AverageMeter()
    model.eval()
    device = torch.device("cuda:0" if model.is_cuda else "cpu")

    with torch.no_grad():
      for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model.forward(inputs)
        loss = self.loss_func(outputs, targets)
        test_loss.update(loss.data.cpu().numpy())

    test_acc = float(test_acc.avg)
    test_loss = float(test_loss.avg)
    return test_acc, test_loss

  def _save_checkpoint(self, state, is_best, fname='checkpoint.pth'):
    """
    Save current state of training and trigger method to save training history.
    """
    print('Saving at checkpoint...')
    path = os.path.join(self.saveDir, fname)
    torch.save(state, path)
    self._save_history()
    if is_best:
      shutil.copyfile(path, os.path.join(self.saveDir, 'model_best.pth'))

  def _reset_history(self):
    """
    Resets train and val histories.
    """
    self.train_loss_history = []
    self.val_acc_history = []
    self.val_loss_history = []

  def _save_history(self, fname="train_history.npz"):
    """
    Save training history. Conventionally the fname should end with "*.npz".
    """
    np.savez(os.path.join(self.saveDir, fname),
            train_loss_history=self.train_loss_history,
            val_loss_history=self.val_loss_history,
            val_acc_history=self.val_acc_history)

  def _load_history(self, fname="train_history.npz"):
    """
    Load training history. Conventionally the fname should end with "*.npz".
    """
    npzfile = np.load(os.path.join(self.saveDir, fname))
    self.train_loss_history = npzfile['train_loss_history'].tolist()
    self.val_acc_history = npzfile['val_acc_history'].tolist()
    self.val_loss_history = npzfile['val_loss_history'].tolist()
