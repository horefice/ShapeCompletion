import numpy as np
import sys
import os
import shutil
import torch

from demo import main as demo
from utils import AverageMeter
from viz import Viz

class Solver(object):
  default_args = {'saveDir': '../models/',
                  'visdom': False,
                  'mask': True,
                  'save_interval': 5}

  def __init__(self, optim=torch.optim.Adam, optim_args={},
               loss_func=torch.nn.L1Loss(), args={}):
    self.optim_args = optim_args
    self.optim = optim
    self.loss_func = loss_func
    self.args = dict(self.default_args, **args)
    self.visdom = Viz() if self.args['visdom'] else False
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
    optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()),
                       **self.optim_args)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=.5)
    
    iter_per_epoch = len(train_loader)
    start_epoch = 0
    best_val_acc = 0.0
    is_best = False

    if len(checkpoint) > 0:
      start_epoch = checkpoint['epoch']
      best_val_acc = checkpoint['best_val_acc']
      optim.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      self._load_history()
      print("=> Loaded checkpoint (epoch {:d})".format(checkpoint['epoch']))
    else:
      self._save_checkpoint({
        'epoch': start_epoch,
        'best_val_acc': best_val_acc,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'scheduler': scheduler.state_dict()
      }, is_best)

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
    #   [Epoch 1/5] VAL acc/loss: 53.90%/1.310                             #
    #   ...                                                                #
    ########################################################################

    for epoch in range(start_epoch, num_epochs):
      # TRAINING
      model.train()
      scheduler.step()
      train_loss = 0

      for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)

        if model.log_transform:
          targets = targets.abs().add(1).log()

        optim.zero_grad()
        outputs = model(inputs)
        if self.args['mask']:
          mask = inputs[:,[1]] == 1
          outputs.masked_fill_(mask, 0)
          targets.masked_fill_(mask, 0)

        loss = self.loss_func(outputs, targets)
        loss.backward()
        optim.step()

        self.train_loss_history.append(loss.item())
        if log_nth and i % log_nth == 0:
          last_log_nth_losses = self.train_loss_history[-log_nth:]
          train_loss = np.mean(last_log_nth_losses)
          print('[Iteration {:d}/{:d}] TRAIN loss: {:.3f}'
                .format(i + epoch * iter_per_epoch,
                  iter_per_epoch * num_epochs,
                  train_loss))

          if self.visdom:
            x = epoch + i / iter_per_epoch
            self.visdom.update_plot(x=x, y=train_loss,
                                    window=iter_plot,
                                    type_upd="append")

      if log_nth:
        print('[Epoch {:d}/{:d}] TRAIN   loss: {:.3f}'.format(epoch + 1,
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


        if log_nth:
          print('[Epoch {:d}/{:d}] VAL acc/loss: {:.2%}/{:.3f}'.format(epoch + 1,
                                                                      num_epochs,
                                                                      val_acc,
                                                                      val_loss))
      # do checkpointing
      if self.args['save_interval'] and (epoch+1) % self.args['save_interval'] == 0:
        self._save_checkpoint({
          'epoch': epoch + 1,
          'best_val_acc': best_val_acc,
          'state_dict': model.state_dict(),
          'optimizer': optim.state_dict(),
          'scheduler': scheduler.state_dict()
        }, is_best)

    if self.visdom:
      self.visdom.matplot(demo('../models/checkpoint.pth', '../datasets/sample/overfit.h5'))

  def test(self, model, test_loader, tolerance=1):
    """
    Test a given model with the provided data.

    Inputs:
    - model: model object initialized from a torch.nn.Module
    - test_loader: test data in torch.utils.data.DataLoader
    - tolerance: acceptable percentage error used to compute test accuracy
    """
    test_acc = AverageMeter()
    test_loss = AverageMeter()
    model.eval()
    device = torch.device("cuda:0" if model.is_cuda else "cpu")

    with torch.no_grad():
      for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if model.log_transform:
          targets = targets.abs().add(1).log()
          tolerance = np.log(tolerance+1)
        
        outputs = model(inputs)
        if self.args['mask']:
          mask = inputs[:,[1]] == 1
          outputs.masked_fill_(mask, 0)
          targets.masked_fill_(mask, 0)

        loss = self.loss_func(outputs, targets)
        test_loss.update(loss.item())

        # L2 approach
        correct = (torch.lt(torch.abs(torch.add(outputs, -targets)), tolerance) - mask).sum().item()
        test_acc.update(correct / (np.prod(targets.size()) - mask.sum().item()))

    return test_acc.avg, test_loss.avg

  def _save_checkpoint(self, state, is_best, fname='checkpoint.pth'):
    """
    Save current state of training and trigger method to save training history.
    """
    print('Saving at checkpoint...')
    path = os.path.join(self.args['saveDir'], fname)
    torch.save(state, path)
    self._save_history()
    if is_best:
      return #shutil.copyfile(path, os.path.join(self.args['saveDir'], 'model_best.pth'))

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
    np.savez(os.path.join(self.args['saveDir'], fname),
            train_loss_history=self.train_loss_history,
            val_loss_history=self.val_loss_history,
            val_acc_history=self.val_acc_history)

  def _load_history(self, fname="train_history.npz"):
    """
    Load training history. Conventionally the fname should end with "*.npz".
    """
    npzfile = np.load(os.path.join(self.args['saveDir'], fname))
    self.train_loss_history = npzfile['train_loss_history'].tolist()
    self.val_acc_history = npzfile['val_acc_history'].tolist()
    self.val_loss_history = npzfile['val_loss_history'].tolist()
