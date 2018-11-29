import numpy as np
import torch
import os

from demo import main as demo
from utils import AverageMeter, Viz, compute_l1_error

class Solver(object):
  default_args = {'saveDir': '../models/',
                  'visdom': False,
                  'mask': True}

  def __init__(self, optim=torch.optim.Adam, optim_args={},
               lrs=torch.optim.lr_scheduler.StepLR, lrs_args={},
               loss_func=torch.nn.SmoothL1Loss(), args={}):
    self.optim_args = optim_args
    self.optim = optim
    self.lrs_args = lrs_args
    self.lrs = lrs
    self.loss_func = loss_func
    self.args = dict(self.default_args, **args)
    self.visdom = Viz() if self.args['visdom'] else False
    self._reset_history()

  def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0,
            save_nth=0, checkpoint={}):
    """
    Train a given model with the provided data.

    Inputs:
    - model: model object initialized from a torch.nn.Module
    - train_loader: train data in torch.utils.data.DataLoader
    - val_loader: val data in torch.utils.data.DataLoader
    - num_epochs: total number of training epochs
    - log_nth: logs training accuracy and loss every nth iteration
    - save_nth: saves current state every nth iteration
    - checkpoint: object used to resume training from a checkpoint
    """
    optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()),
                       **self.optim_args)
    scheduler = self.lrs(optim, **self.lrs_args)
    
    iter_per_epoch = len(train_loader)
    start_epoch = 0

    if len(checkpoint) > 0:
      start_epoch = checkpoint['epoch']
      optim.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      self._load_history()
      print("=> Loaded checkpoint (epoch {:d})".format(checkpoint['epoch']))
    else:
      self._save_checkpoint({
        'epoch': start_epoch,
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'scheduler': scheduler.state_dict()
      })

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
    #   [Epoch 1/5] TRAIN   loss: 1.374                                    #
    #   [Epoch 1/5] VAL acc/loss: 53.90%/1.310                             #
    #   ...                                                                #
    ########################################################################
    for epoch in range(start_epoch, num_epochs):
      # TRAINING
      model.train()
      scheduler.step()
      train_loss = 0

      for i, (inputs, targets) in enumerate(train_loader, 1):
        # Prepare data
        inputs, targets = inputs.to(device), targets.to(device)
        if model.log_transform:
          targets = targets.abs().add(1).log()

        # Forward pass
        optim.zero_grad()
        outputs = model(inputs)
        if self.args['mask']:
          mask = inputs[:,[1]].eq(1)
          outputs.masked_fill_(mask, 0)
          targets.masked_fill_(mask, 0)

        # Computes loss and backward pass
        loss = self.loss_func(outputs, targets)
        loss.backward()
        optim.step()

        self.train_loss_history.append(float(loss))
        if log_nth and i % log_nth == 0:
          last_log_nth_losses = self.train_loss_history[-log_nth:]
          train_loss = np.mean(last_log_nth_losses)
          print('[Iteration {:d}/{:d}] TRAIN loss: {:.2e}'
                .format(i + epoch * iter_per_epoch,
                  iter_per_epoch * num_epochs,
                  train_loss))

          if self.visdom:
            x = epoch + i / iter_per_epoch
            self.visdom.update_plot(x=x, y=train_loss,
                                    window=iter_plot,
                                    type_upd="append")

      # Free up memory
      del inputs, outputs, targets, mask, loss

      if log_nth:
        print('[Epoch {:d}/{:d}] TRAIN   loss: {:.2e}'.format(epoch + 1,
                                                              num_epochs,
                                                              train_loss))

      # VALIDATION
      if len(val_loader):
        val_err, val_loss = self.test(model, val_loader)
        self.val_loss_history.append(val_loss)

        if log_nth:
          print('[Epoch {:d}/{:d}] VAL loss/err: {:.2e}/{:.3f}'.format(epoch + 1,
                                                             num_epochs,
                                                             val_loss,
                                                             val_err))

      # CHECKPOINT
      if (save_nth and (epoch+1) % save_nth == 0) or (epoch+1) == num_epochs:
        self._save_checkpoint({
          'epoch': epoch + 1,
          'model': model.state_dict(),
          'optimizer': optim.state_dict(),
          'scheduler': scheduler.state_dict()
        })

        demo(model, '../datasets/test/test100.h5', epoch=epoch+1, n_samples=15)

  def test(self, model, data_loader):
    """
    Computes the loss for a given model with the provided data.

    Inputs:
    - model: model object initialized from a torch.nn.Module
    - data_loader: provided data in torch.utils.data.DataLoader
    """
    test_err = AverageMeter()
    test_loss = AverageMeter()
    model.eval()
    device = torch.device("cuda:0" if model.is_cuda else "cpu")

    with torch.no_grad():
      for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if model.log_transform:
          targets = targets.abs().add(1).log()
        
        outputs = model(inputs)
        if self.args['mask']:
          mask = inputs[:,[1]].eq(1)
          outputs.masked_fill_(mask, 0)
          targets.masked_fill_(mask, 0)

        loss = self.loss_func(outputs, targets)
        test_loss.update(float(loss))

        err = compute_l1_error(outputs,targets)
        test_err.update(float(err))

    return test_err.avg, test_loss.avg

  def _save_checkpoint(self, state, fname='checkpoint.pth'):
    """
    Saves current state of training.
    """
    print('Saving at checkpoint...')
    path = os.path.join(self.args['saveDir'], fname)
    torch.save(state, path)
    self._save_history()

  def _reset_history(self):
    """
    Resets train and val histories.
    """
    self.train_loss_history = []
    self.val_loss_history = []

  def _save_history(self, fname="train_history.npz"):
    """
    Saves training history. Conventionally the fname should end with "*.npz".
    """
    np.savez(os.path.join(self.args['saveDir'], fname),
            train_loss_history=self.train_loss_history,
            val_loss_history=self.val_loss_history)

  def _load_history(self, fname="train_history.npz"):
    """
    Loads training history. Conventionally the fname should end with "*.npz".
    """
    npzfile = np.load(os.path.join(self.args['saveDir'], fname))
    self.train_loss_history = npzfile['train_loss_history'].tolist()
    self.val_loss_history = npzfile['val_loss_history'].tolist()
