import numpy as np
import torch
import os

from tqdm import tqdm
from demo import main as demo
from utils import AverageMeter, Viz


class Solver(object):
    default_args = {'saveDir': '../models/',
                    'visdom': False,
                    'mask': False}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 lrs=torch.optim.lr_scheduler.StepLR, lrs_args={},
                 loss_func=torch.nn.L1Loss(reduction='sum'), args={}):
        self.optim_args = optim_args
        self.optim = optim
        self.lrs_args = lrs_args
        self.lrs = lrs
        self.loss_func = loss_func
        self.args = dict(self.default_args, **args)
        self.visdom = Viz() if self.args['visdom'] else False
        self._reset_history()

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0,
              save_nth=0, sub_epochs=0, checkpoint={}):
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
        optim = self.optim(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           **self.optim_args)
        scheduler = self.lrs(optim, **self.lrs_args)

        iter_per_epoch = len(train_loader)
        start_epoch = 0

        if len(checkpoint) > 0:
            start_epoch = checkpoint['epoch']
            optim.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            self._load_history(checkpoint)
            print("=> Loaded checkpoint (epoch {:d})"
                  .format(checkpoint['epoch']))
        else:
            self._save_checkpoint({'epoch': start_epoch,
                                   'model': model.state_dict(),
                                   'optimizer': optim.state_dict(),
                                   'scheduler': scheduler.state_dict()
                                   }, False)

        device = torch.device("cuda:0" if model.is_cuda else "cpu")

        if self.visdom:
            iter_plot = self.visdom.create_plot('Epoch', 'Loss',
                                                'Training History',
                                                {'ytype': 'log'})

        #######################################################################
        # The log should like something like:                                 #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN loss: 1.374                                     #
        #   [Epoch 1/5] VAL   loss: 1.310                                     #
        #   ...                                                               #
        #######################################################################
        for epoch in range(start_epoch, num_epochs):
            scheduler.step()

            # TRAINING
            for i, (inputs, targets) in enumerate(train_loader, 1):
                # Prepare data
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                model.train()
                optim.zero_grad()
                outputs = model(inputs)

                # Masked loss handling
                if self.args['mask']:
                    mask = inputs[:, [1]].eq(-1).float()  # unknown values
                    outputs.mul_(mask)
                    targets.mul_(mask)

                # Log-Transform handling
                if model.log_transform:
                    targets.add_(1).log_()

                # Compute loss and backward pass
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                # Update progress
                batch_loss = float(loss)
                batch_loss /= mask.sum().item() if self.args['mask'] else mask.numel()
                self.train_loss_history.append(batch_loss)

                # Logging iteration
                if log_nth and i % log_nth == 0:
                    mean_nth_loss = np.mean(self.train_loss_history[-log_nth:])
                    print('[Iteration {:d}/{:d}] TRAIN loss: {:.2e}'
                          .format(i + epoch * iter_per_epoch,
                                  iter_per_epoch * num_epochs,
                                  mean_nth_loss))

                    if self.visdom:
                        x = epoch + i / iter_per_epoch
                        self.visdom.update_plot(x=x, y=mean_nth_loss,
                                                window=iter_plot)

                # VALIDATION
                if i % (iter_per_epoch / (sub_epochs + 1)) < 1:
                    sub_val_loss = self.eval(model, val_loader)
                    self.val_loss_history.append(sub_val_loss)

                    if log_nth:
                        print('[Iteration {:d}/{:d}] VAL   loss: {:.2e}'
                              .format(i + epoch * iter_per_epoch,
                                      iter_per_epoch * num_epochs,
                                      sub_val_loss))

                        if self.visdom:
                            x = epoch + i / iter_per_epoch
                            self.visdom.update_plot(x=x, y=sub_val_loss,
                                                    window=iter_plot,
                                                    name='val')

            # Free up memory
            del inputs, outputs, targets, mask, loss

            # Epoch logging
            train_loss = self.train_loss_history[-1]
            print('[Epoch {:d}/{:d}] TRAIN loss: {:.2e}'.format(epoch + 1,
                                                                num_epochs,
                                                                train_loss))
            val_loss = self.val_loss_history[-1]
            print('[Epoch {:d}/{:d}] VAL   loss: {:.2e}'.format(epoch + 1,
                                                                num_epochs,
                                                                val_loss))

            # Checkpoint
            if (save_nth and (epoch + 1) % save_nth == 0) or \
               (epoch + 1) == num_epochs:
                self._save_checkpoint({'epoch': epoch + 1,
                                       'model': model.state_dict(),
                                       'optimizer': optim.state_dict(),
                                       'scheduler': scheduler.state_dict()
                                       }, True)

                # demo(model, '../datasets/test100.h5', epoch=epoch + 1,
                #     n_samples=15, savedir=self.args['saveDir'])

    def eval(self, model, data_loader, threshold=2.5, progress_bar=False):
        """
        Compute the loss for a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - data_loader: provided data in torch.utils.data.DataLoader
        - threshold: sets the boundary between object and empty space
        - progress_bar: boolean for leaving the progress bar after return
        """
        test_loss = AverageMeter()
        device = torch.device("cuda:0" if model.is_cuda else "cpu")
        pb = tqdm(total=len(data_loader), desc="EVAL", leave=progress_bar)

        model.eval()
        with torch.no_grad():
            for (inputs, targets) in data_loader:
                # Prepare data
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Masked loss handling
                if self.args['mask']:
                    unknown = inputs[:, [1]].eq(-1)  # unknown values
                    close1 = outputs[:, [0]].le(threshold)
                    close2 = targets[:, [0]].le(threshold)
                    mask = (unknown & (close1 | close2)).float()

                    outputs.mul_(mask)
                    targets.mul_(mask)

                # Log-Transform handling
                if model.log_transform:
                    targets.add_(1).log_()

                # Compute loss
                batch_loss = float(self.loss_func(outputs, targets))
                batch_loss /= mask.sum().item() if self.args['mask'] else mask.numel()
                test_loss.update(batch_loss, n=targets.size(0))

                # Update progress
                pb.set_postfix_str("x={:.2e}".format(batch_loss))
                pb.update()

        pb.close()

        return test_loss.avg

    def _save_checkpoint(self, state, overwrite=False, fname='checkpoint.pth'):
        """
        Save current state of training.
        """
        path = os.path.join(self.args['saveDir'], fname)
        if not overwrite and os.path.isfile(path):
            return

        print('Saving at checkpoint...')
        self._save_history(state)
        torch.save(state, path)

    def _reset_history(self):
        """
        Reset train and val histories.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def _save_history(self, checkpoint):
        """
        Save training history.
        """
        checkpoint.update(train_loss_history=self.train_loss_history,
                          val_loss_history=self.val_loss_history)

    def _load_history(self, checkpoint):
        """
        Load training history.
        """
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_history = checkpoint['val_loss_history']
