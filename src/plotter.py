import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('--input', type=str, default='../models/train_history.npz',
                    help='Path to file for plotting')
parser.add_argument('--title', type=str, default='',
                    help='Add info to title')
parser.add_argument('-n', '--smooth', type=int, default=1,
                    help='Use n previous values to smooth curve')
args = parser.parse_args()

class Plotter(object):
  """Loads and plots training history"""
  def __init__(self, path='../models/train_history.npz'):
    self.path = path
    self._load_histories()

  def _load_histories(self):
    """
    Load training history with its parameters to self.path.
    """
    npzfile = np.load(self.path)
    self.train_loss_history = npzfile['train_loss_history']
    self.val_acc_history = npzfile['val_acc_history']
    self.val_loss_history = npzfile['val_loss_history']

  def plot_histories(self, extra_title='', n_smoothed=1):
    """
    Plot losses and accuracies from training and validation. Also plots a 
    smoothed curve for train_loss.

    Inputs:
    - extra_title: extra string to be appended to plot's title
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.suptitle('Training history ' + extra_title)

    x_epochs = np.arange(1,len(self.val_loss_history)+1)*len(self.train_loss_history)/len(self.val_loss_history)

    ax1.set_yscale('log')
    ax1.plot(self.train_loss_history, label="train")
    ax1.plot(x_epochs,self.val_loss_history, label="validation", marker='x')
    if n_smoothed > 1:

      cumsum = np.cumsum(np.insert(self.train_loss_history, 0, 0))
      N = n_smoothed # Moving average size
      smoothed = (cumsum[N:] - cumsum[:-N]) / float(N)
      ax1.plot(smoothed, label="train_smoothed")
    ax1.legend()
    ax1.set_ylabel('loss')
    ax1.set_xlabel('batch')
    
    ax2.plot(np.arange(1,len(self.val_acc_history)+1),self.val_acc_history, label="validation", marker='x')
    ax2.legend()
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    
    plt.show();

if __name__ == '__main__':
  plotter = Plotter(args.input)
  plotter.plot_histories(args.title, args.smooth)
