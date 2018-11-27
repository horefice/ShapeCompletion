import numpy as np
import matplotlib.pyplot as plt

class Plotter(object):
  """Loads and plots training history"""
  def __init__(self, path='../models/train_history.npz'):
    self.path = path
    self._load_histories()

  def _load_histories(self):
    """
    Loads training history with its parameters to self.path.
    """
    npzfile = np.load(self.path)
    self.train_loss_history = npzfile['train_loss_history']
    self.val_loss_history = npzfile['val_loss_history']

  def plot_histories(self, extra_title='', n_smoothed=1):
    """
    Plots losses from training and validation. Also plots a 
    smoothed curve for train_loss.

    Inputs:
    - extra_title: extra string to be appended to plot's title
    - n_smoothed: moving average length for smoothed curve
    """
    f, (ax1) = plt.subplots(1, 1, figsize=(20,10))
    f.suptitle('Training history ' + extra_title)

    x_epochs = np.arange(1,len(self.val_loss_history)+1)*len(self.train_loss_history)/len(self.val_loss_history)

    ax1.set_yscale('log')
    ax1.plot(np.arange(1,len(self.train_loss_history)+1), self.train_loss_history, label="train")
    ax1.plot(x_epochs,self.val_loss_history, label="validation", marker='x')
    if n_smoothed > 1:

      cumsum = np.cumsum(np.insert(self.train_loss_history, 0, 0))
      N = n_smoothed # Moving average size
      smoothed = (cumsum[N:] - cumsum[:-N]) / float(N)
      ax1.plot(np.arange(n_smoothed,len(smoothed)+n_smoothed), smoothed, label="train_smoothed")
    ax1.legend()
    ax1.set_ylabel('loss')
    ax1.set_xlabel('batch')
    
    plt.show();

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Plotter')
  parser.add_argument('--input', type=str, default='../models/train_history.npz',
                      help='Path to file for plotting')
  parser.add_argument('--title', type=str, default='',
                      help='Add info to title')
  parser.add_argument('-n', '--smooth', type=int, default=1,
                      help='Use n previous values to smooth curve')
  args = parser.parse_args()

  plotter = Plotter(args.input)
  plotter.plot_histories(args.title, args.smooth)
