import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from datetime import datetime
from utils import IndexTracker
from demo import main


parser = argparse.ArgumentParser(description='Slicer')
parser.add_argument('--model', type=str, default='../models/checkpoint.pth',
                    help='Trained model path')
parser.add_argument('--input', type=str, default='../datasets/sample/overfit.h5',
                    help='Use file as input')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def plot_slicer(inputs, result, target=None, **kargs):
  fig = plt.figure('Slicer', figsize=(20,10))
  fig.suptitle('use scroll wheel to navigate images')

  ax1 = fig.add_subplot(131)
  ax1.set_title('Input', y=1.1)
  tracker1 = IndexTracker(ax1, inputs)
  fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)

  ax2 = fig.add_subplot(132)
  ax2.set_title('Prediction', y=1.1)
  tracker2 = IndexTracker(ax2, result)
  fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)

  if target is not None:
    ax3 = fig.add_subplot(133)
    ax3.set_title('Target', y=1.1)
    tracker3 = IndexTracker(ax3, np.log(np.add(np.abs(target),1)))
    fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)

  plt.savefig(datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S") +'.png')
  plt.show()
  return 0

main(args.model, args.input, args.cuda, 1, cb=plot_slicer)
