import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils import IndexTracker
from demo import main


def plot_slicer(inputs, result, target=None, **kargs):
    fig = plt.figure('Slicer', figsize=(20, 10))
    fig.suptitle('use scroll wheel to navigate images')

    ax1 = fig.add_subplot(221)
    ax1.set_title('Input')
    tracker1 = IndexTracker(ax1, inputs[0])
    fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)

    ax2 = fig.add_subplot(222)
    ax2.set_title('Mask')
    mask_input = inputs[1]
    mask_input[np.where(np.logical_and(mask_input == -1,
                                       np.logical_or(result >= 2.5,
                                                     target >= 2.5)))] = 0
    tracker2 = IndexTracker(ax2, mask_input)
    fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)

    ax3 = fig.add_subplot(223)
    ax3.set_title('Prediction')
    tracker3 = IndexTracker(ax3, result)
    fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)

    if target is not None:
        ax4 = fig.add_subplot(224)
        ax4.set_title('Target')
        tracker4 = IndexTracker(ax4, target)
        fig.canvas.mpl_connect('scroll_event', tracker4.onscroll)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slicer')
    parser.add_argument('--model', type=str, default='../models/checkpoint.pth',
                        help='trained model path')
    parser.add_argument('--input', type=str, default='../datasets/sample/overfit.h5',
                        help='uses file as input')
    args = parser.parse_args()

    main(args.model, args.input, n_samples=1, cb=plot_slicer)
