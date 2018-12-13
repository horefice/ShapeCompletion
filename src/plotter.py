import torch
import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    """Loads and plots training history"""
    def __init__(self, path='../models/checkpoint.pth'):
        try:
            self.checkpoint = torch.load(path, map_location=torch.device('cpu'))
        except Exception as e:
            try:
                self.checkpoint = np.load(path)
            except Exception as e:
                raise e

    def _load_histories(self):
        """
        Loads training history with its parameters to self.path.
        """
        try:
            self.train_loss_history = self.checkpoint['train_loss_history']
            self.val_loss_history = self.checkpoint['val_loss_history']
            self.epoch = self.checkpoint['epoch']
        except KeyError:
            raise ("Keys for training history not found.")

    def plot_histories(self, extra_title='', n_smoothed=1):
        """
        Plots losses from training and validation. Also plots a
        smoothed curve for train_loss.

        Inputs:
        - extra_title: extra string to be appended to plot's title
        - n_smoothed: moving average length for smoothed curve
        """
        self._load_histories()

        f, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
        f.suptitle('Training history ' + extra_title)

        x_train = np.arange(1, len(self.train_loss_history) + 1) * self.epoch / len(self.train_loss_history)
        ax1.plot(x_train, self.train_loss_history, label="train_loss")
        ax1.set_yscale('log')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')

        x_val = np.arange(1, len(self.val_loss_history) + 1) * self.epoch / len(self.val_loss_history)
        ax1.plot(x_val, self.val_loss_history, label="validation_loss", marker='x')

        if n_smoothed > 1:
            cumsum = np.cumsum(np.insert(self.train_loss_history, 0, 0))
            N = n_smoothed  # Moving average size
            smoothed = (cumsum[N:] - cumsum[:-N]) / float(N)
            ax1.plot(np.arange(n_smoothed, len(smoothed) + n_smoothed), smoothed, label="train_loss_MA")

        epochs_line = 20
        for epoch in range(epochs_line, self.epoch, epochs_line):
            ax1.axvline(x=epoch, color='gray', linestyle='-.', alpha=0.5)

        ax1.legend()

    def plot_histogram(self, n_bins=20):
        from matplotlib import colors
        from matplotlib.ticker import PercentFormatter
        from nn import MyNet

        def flatten_weights(net):
            weights = []
            for p in net.parameters():
                for weight in p.data.cpu().numpy().flatten():
                    weights.append(weight)
            return weights

        def compute_bins(data, bin_size):
            min_val = np.min(data)
            max_val = np.max(data)
            min_bound = -1.0 * (min_val % bin_size - min_val)
            max_bound = max_val - max_val % bin_size + bin_size
            num_bins = int((max_bound - min_bound) / bin_size) + 1
            bins = np.linspace(min_bound, max_bound, num_bins)
            return bins

        def set_color(N, patches):
            fracs = N / N.max()
            norm = colors.Normalize(fracs.min(), fracs.max())
            for thisfrac, thispatch in zip(fracs, patches):
                color = plt.cm.viridis(norm(thisfrac))
                thispatch.set_facecolor(color)

        net = MyNet()
        init = flatten_weights(net)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        N, _, patches = ax1.hist(init, bins=compute_bins(init, 1 / n_bins), histtype='bar', ec='black')
        set_color(N, patches)
        ax1.set_title("Initialization")
        ax1.set_ylabel("# of parameters")
        ax1.set_xlabel("parameter value")

        try:
            net.load_state_dict(self.checkpoint['model'])
        except KeyError:
            return

        final = flatten_weights(net)
        bins = compute_bins(final, 1 / n_bins)

        N, _, patches = ax2.hist(final, bins=bins, density=True, histtype='bar', ec='black')
        set_color(N, patches)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=n_bins))
        ax2.set_title("Loaded Network")
        ax2.set_ylabel("parameter density")
        ax2.set_xlabel("parameter value")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plotter')
    parser.add_argument('input', type=str, nargs='?', default='../models/checkpoint.pth',
                        help='path to file for plotting')
    parser.add_argument('--title', type=str, default='',
                        help='adds info to title')
    parser.add_argument('-n', '--smooth', type=int, default=1,
                        help='uses n previous values to smooth loss curve')
    parser.add_argument('--bins', type=int, default=200,
                        help='uses n bins for histogram')
    parser.add_argument('--histogram', action='store_true',
                        help='plots network histogram instead')
    args = parser.parse_args()

    plotter = Plotter(args.input)

    if args.histogram:
        plotter.plot_histogram(args.bins)
    else:
        plotter.plot_histories(args.title, args.smooth)

    plt.show()
