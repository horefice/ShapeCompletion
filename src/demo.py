import numpy as np
import torch
import argparse
import os
import h5py
import matplotlib.pyplot as plt

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import isosurface, improveSDF, colored_isosurface
from nn import MyNet


def main(argmodel, argfile, n_samples=1, epoch=0, savedir=None, cb=None):
    device = torch.device('cpu')

    if isinstance(argmodel, str):
        model = MyNet()

        checkpoint = torch.load(argmodel, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        model = argmodel

    model.log_transform = False
    model.to(device)

    if epoch is 0 and checkpoint is not None:
        epoch = checkpoint['epoch']

    N = 64  # >1
    with h5py.File(argfile) as file:
        inputs = torch.from_numpy(file['data'][()]).view(-1, model.channels, 32, 32, 32)

        inputs = inputs[:N].float()
        inputs[:, 0].abs_().clamp_(max=3)
        if inputs.shape[1] > 2:
            inputs[:, 1:4].div_(255)

        try:
            targets = file['target'][:N].astype(np.float32)
            np.clip(targets[:, 0], 0, 3, out=targets[:, 0])  # (N,1||4,32,32,32)
            if len(targets.shape) > 4:
                targets[:, 1:] /= 255
            else:
                targets[1:] /= 255
        except Exception as e:
            print(e)
            targets = None


    with torch.no_grad():
        result = model(inputs)
        mask = inputs[:, [-1]].eq(1)  # position of known values
        result = result.mul((~mask).float()) + inputs[:,:4].mul(mask.float())

    for n, i in enumerate(range(n_samples)):
        improveSDF(result[i, 0])

        target = targets
        if targets is not None and targets.ndim > 4:
            target = targets[i]

        (plot_3d if cb is None else cb)(inputs.data.numpy()[i],
                                        result.data.numpy()[i],
                                        target, n=n_samples, i=n,
                                        title='Demo - Epoch {:d}'.format(epoch))

    if savedir is not None:
        filename = datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S") + '.png'
        plt.savefig(os.path.join(savedir, filename))


def plot_3d(inputs, result, target=None, title='Demo', n=1, i=1):
    fig = plt.figure(title.partition("-")[0], figsize=(16, 6 * n))
    fig.suptitle(title)

    ax1 = fig.add_subplot(n, 3, 3 * i + 1, projection='3d')
    ax1.set_title('Input', y=1.1)
    verts1, faces1, colors1 = colored_isosurface(inputs, 1, 1)
    coll1 = Poly3DCollection(verts1[faces1], facecolor=colors1, linewidths=0.1, edgecolors='k')
    ax1.add_collection(coll1)
    ax1.view_init(elev=150, azim=-120)
    ax1.set_xlim(0,32)
    ax1.set_ylim(0,32)
    ax1.set_zlim(0,32)

    ax2 = fig.add_subplot(n, 3, 3 * i + 2, projection='3d')
    ax2.set_title('Prediction', y=1.1)
    verts2, faces2, colors2 = colored_isosurface(result, 1, 1)
    coll2 = Poly3DCollection(verts2[faces2], facecolor=np.clip(colors2,0,1), linewidths=0.1, edgecolors='k')
    ax2.add_collection(coll2)
    ax2.view_init(elev=150, azim=-120)
    ax2.set_xlim(0,32)
    ax2.set_ylim(0,32)
    ax2.set_zlim(0,32)

    if target is not None:
        ax3 = fig.add_subplot(n, 3, 3 * i + 3, projection='3d')
        ax3.set_title('Target', y=1.1)
        verts3, faces3, colors3 = colored_isosurface(target, 1, 1)
        coll3 = Poly3DCollection(verts3[faces3], facecolor=colors3, linewidths=0.1, edgecolors='k')
        ax3.add_collection(coll3)
        ax3.view_init(elev=150, azim=-120)
        ax3.set_xlim(0,32)
        ax3.set_ylim(0,32)
        ax3.set_zlim(0,32)

        # Create translucid overlays between result and target
        #ax2.plot_trisurf(verts3[:, 0], verts3[:, 1], faces3, verts3[:, 2],
        #                 lw=1, cmap="Spectral", alpha=0.2)
        #ax3.plot_trisurf(verts2[:, 0], verts2[:, 1], faces2, verts2[:, 2],
        #                 lw=1, cmap="Spectral", alpha=0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--model', type=str, default='../models/checkpoint.pth',
                        help='trained model path')
    parser.add_argument('--input', type=str, default='../datasets/sample/overfit.h5',
                        help='uses file as input')
    parser.add_argument('-n', '--n-samples', type=int, default=1,
                        help='plots n samples as figure')
    parser.add_argument('--no-live', action='store_true',
                                            help='disables live updates')
    parser.add_argument('--no-plot', action='store_true',
                                            help='disables plots (only saves)')
    args = parser.parse_args()

    main(args.model, args.input, args.n_samples)
    if args.no_plot:
        quit()

    cached = os.stat(args.model).st_mtime
    while not args.no_live:
        stamp = os.stat(args.model).st_mtime

        if stamp != cached:
            cached = stamp
            main(args.model, args.input, args.n_samples)

        plt.pause(8)

    plt.show()
