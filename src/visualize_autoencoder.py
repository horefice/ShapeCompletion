#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import h5py
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import isosurface, improveSDF, colored_isosurface

from nn import AENet, DeepSDF
from utils import AverageMeter, improveSDF, sample_points


def main(argmodel, argfile, upscale=1):
    device = torch.device('cpu')

    codenet = AENet(n_features=32)
    codenet.load_state_dict(torch.load(argmodel, map_location=device)['model'])
    codenet.eval()
    codenet.to(device)

    with h5py.File(argfile) as file:
        inputs = torch.from_numpy(file['target'][()]).view(1, 4, 32, 32, 32).float()

    with torch.no_grad():
        results, mu, logvar = codenet(inputs[:, 0:4])
    #print(torch.min(results), torch.max(results), inputs[0,0,13:18,13:18,13:18], results[0,0,13:18,13:18,13:18])
    improveSDF(results[0,0],3)
    plot(inputs.data.numpy()[0], results.abs().clamp(max=3).data.numpy()[0])

def plot(inputs, results):
    fig = plt.figure("Demo AutoEncoder", figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Input/Output', y=1.1)
    verts1, faces1, colors1 = colored_isosurface(inputs, 1, 1)
    coll1 = Poly3DCollection(verts1[faces1], facecolor=np.clip(colors1,0,1), linewidths=0.1, edgecolors='k')
    ax1.add_collection(coll1)
    ax1.view_init(elev=150, azim=-120)
    ax1.set_xlim(0,32)
    ax1.set_ylim(0,32)
    ax1.set_zlim(0,32)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Prediction', y=1.1)
    verts2, faces2, colors2 = colored_isosurface(results, 1, 1)
    coll2 = Poly3DCollection(verts2[faces2], facecolor=np.clip(colors2,0,1), linewidths=0.1, edgecolors='k')
    ax2.add_collection(coll2)
    ax2.view_init(elev=150, azim=-120)
    ax2.set_xlim(0,32)
    ax2.set_ylim(0,32)
    ax2.set_zlim(0,32)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slicer')
    parser.add_argument('--model', type=str, default='../models/AENet.pth',
                        help='trained codenet model path')
    parser.add_argument('--input', type=str, default='../datasets/sample/overfit.h5',
                        help='uses file as input')
    parser.add_argument('--upscale', type=int, default=1,
                        help='uses specified upscale from 32')
    args = parser.parse_args()

    main(args.model, args.input, args.upscale)
