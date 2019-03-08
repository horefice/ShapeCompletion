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


def main(argcodenet, argmodel, argfile, upscale=1):
    device = torch.device('cpu')

    codenet = AENet(n_features=16)
    codenet.load_state_dict(torch.load(argcodenet, map_location=device)['model'])
    codenet.eval()
    codenet.to(device)

    model = DeepSDF(code_length=128, n_features=128)
    model.load_state_dict(torch.load(argmodel, map_location=device)['model'])
    model.eval()
    model.to(device)

    with h5py.File(argfile) as file:
        inputs = torch.from_numpy(file['data'][()]).view(5, 32, 32, 32).float()

        targets = file['target'][()].astype(np.float32)
        np.clip(targets[0], 0, 3, out=targets[0])  # (N,1||4,32,32,32)
        targets[1:4] /= 255

    with torch.no_grad():
        code = codenet(inputs[0:4].unsqueeze(0))

        results = torch.zeros(4, 32, 32, 32)
        mask = inputs[[-1]].eq(1).float()  # position of known values

        grid_res = range(upscale*32)
        for k in grid_res:
            for j in grid_res:
                for i in grid_res:
                    if mask[-1, i, j, k] == 1:
                        results[:, i, j, k] == inputs[0:4, i, j, k]
                    else:
                        cat = torch.cat([code.view(1,128), torch.Tensor([[i, j, k]])], dim=1)
                        result = model(cat)
                        #print(cat, result)
                        results[0, i, j, k] = result
                        results[1:4, i, j, k] = 0.8

    improveSDF(results[0])
    plot(inputs.data.numpy(), results.data.numpy(), targets)

def plot(inputs, results, targets):
    fig = plt.figure("Demo DeepSDF", figsize=(16, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Input', y=1.1)
    verts1, faces1, colors1 = colored_isosurface(inputs, 1, 1)
    coll1 = Poly3DCollection(verts1[faces1], facecolor=np.clip(colors1,0,1), linewidths=0.1, edgecolors='k')
    ax1.add_collection(coll1)
    ax1.view_init(elev=150, azim=-120)
    ax1.set_xlim(0,32)
    ax1.set_ylim(0,32)
    ax1.set_zlim(0,32)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Prediction', y=1.1)
    #verts2, faces2, colors2 = colored_isosurface(results, 1, 1)
    #coll2 = Poly3DCollection(verts2[faces2], facecolor=np.clip(colors2,0,1), linewidths=0.1, edgecolors='k')
    #ax2.add_collection(coll2)
    ax2.view_init(elev=150, azim=-120)
    ax2.set_xlim(0,32)
    ax2.set_ylim(0,32)
    ax2.set_zlim(0,32)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Target', y=1.1)
    verts3, faces3, colors3 = colored_isosurface(targets, 1, 1)
    coll3 = Poly3DCollection(verts3[faces3], facecolor=np.clip(colors3,0,1), linewidths=0.1, edgecolors='k')
    ax3.add_collection(coll3)
    ax3.view_init(elev=150, azim=-120)
    ax3.set_xlim(0,32)
    ax3.set_ylim(0,32)
    ax3.set_zlim(0,32)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slicer')
    parser.add_argument('--codenet', type=str, default='../models/AENet.pth',
                        help='trained codenet model path')
    parser.add_argument('--model', type=str, default='../models/DeepSDF.pth',
                        help='trained deepsdf model path')
    parser.add_argument('--input', type=str, default='../datasets/sample/overfit.h5',
                        help='uses file as input')
    parser.add_argument('--upscale', type=int, default=1,
                        help='uses specified upscale from 32')
    args = parser.parse_args()

    main(args.codenet, args.model, args.input, args.upscale)
