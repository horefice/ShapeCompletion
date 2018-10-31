# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import argparse
import time
import os
import h5py
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from nn import MyNet

def main(argmodel, argfile, argcuda=False, timeit=False):
  model = MyNet()
  device = torch.device("cuda:0" if argcuda else "cpu")

  checkpoint = torch.load(argmodel, map_location=device)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  file = h5py.File(argfile, 'r')
  inputs = torch.from_numpy(file['data'][()]).unsqueeze(0).float()
  inputs[0,0].abs_().clamp_(max=3)
  target = None
  try:
    target = file['target'][0]
  except Exception as e:
    pass

  start = time.time()
  result = model.forward(inputs)

  if timeit:
    print('Prediction function took {:.2f} ms'.format((time.time()-start)*1000.0))

  create_plot(inputs.data.cpu().numpy()[0,0], result.data.cpu().numpy()[0,0], target,
              title='Demo - Epoch {:d}'.format(checkpoint['epoch']))


def isosurface(M,v,step):
    """
    returns vertices and faces from the isosurface of value v of M, subsetting M with the steps argument
    """
    sel = np.arange(0,np.shape(M)[0],step)
    verts, faces, _, _ = measure.marching_cubes_lewiner(M[np.ix_(sel,sel,sel)], v, spacing=(1.0, 1.0, 1.0))

    return verts, faces
    

def create_plot(inputs, result, target=None, title='Demo'):
  f = plt.figure(num=1, figsize=(20,10))
  f.suptitle(title)

  ax1 = f.add_subplot(131, projection='3d')
  ax1.set_title('Input')
  verts, faces = isosurface(inputs,1,1)
  ax1.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=1, cmap="Spectral")
  ax1.view_init(elev=150, azim=-120)

  ax2 = f.add_subplot(132, projection='3d')
  ax2.set_title('Prediction')
  verts, faces = isosurface(result,1,1)
  ax2.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=1, cmap="Spectral")
  ax2.view_init(elev=150, azim=-120)

  if target is not None:
    ax3 = f.add_subplot(133, projection='3d')
    ax3.set_title('Target')
    verts, faces = isosurface(target,1,1)
    ax3.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=1, cmap="Spectral")
    ax3.view_init(elev=150, azim=-120)

  f.canvas.draw()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Demo')
  parser.add_argument('--model', type=str, default='../models/checkpoint.pth',
                      help='Trained model path')
  parser.add_argument('--file', type=str, default='../datasets/sample/overfit.h5',
                      help='Use file as input')
  parser.add_argument('--no-live', action='store_true', default=False,
                      help='disables live updates')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA')
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  plt.ion()
  plt.show(False)

  main(args.model, args.file, args.cuda)
  cached = os.stat(args.model).st_mtime

  while not args.no_live:
    stamp = os.stat(args.model).st_mtime
    if stamp != cached:
      cached = stamp
      main(args.model, args.file, args.cuda)
      
    plt.pause(5)