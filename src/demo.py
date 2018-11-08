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
from utils import IndexTracker

idx = [0,12,34,61]

def main(argmodel, argfile, argcuda=False, argslicer=False, n_samples=1, timeit=False):
  model = MyNet()
  device = torch.device("cuda:0" if argcuda else "cpu")

  checkpoint = torch.load(argmodel, map_location=device)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  with h5py.File(argfile) as file:
    inputs = torch.from_numpy(file['data'][()]).view(-1,2,32,32,32).float()
    N = inputs.size(0)

    inputs = inputs[:64]
    inputs[:,0].abs_().clamp_(max=3)
    try:
      targets = file['target'][:64].squeeze() # ([N],32,32,32)
    except Exception as e:
      targets = None

  start = time.time()
  with torch.no_grad():
    result = model(inputs)

  if timeit:
    print('Prediction function took {:.2f} ms'.format((time.time()-start)*1000.0))

  for i in idx[:n_samples]:
    target = targets
    if targets is not None and targets.ndim > 3:
      target = targets[i]

    (plot_3d if not argslicer else plot_slicer)(inputs.data.cpu().numpy()[i,0], result.data.cpu().numpy()[i,0], target,
              title='Demo {:d} - Epoch {:d}'.format(i, checkpoint['epoch']), scale_down=(n_samples+1)//2)

def isosurface(M,v,step):
    """
    returns vertices and faces from the isosurface of value v of M, subsetting M with the steps argument
    """
    sel = np.arange(0,np.shape(M)[0],step)
    verts, faces, _, _ = measure.marching_cubes_lewiner(M[np.ix_(sel,sel,sel)], v, spacing=(1.0, 1.0, 1.0))

    return verts, faces
    
def plot_3d(inputs, result, target=None, title='Demo', scale_down=1):
  fig = plt.figure(title.partition("-")[0], figsize=(20/scale_down,10/scale_down))
  fig.suptitle(title)

  ax1 = fig.add_subplot(131, projection='3d')
  ax1.set_title('Input', y=1.1)
  verts1, faces1 = isosurface(inputs,1,1)
  ax1.plot_trisurf(verts1[:, 0], verts1[:,1], faces1, verts1[:, 2], lw=1, cmap="Spectral")
  ax1.view_init(elev=150, azim=-120)

  ax2 = fig.add_subplot(132, projection='3d')
  ax2.set_title('Prediction', y=1.1)
  verts2, faces2 = isosurface(result,1,1)
  ax2.plot_trisurf(verts2[:, 0], verts2[:,1], faces2, verts2[:, 2], lw=1, cmap="Spectral")
  ax2.view_init(elev=150, azim=-120)

  if target is not None:
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Target', y=1.1)
    verts3, faces3 = isosurface(target,1,1)
    ax3.plot_trisurf(verts3[:, 0], verts3[:,1], faces3, verts3[:, 2], lw=1, cmap="Spectral")
    ax3.view_init(elev=150, azim=-120)

    # Create translucid overlays between result and target
    ax2.plot_trisurf(verts3[:, 0], verts3[:,1], faces3, verts3[:, 2], lw=1, cmap="Spectral", alpha=0.2)
    ax3.plot_trisurf(verts2[:, 0], verts2[:,1], faces2, verts2[:, 2], lw=1, cmap="Spectral", alpha=0.2)

  fig.canvas.draw()

def plot_slicer(inputs, result, target=None, title='Demo', scale_down=1):
  fig = plt.figure(title.partition("-")[0], figsize=(20/scale_down,10/scale_down))
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

  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Demo')
  parser.add_argument('--model', type=str, default='../models/checkpoint.pth',
                      help='Trained model path')
  parser.add_argument('--input', type=str, default='../datasets/sample/overfit.h5',
                      help='Use file as input')
  parser.add_argument('--no-live', action='store_true', default=False,
                      help='disables live updates')
  parser.add_argument('--slicer', action='store_true', default=False,
                      help='use slicer instead of 3D plots')
  parser.add_argument('-n', '--n-samples', type=int, default=1,
                      help='plot n samples as figure')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA')
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  main(args.model, args.input, args.cuda, args.slicer, args.n_samples)
  cached = os.stat(args.model).st_mtime

  while True:
    stamp = os.stat(args.model).st_mtime
    if stamp != cached and not args.no_live:
      cached = stamp
      main(args.model, args.input, args.cuda, args.slicer, args.n_samples)
      
    plt.pause(8)