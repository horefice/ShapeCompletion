import numpy as np
import torch
import argparse
import os
import h5py
import matplotlib.pyplot as plt

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from nn import MyNet
from utils import isosurface

def main(argmodel, argfile, n_samples=1, epoch=0, savedir='', cb=None):
  device = torch.device('cpu')
  
  if isinstance(argmodel, str):
    model = MyNet()

    checkpoint = torch.load(argmodel, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
  else:
    model = argmodel
  model.to(device)

  if epoch is 0 and checkpoint is not None:
    epoch = checkpoint['epoch']

  N = 64 # >1
  with h5py.File(argfile) as file:
    inputs = torch.from_numpy(file['data'][()]).view(-1,2,32,32,32).float()

    inputs = inputs[:N]
    inputs[:,0].abs_().clamp_(max=3)
    try:
      targets = file['target'][:N].squeeze().float() # ([N],32,32,32)
    except Exception as e:
      targets = None

  with torch.no_grad():
    result = model(inputs)
    if model.log_transform:
      result = np.expm1(result)

  for n,i in enumerate(range(n_samples)):
    target = targets
    if targets is not None and targets.ndim > 3:
      target = targets[i]

    (plot_3d if cb is None else cb)(inputs.data.numpy()[i,0], result.data.numpy()[i,0], target,
              title='Demo - Epoch {:d}'.format(epoch), n=n_samples, i=n)

  plt.savefig(os.path.join(savedir, datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S") +'.png'))
    
def plot_3d(inputs, result, target=None, title='Demo', n=1, i=1):
  fig = plt.figure(title.partition("-")[0], figsize=(16,6*n))
  fig.suptitle(title)

  ax1 = fig.add_subplot(n,3,3*i+1, projection='3d')
  ax1.set_title('Input', y=1.1)
  verts1, faces1 = isosurface(inputs,1,1)
  ax1.plot_trisurf(verts1[:, 0], verts1[:,1], faces1, verts1[:, 2], lw=1, cmap="Spectral")
  ax1.view_init(elev=150, azim=-120)

  ax2 = fig.add_subplot(n,3,3*i+2, projection='3d')
  ax2.set_title('Prediction', y=1.1)
  verts2, faces2 = isosurface(result,1,1)
  ax2.plot_trisurf(verts2[:, 0], verts2[:,1], faces2, verts2[:, 2], lw=1, cmap="Spectral")
  ax2.view_init(elev=150, azim=-120)

  if target is not None:
    ax3 = fig.add_subplot(n,3,3*i+3, projection='3d')
    ax3.set_title('Target', y=1.1)
    verts3, faces3 = isosurface(target,1,1)
    ax3.plot_trisurf(verts3[:, 0], verts3[:,1], faces3, verts3[:, 2], lw=1, cmap="Spectral")
    ax3.view_init(elev=150, azim=-120)

    # Create translucid overlays between result and target
    ax2.plot_trisurf(verts3[:, 0], verts3[:,1], faces3, verts3[:, 2], lw=1, cmap="Spectral", alpha=0.2)
    ax3.plot_trisurf(verts2[:, 0], verts2[:,1], faces2, verts2[:, 2], lw=1, cmap="Spectral", alpha=0.2)

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