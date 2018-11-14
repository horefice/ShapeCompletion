import numpy as np
import visdom
import datetime
import os

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

class Viz(object):
  """Handles the Visdom connection and plots"""

  def __init__(self):
    self.viz = visdom.Visdom(port=8099)
    # self.viz.close(None) #Close all previously

  def create_plot(self, xlabel='', ylabel='', title='', opts_dict={}):
    options = dict(xlabel=xlabel,
      ylabel=ylabel,
      title=title)
    options.update(opts_dict)

    return self.viz.line(X=np.array([0]),
                         Y=np.array([0]),
                         opts=options)

  def update_plot(self, x, y, window, type_upd):
    self.viz.line(X=np.array([x]),
                  Y=np.array([y]),
                  win=window,
                  update=type_upd)

  def matplot(self, x):
    return self.viz.matplot(x)

class IndexTracker(object):
  """Plot 3D arrays as 2D images with scrollable slice selector"""
  def __init__(self, ax, X):
    self.ax = ax

    self.X = X
    rows, cols, self.slices = X.shape
    self.ind = self.slices//2

    self.im = ax.imshow(self.X[:, :, self.ind])
    self.update()

  def onscroll(self, event):
    if event.button == 'up':
        self.ind = (self.ind + 1) % self.slices
    elif event.button == 'down':
        self.ind = (self.ind - 1) % self.slices
    self.update()

  def update(self):
    self.im.set_data(self.X[:, :, self.ind])
    self.ax.set_ylabel('slice %s' % self.ind)
    self.im.axes.figure.canvas.draw()

def writeArgsFile(args,saveDir):
  os.makedirs(saveDir, exist_ok=True)
  args_list = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
  file_name = os.path.join(saveDir, 'args.txt')
  with open(file_name, 'a') as opt_file:
    opt_file.write('\n==> Args ('+datetime.datetime.now().isoformat()+'):\n')
    for k, v in sorted(args_list.items()):
       opt_file.write('  {}: {}\n'.format(str(k), str(v)))

def isosurface(M,v,step):
  """
  returns vertices and faces from the isosurface of value v of M, subsetting M with the steps argument
  """
  from skimage import measure

  sel = np.arange(0,np.shape(M)[0],step)
  verts, faces, _, _ = measure.marching_cubes_lewiner(M[np.ix_(sel,sel,sel)], v, spacing=(1.0, 1.0, 1.0))
  
  return verts, faces

'''
def get_random_idx(seed=1, len_samples=10000, samples=0):
  indices = list(range(len_samples))
  split = int(np.floor(0.2 * len_samples))

  np.random.seed(seed)
  np.random.shuffle(indices)

  train_idx, val_idx = indices[split:], indices[:split]

  if samples == 0:
    return train_idx, val_idx

  return train_idx[:samples], val_idx[:samples]
'''