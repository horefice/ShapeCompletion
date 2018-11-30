import numpy as np
import visdom
import datetime
import math
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

def writeArgsFile(args, saveDir):
  os.makedirs(saveDir, exist_ok=True)
  args_list = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
  file_name = os.path.join(saveDir, 'args.txt')
  with open(file_name, 'a') as opt_file:
    opt_file.write('\n==> Args ('+datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")+'):\n')
    for k, v in sorted(args_list.items()):
       opt_file.write('  {}: {}\n'.format(str(k), str(v)))

def isosurface(M, v, step):
  """
  returns vertices and faces from the isosurface of value v of M, subsetting M with the steps argument
  """
  from skimage import measure

  sel = np.arange(0,np.shape(M)[0],step)
  verts, faces, _, _ = measure.marching_cubes_lewiner(M[np.ix_(sel,sel,sel)], v, spacing=(1.0, 1.0, 1.0))
  
  return verts, faces

def get_areas_and_vectors(verts, faces):
  areas = []
  anchor = verts[faces[:,0]]
  v1 = verts[faces[:,1]] - anchor
  v2 = verts[faces[:,2]] - anchor

  cross = np.cross(v1,v2)
  area = np.linalg.norm(cross, axis=1)/2

  return np.cumsum(area), [anchor,v1,v2]

def choose_random_faces(areas, n=1):
  random_v = np.random.uniform(areas[-1], size=(n))
  random_i = [np.searchsorted(areas,v) for v in random_v]

  return random_i

def sample_triangle_uniform(v, faces):

  samples = []
  for face in faces:
    a1=a2=1
    while a1+a2 > 1:
      a1 = np.random.uniform(0,1)
      a2 = np.random.uniform(0,1)
    x = v[0][face] + a1*v[1][face] + a2*v[2][face]
    samples.append(x)

  return samples

def compute_distance(samples, df):
  from scipy.interpolate import RegularGridInterpolator

  dist = AverageMeter()
  grid = np.linspace(0,31,32)
  interpolator = RegularGridInterpolator((grid,grid,grid), df, method='linear')

  for v in interpolator(samples):
    if v is not None:
      dist.update(np.asscalar(v))

  return dist.avg

def compute_l1_error(inputs, targets, n=1):
  inputs, targets = inputs.data.cpu().numpy(), targets.data.cpu().numpy()

  skipped = 0
  err = AverageMeter()
  for i,df in enumerate(inputs):
    try:
      verts, faces = isosurface(df[0], 1, 1)
    except:
      skipped += 1
      continue

    areas, v = get_areas_and_vectors(verts, faces)
    random_faces = choose_random_faces(areas, n=n)
    samples = sample_triangle_uniform(v, random_faces)
    dists = compute_distance(samples, targets[i,0])
    err.update(dists)

  if skipped > 1:
    print("[Warning]: Skipped samples due to lack of surfaces: {:d}".format(skipped))

  return err.avg
