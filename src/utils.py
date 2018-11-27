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

  def item(self):
    return self.avg

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

def get_areas(verts, faces):
  areas = []
  for face in faces:
    anchor = verts[face[0]]
    v1 = anchor - verts[face[1]]
    v2 = anchor - verts[face[2]]

    area = np.linalg.norm(np.cross(v1,v2))/2
    areas.append(area)

  return areas

def choose_random_face(verts, faces):
  areas = get_areas(verts, faces)
  cumsum = np.cumsum(areas)
  random_v = np.random.uniform(cumsum[-1])
  random_i = np.searchsorted(cumsum,random_v)

  return faces[random_i]

def sample_triangle_uniform(verts, face, n=1):
  anchor = verts[face[0]]
  v1 = verts[face[1]] - anchor
  v2 = verts[face[2]] - anchor

  samples = []
  for _ in range(n):
    a1=a2=1
    while a1+a2 > 1:
      a1 = np.random.uniform(0,1)
      a2 = np.random.uniform(0,1)
    x = anchor + a1*v1 + a2*v2
    samples.append(x)

  return samples

def compute_distance(samples, df):
  from scipy.interpolate import RegularGridInterpolator

  dist = AverageMeter()
  grid = np.linspace(0,31,32)
  interpolator = RegularGridInterpolator((grid,grid,grid), df, method='linear')

  for v in interpolator(samples):
    if v is not None:
      dist.update(v)

  return dist.avg

def compute_l1_error(inputs, targets, n=1):
  inputs, targets = inputs.data.cpu().numpy(), targets.data.cpu().numpy()

  '''
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure(1, figsize=(15,10))
  ax1 = fig.add_subplot(111, projection='3d')
  ax1.view_init(elev=150, azim=-120)
  '''

  skipped = 0
  err = AverageMeter()
  for i,df in enumerate(inputs):
    try:
      verts, faces = isosurface(df[0], 1, 1)
    except:
      skipped += 1
      continue
    #ax1.plot_trisurf(verts[:,0],verts[:,1],faces,verts[:,2], lw=1, cmap="Spectral")
    #plt.show()
    for _ in range(n):
      face = choose_random_face(verts, faces)
      samples = sample_triangle_uniform(verts, face)
      dist = compute_distance(samples, targets[i,0])
      err.update(dist)

  if skipped > 1:
    print("Skipped samples due to lack of surfaces: {:d}".format(skipped))

  return err
