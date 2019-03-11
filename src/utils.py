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
        options = dict(xlabel=xlabel, ylabel=ylabel, title=title)
        options.update(opts_dict)

        return self.viz.line(X=np.array([0]), Y=np.array([0]), opts=options)

    def update_plot(self, x, y, window, name='1', type_upd='append'):
        self.viz.line(X=np.array([x]), Y=np.array([y]), win=window,
                      name=name, update=type_upd)

    def matplot(self, x):
        return self.viz.matplot(x)


class IndexTracker(object):
    """Plot 3D arrays as 2D images with scrollable slice selector"""

    def __init__(self, ax, X):
        self.ax = ax

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

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
        opt_file.write('\n==> Args (' + datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + '):\n')
        for k, v in sorted(args_list.items()):
            opt_file.write('  {}: {}\n'.format(str(k), str(v)))


def isosurface(M, v, step):
    """
    returns vertices and faces from the isosurface of value v of M,
    subsetting M with the steps argument
    """
    from skimage import measure

    sel = np.arange(0, np.shape(M)[0], step)
    m = M[np.ix_(sel, sel, sel)]
    verts, faces, _, _ = measure.marching_cubes_lewiner(m, v,
                                                        spacing=(1.0, 1.0, 1.0))

    return verts, faces


def colored_isosurface(M, v, step):
    verts, faces = isosurface(M[0], v, step)

    colors = []
    for face in faces:
        color = [0,0,0]
        for vert in face:
            voxel = [int(i) for i in verts[vert]]
            color += M[1:4,voxel[0],voxel[1],voxel[2]] # rgb
            #print(color)
        colors.append(np.append(color/3,[1])) # rgba
    return verts, faces, colors


def angelaEval(model, data_loader, progress_bar=False):
    import torch
    from tqdm import tqdm
    test_loss = AverageMeter()
    device = torch.device("cuda:0" if model.is_cuda else "cpu")
    pb = tqdm(total=len(data_loader), desc="EVAL", leave=progress_bar)

    model.eval()
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            # Prepare data
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Masked loss handling
            mask = inputs[:, [1]].eq(-1).float()  # position of unknown values
            outputs.mul_(mask)
            targets.mul_(mask)

            # Log-Transform handling
            if model.log_transform:
                targets.add_(1).log_()

            # Compute loss
            for i, sdf in enumerate(outputs.squeeze()):
                for z in range(sdf.shape[0]):
                    for y in range(sdf.shape[1]):
                        for x in range(sdf.shape[2]):
                            if inputs[i, 1, z, y, x] == -1 and (targets[i, 0, z, y, x] < 2.5 or outputs[i, 0, z, y, x] < 2.5):
                                diff = targets[i, 0, z, y, x].sub(outputs[i, 0, z, y, x]).abs()
                                test_loss.update(diff)

            # Update progress
            pb.set_postfix_str("x={:.2e}".format(diff))
            pb.update()

    pb.close()

    return test_loss.avg


def checkDistToNeighborAndUpdate(sdf, x, y, z, dim=32):
    kernelSize = 3
    foundBetter = False

    for i in range(-(kernelSize // 2), kernelSize // 2):
        for j in range(-(kernelSize // 2), kernelSize // 2):
            for k in range(-(kernelSize // 2), kernelSize // 2):
                if (k == 0 and j == 0 and i == 0) or \
                   (x + i not in range(dim)) or \
                   (y + j not in range(dim)) or \
                   (z + k not in range(dim)):
                    continue

                neighbor = sdf[x + i, y + j, z + k]
                d = np.linalg.norm((i, j, k))
                sgn = np.sign(neighbor)

                if sgn != 0:
                    dToN = neighbor + sgn * d
                    if np.abs(dToN) < np.abs(sdf[x, y, z]):
                        # print(x,y,z,i,j,k,v,dToN)
                        sdf[x, y, z] = dToN
                        foundBetter = True

    return foundBetter


def improveSDF(sdf, num_it=1, dim=32):
    for _ in range(num_it):
        hasUpdate = False
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    if checkDistToNeighborAndUpdate(sdf, x, y, z, dim):
                        hasUpdate = True
        if not hasUpdate:
            break

    return sdf

def sample_points(batch_size, N=100):
    return np.random.randint(0, high=31, size=(N, batch_size, 3))

def get_areas_and_vectors(verts, faces):
  areas = []
  anchor = verts[faces[:,0]]
  v1 = verts[faces[:,1]] - anchor
  v2 = verts[faces[:,2]] - anchor

  cross = np.cross(v1,v2)
  area = np.linalg.norm(cross, axis=1)/2

  return np.cumsum(area), [anchor,v1,v2]

def choose_random_faces(areas, n=1):
  random_v = np.random.uniform(areas[-1], size=(n,1))
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
