import numpy as np
import torch
import h5py
import os
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class DataHandler(Dataset):
  def __init__(self, path, truncation=3):
    super(DataHandler, self).__init__()
    self.truncation = truncation
    self.path = []
    self.size = []

    for root, dirs, files in os.walk(path):
      for file in files:
        if file.endswith('.h5'):
          path = os.path.join(root,file)
          self.path.append(path)

          with h5py.File(path) as h5_file:
            self.size.append(h5_file['data'][()].shape[0])
    self.size = np.cumsum(self.size)

  def __getitem__(self, key):
    if isinstance(key, slice):
      # get the start, stop, and step from the slice
      return [self[ii] for ii in range(*key.indices(len(self)))]
    elif isinstance(key, int):
      # handle negative indices
      if key < 0:
        key += len(self)
      if key < 0 or key >= len(self):
        raise IndexError("The index (%d) is out of range." % key)
      # get the data from direct index
      return self.get_item_from_index(key)
    else:
      raise TypeError("Invalid argument type.")

  def __len__(self):
    return self.size[-1]

  def get_item_from_index(self, index):
    file_idx = next(i for i,v in enumerate(self.size) if v > index)
    index = index - self.size[file_idx]

    with h5py.File(self.path[file_idx]) as h5_file:
      tsdf = torch.from_numpy(h5_file['data'][index]).float()
      tsdf[0].abs_().clamp_(max=self.truncation)
      target = torch.from_numpy(h5_file['target'][index]).clamp(max=self.truncation).float()

    return tsdf, target

  def subdivide_dataset(self, val_size, shuffle=False, seed=1):
    num_samples = int(len(self))
    indices = list(range(num_samples))
    split = int(np.floor(val_size * num_samples))

    if shuffle:
      np.random.seed(seed)
      np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return (train_sampler, val_sampler)