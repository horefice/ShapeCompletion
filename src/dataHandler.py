import numpy as np
import torch
import h5py
import os
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

class DataHandler(Dataset):
  def __init__(self, path, truncation=3):
    super(DataHandler, self).__init__()
    self.truncation = truncation
    self.file_idx = -1
    self.data = None
    self.target = None
    self.shape = None
    self.files = []
    self.size = []

    for root, _, files in os.walk(path):
      for file in files:
        if file.endswith('15.h5'):
          path = os.path.join(root,file)
          self.files.append(path)

          with h5py.File(path) as h5_file:
            self.size.append(len(h5_file['data']))

            if self.shape is None:
              self.shape = h5_file['data'].shape[1:]
            elif self.shape != h5_file['data'].shape[1:]:
              raise ValueError("Invalid dataset shape across files")
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

    if self.file_idx != file_idx:
      self.file_idx = file_idx
      h5_file = h5py.File(self.files[self.file_idx], 'r', libver='latest', swmr=True)
      self.data = torch.from_numpy(h5_file['data'][()])
      self.target = torch.from_numpy(h5_file['target'][()])

    index = index - self.size[file_idx]

    tsdf = self.data[index].float()
    tsdf[0].abs_().clamp_(max=self.truncation)
    target = self.target[index].clamp(max=self.truncation).float()

    return tsdf, target

  def subdivide_dataset(self, val_size, shuffle=False, seed=1):
    num_samples = int(len(self))
    indices = list(range(num_samples))
    split = int(np.floor(val_size * num_samples))

    if shuffle:
      # Shuffle indices
      np.random.seed(seed)
      np.random.shuffle(indices)

      # Shuffle groups
      groups = list(range(len(self.size)))
      np.random.shuffle(groups)

      # Undo cumsum for size
      group_size = self.size.copy()
      group_size[1:] -= group_size[:-1]

      # New cumsum size after shuffle
      size = np.cumsum([group_size[group] for group in groups])
      size = np.insert(size,0,0)

      # Initilize mapping and indices
      mapping = {self.size[v]: size[i] for i,v in enumerate(groups)}
      new_indices = [0] * num_samples

      # Loop through indices to find to which group it belongs to 
      # and assign its value to the new arraz based on the mapping
      for index, value in enumerate(indices):
        group_id = next(i for i,v in enumerate(self.size) if v > value)
        new_indices[mapping[self.size[group_id]]] = indices[index]
        mapping[self.size[group_id]] += 1
    else:
      new_indices = indices

    # Split into training and validation
    train_idx, val_idx = new_indices[:split], new_indices[split:]

    train_sampler = SubsetSequentialSampler(train_idx)
    val_sampler = SubsetSequentialSampler(val_idx)

    return (train_sampler, val_sampler)

class SubsetSequentialSampler(Sampler):
  def __init__(self, indices):
    self.indices = indices

  def __iter__(self):
    return (self.indices[i] for i in range(len(self.indices)))

  def __len__(self):
    return len(self.indices)