import numpy as np
import torch
import h5py
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class DataHandler(Dataset):
  def __init__(self, path, truncation=3):
    super(DataHandler, self).__init__()
    file = h5py.File(path, 'r')
    self.data = file['data']
    self.target = file['target']
    self.truncation = truncation

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
    return len(self.data)

  def get_item_from_index(self, index):
    to_tensor = transforms.ToTensor()
    vol = torch.from_numpy(self.data[index]).float()
    vol[0].abs_().clamp_(max=self.truncation)
    vol.requires_grad_()
    target = torch.from_numpy(self.target[index]).clamp(max=self.truncation).float()

    return vol, target

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