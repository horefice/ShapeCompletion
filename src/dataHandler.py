import numpy as np
import torch
import h5py
import os

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.parallel.data_parallel import DataParallel


class MyDataset(Dataset):
    def __init__(self, path, truncation=3):
        super().__init__()
        self.truncation = truncation
        self.files = []

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.h5'):
                    self.files.append(os.path.join(root, file))

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
        return len(self.files)

    def get_item_from_index(self, index):
        try:
            with h5py.File(self.files[index], 'r', libver='latest') as file:
                data = torch.from_numpy(file['data'][:]).float()
                target = torch.from_numpy(file['target'][:]).float()
        except Exception as e:
            print(self.files[index])
            print(e)

        data[0].abs_().clamp_(max=self.truncation)
        target[0].clamp_(max=self.truncation)

        if data.shape[0] > 2:
            data[1:4].div_(255)
            target[1:].div_(255)

        #print(self.files[index], data.shape, target.shape)
        return data, target

    def subdivide_dataset(self, val_size, shuffle=False, seed=1):
        num_samples = int(len(self))
        indices = list(range(num_samples))
        split = int(np.floor(val_size * num_samples))

        if shuffle:
            # Shuffle indices
            np.random.seed(seed)
            np.random.shuffle(indices)

        # Split into training and validation
        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        return (train_sampler, val_sampler)


class MyDataParallel(DataParallel):
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    @property
    def is_cuda(self):
        return self.module.is_cuda

    @property
    def log_transform(self):
        return self.module.log_transform

    def state_dict(self):
        return self.module.state_dict()
