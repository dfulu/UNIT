"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from sklearn.model_selection import train_test_split as _array_train_test_split
import xarray as xr
import numpy as np
import xesmf as xe
import torch

def construct_regridders(ds_a, ds_b):

    ds_out = xr.Dataset({'lat': min([ds_a.lat, ds_b.lat], key=lambda x: len(x)), 
                         'lon': min([ds_a.lon, ds_b.lon], key=lambda x: len(x))})
    
    if not ds_out.equals(ds_a[['lat', 'lon']]):
        regridder_a = xe.Regridder(ds_a, ds_out, 'bilinear', periodic=True)
        regridder_a.clean_weight_file()
    else: 
        regridder_a = None
        
    if not ds_out.equals(ds_b[['lat', 'lon']]):
        regridder_b = xe.Regridder(ds_b, ds_out, 'bilinear', periodic=True)
        regridder_b.clean_weight_file()
    else: 
        regridder_b = None
        
    return regridder_a, regridder_b

class ModelRunsDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds.time)*len(self.ds.run)
    
    def __getitem__(self, index):
        index_t = index%len(self.ds.time)
        index_r = index//len(self.ds.time)
        X = self.ds.isel(time=index_t, run=index_r).to_array().load() - 273
        return torch.from_numpy(X.values)
    
    @property
    def shape(self):
        return (len(self),)+self.ds.isel(time=0, run=0).to_array().shape
    
    @property
    def dims(self):
        return ('sample',)+self.ds.isel(time=0, run=0).to_array().dims
    
class SplitModelRunsDataset(ModelRunsDataset):
    
    def __init__(self, ds, allowed_indices):
        super().__init__(ds)
        self.allowed_indices = allowed_indices
        
    def __len__(self):
        return len(self.allowed_indices)
    
    def __getitem__(self, index):
        index = self.allowed_indices[index]
        return super().__getitem__(index)

def train_test_split(dataset: ModelRunsDataset, test_size: float, 
                                random_state: int =  None) -> ModelRunsDataset:
    indices = np.arange(len(dataset))
    train_indices, test_indices = _array_train_test_split(indices, test_size=test_size, 
                                                   shuffle=True, 
                                                   random_state=random_state)
    train_dataset = SplitModelRunsDataset(dataset.ds, train_indices)
    test_dataset = SplitModelRunsDataset(dataset.ds, test_indices)
    return train_dataset, test_dataset

