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

from abc import ABC, abstractmethod


def reduce_height(ds, level_vars):
    ds_list = [ds.sel(height=h)[v].drop('height') for h,v in level_vars.items()]
    if len(ds_list)>1:
        ds = reduce(lambda ds1, ds2: ds1.merge(ds2), ds_list)
    else:
        ds = ds_list[0]
    return ds


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


def kelvin_to_celcius(ds):
    temp_vars = [v for v in ds.data_vars if 'tas' in v]
    for v in temp_vars:
        ds[v] = ds[v] - 273.15
    return ds


def celcius_to_kelvin(ds):
    temp_vars = [v for v in ds.data_vars if 'tas' in v]
    for v in temp_vars:
        ds[v] = ds[v] + 273.15
    return ds


def precip_kilograms_to_mm(ds):
    """Convert from (kg m^-2 s^-1) to (mm day^-1)"""
    precip_vars = [v for v in ds.data_vars if v=='pr']
    for v in precip_vars:
        ds[v] = ds[v] * 24*60**2
    return ds


def precip_mm_to_kg(ds):
    """Convert from (mm day^-1) to (kg m^-2 s^-1) """
    precip_vars = [v for v in ds.data_vars if v=='pr']
    for v in precip_vars:
        ds[v] = ds[v] / (24*60**2)
    return ds


class Transformer(ABC):
    def __init__(self, downscale_consolidate=True):
        self.downscale_consolidate = downscale_consolidate
        self._fit = False
        self.ds_agg_a = None
        self.ds_agg_b = None
        self.rg_a = None
        self.rg_b = None
            
    def _check_fit(self):
        if not self.self._fit:
            raise ValueError("Need to call .fit() method first")
            
    def fit(ds_a, ds_b):
        # match to the coarsest resolution of the pair
        if self.downscale_consolidate:
            self.rg_a, self.rg_b = construct_regridders(ds_a, ds_b)
            # modify aggregates since regridding is done before preprocessing
            if self.ds_agg_a is not None and self.rg_a is not None:
                self.ds_agg_a = self.rg_a(self.ds_agg_a).astype(np.float32)
            if self.ds_agg_b is not None and self.rg_b is not None:
                self.ds_agg_b = self.rg_b(self.ds_agg_b).astype(np.float32)
        else:
            self.rg_a = self.rg_b = None
        self._fit=True
        
    @abstractmethod
    def _transform(self, ds, rg, ds_agg):
        pass
        
    @abstractmethod
    def _inverse(self, ds, ds_agg):
        pass
    
    def transform_a(self, ds):
        self._check_fit()
        return self._transform(ds, self.rg_a, self.ds_agg_a)
    
    def transform_b(self, ds):
        self._check_fit()
        return self._transform(ds, self.rg_b, self.ds_agg_b)
    
    def inverse_a(self, ds):
        self._check_fit()
        return self._inverse(ds, self.ds_agg_a)
        
    def inverse_b(self, ds):
        self._check_fit()
        return self._inverse(ds, self.ds_agg_b)


class Normaliser(Transformer):
    def __init__(self, conf, downscale_consolidate=True):
        super().__init__(downscale_consolidate)
        self.ds_agg_a = reduce_height(xr.load_dataset(conf['agg_data_a']), conf['level_vars'])
        self.ds_agg_b = reduce_height(xr.load_dataset(conf['agg_data_b']), conf['level_vars'])
        
    def _transform(self, ds, rg, ds_agg):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        ds = ds - ds_agg.sel(aggregate_statistic='mean')
        ds = ds / ds_agg.sel(aggregate_statistic='std')
        return ds
        
    def _inverse(self, ds, ds_agg):
        ds = ds * ds_agg.sel(aggregate_statistic='std')
        ds = ds + ds_agg.sel(aggregate_statistic='mean')   
        return ds


class ZeroMeaniser(Normaliser):
    def __init__(self, conf, downscale_consolidate=True):
        super().__init__(conf, downscale_consolidate)

    def _transform(self, ds, rg, ds_agg):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        ds = ds - ds_agg.sel(aggregate_statistic='mean')
        return ds
        
    def _inverse(self, ds, ds_agg):
        ds = ds + ds_agg.sel(aggregate_statistic='mean')   
        return ds

class UnitModifier(Transformer):
    def __init__(self, conf, downscale_consolidate=True):
        super().__init__(downscale_consolidate)
        
    def _transform(self, ds, rg, *args):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        ds = kelvin_to_celcius(ds)
        ds = precip_kilograms_to_mm(ds)
        return ds
        
    def _inverse(self, ds, *args):
        ds = precip_mm_to_kg(ds)
        ds = celcius_to_kelvin(ds)
        return ds

    
class CustomTransformer(Normaliser):
    """A non-standard set of transforms for (tas, tasmin, tasmax, pr).
    
    To make the precip distribution less extreme:
        pr -> pr^1/4
        
    Shift temperatures to celcius so significance of zero C is easy.
    
    Scale min/mean/max temperatures in same way so relation between them is obvious.
    
    Scale all variables so precip and temps are given equivalent losses (ish).
    """
    def __init__(self, conf, downscale_consolidate=True, tas_field_norm=True, pr_field_norm=False):
        super().__init__(conf, downscale_consolidate)
        self.tas_field_norm = tas_field_norm
        self.pr_field_norm = pr_field_norm

    def fit(ds_a, ds_b):
        # match to the coarsest resolution of the pair
        if self.downscale_consolidate:
            self.rg_a, self.rg_b = construct_regridders(ds_a, ds_b)
            # modify aggregates since regridding is done before preprocessing
            if self.ds_agg_a is not None and self.rg_a is not None:
                self.ds_agg_a = self.rg_a(self.ds_agg_a).astype(np.float32)
            if self.ds_agg_b is not None and self.rg_b is not None:
                self.ds_agg_b = self.rg_b(self.ds_agg_b).astype(np.float32)
        else:
            self.rg_a = self.rg_b = None
            
        # same transforms to both datasets
        self.ds_agg = 0.5 * (self.ds_agg_a + self.ds_agg_b)
        if not self.tas_field_norm:
            temp_vars = [k for k in ds.keys() if k.startswith('tas')]
            self.ds_agg[temp_vars] = self.ds_agg[temp_vars].mean(dim=('lat', 'lon'))
        if not self.pr_field_norm:
            self.ds_agg['pr_4root'] = self.ds_agg['pr_4root'].mean(dim=('lat', 'lon'))
            
        self.ds_agg_a = self.ds_agg_b = self.ds_agg
            
        self._fit=True
        
    def _transform(self, ds, rg, ds_agg):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        
        if 'pr' in ds.keys():
            ds['pr'] = ds['pr']**(1/4)
            ds['pr'] /= ds_agg['pr_4root'].sel(aggregate_statistic='std')
            
        ds = kelvin_to_celcius(ds)
        temp_vars = [k for k in ds.keys() if k.startswith('tas')]
        ds[temp_vars] /= ds_agg['tas'].sel(aggregate_statistic='std')
        return ds
        
    def _inverse(self, ds, ds_agg):
        if 'pr' in ds.keys():
            ds['pr'] *= ds_agg['pr_4root'].sel(aggregate_statistic='std')
            ds['pr'] = ds['pr']**4
            
        temp_vars = [k for k in ds.keys() if k.startswith('tas')]
        ds[temp_vars] *= ds_agg['tas'].sel(aggregate_statistic='std')
        ds = celcius_to_kelvin(ds)
        return ds


class ModelRunsDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds.time)*len(self.ds.run)
    
    def __getitem__(self, index):
        index_t = index%len(self.ds.time)
        index_r = index//len(self.ds.time)
        X = self.ds.isel(time=index_t, run=index_r).to_array().load()
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


def get_dataset(zarr_path, level_vars=None, filter_bounds=True):
    """
    zarr_path
    reduce_height: {height: [variables],}
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    if filter_bounds:
        ds = ds[[v for v in ds.data_vars if not 'bnds' in v]]
    if level_vars is not None:
        ds = reduce_height(ds, level_vars)
    return ds

def get_all_data_loaders(conf, downscale_consolidate=True):

    # Parameters
    params = {'batch_size': conf['batch_size'],
              'num_workers': conf['num_workers']}
    
    ds_a = get_dataset(conf['data_zarr_a'], conf['level_vars'])
    ds_b = get_dataset(conf['data_zarr_b'], conf['level_vars'])
    
    if conf['preprocess_method']=='zeromean':
        trans = ZeroMeaniser(conf, downscale_consolidate=downscale_consolidate)
    elif conf['preprocess_method']=='normalise':
        trans = Normaliser(conf, downscale_consolidate=downscale_consolidate)
    elif conf['preprocess_method']=='units':
        trans = UnitModifier(conf, downscale_consolidate=downscale_consolidate)
    elif conf['preprocess_method']=='custom_allfield':
        trans = CustomTransformer(conf, downscale_consolidate=downscale_consolidate, tas_field_norm=True, pr_field_norm=True)
    elif conf['preprocess_method']=='custom_tasfield':
        trans = CustomTransformer(conf, downscale_consolidate=downscale_consolidate, tas_field_norm=True, pr_field_norm=False)
    elif conf['preprocess_method']=='custom_prfield':
        trans = CustomTransformer(conf, downscale_consolidate=downscale_consolidate, tas_field_norm=False, pr_field_norm=True)
    else:
        raise ValueError(f'Unrecognised preprocess_method : {conf['preprocess_method']}')
    trans.fit(ds_a, ds_b)
    
    ds_a = trans.tranform_a(ds_a)
    ds_b = trans.tranform_b(ds_b)
    
    dataset_a_train, dataset_a_test = train_test_split(ModelRunsDataset(ds_a), conf['test_size'])
    dataset_b_train, dataset_b_test = train_test_split(ModelRunsDataset(ds_b), conf['test_size'])
    
    loaders = [torch.utils.data.DataLoader(d, **params) for d in 
                  [dataset_a_train, dataset_a_test, 
                   dataset_b_train, dataset_b_test]
              ]
    return loaders
