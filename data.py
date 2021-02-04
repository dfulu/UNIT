"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from sklearn.model_selection import train_test_split as _array_train_test_split
import xarray as xr
import numpy as np
import xesmf as xe
from global_land_mask import globe
import torch
from functools import reduce
import cftime
import datetime
import warnings
from dask.array.core import PerformanceWarning

from abc import ABC, abstractmethod


def reduce_height(ds, level_vars):
    ds_list = []
    if 'height' in ds.dims:
        for h, v in level_vars.items():
            ds_list += [ds.sel(height=h)[[vi for vi in v]].drop('height')]
        if len(ds_list)>1:
            ds = reduce(lambda ds1, ds2: ds1.merge(ds2), ds_list)
        else:
            ds = ds_list[0]
    else:
        ds = ds[[vi for _, v in level_vars.items() for vi in v]]
    return ds

    
def any_calendar_to_string(dt):
    if isinstance(dt, np.datetime64):
        return str(dt).split('.')[0]
    elif isinstance(dt, datetime.datetime):
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    elif isinstance(dt, cftime.datetime):
        return dt.strftime().replace(' ', 'T')


def any_calendar_to_datetime(dt):
    if isinstance(dt, datetime.datetime):
        return dt
    elif isinstance(dt, np.datetime64):
        return datetime.datetime.strptime(
        str(dt).split('.')[0], 
        "%Y-%m-%dT%H:%M:%S"
        )
    elif isinstance(dt, cftime.datetime):
        return datetime.datetime.strptime(
            dt.strftime(),
            "%Y-%m-%d %H:%M:%S"
        )


def datemin_to_string(dts):
    dts = [any_calendar_to_datetime(dt) for dt in dts]
    return any_calendar_to_string(min(dts))


def datemax_to_string(dts):
    dts = [any_calendar_to_datetime(dt) for dt in dts]
    return any_calendar_to_string(max(dts))


def dataset_time_overlap(datasets_list):
    start_time = datemax_to_string([ds.time.values.min() for ds in datasets_list])
    end_time = datemin_to_string([ds.time.values.max() for ds in datasets_list])
    if (
        datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S") >=
        datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
    ):
        raise ValueError("start time ({}) and end time ({}) leave no overlap".format(start_time, end_time))
    print("start time ({}) and end time ({})".format(start_time, end_time))
    return [ds.sel(time=slice(start_time, end_time)) for ds in datasets_list]


def filter_bounds(ds):
    return ds[[v for v in ds.data_vars if not 'bnds' in v]]


def split_lon_at(ds, degree):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        lons = ds.lon.values.copy()
        too_big = lons>=degree
        too_small = lons<degree-360
        lons[too_big] = lons[too_big] - 360
        lons[too_small] = lons[too_small] + 360
        ds['lon'] = lons
        ds =  ds.sortby(ds.lon)
    return ds


def _quick_add_bounds(ds):
    assert len(np.unique(np.diff(ds.lon).round(3))) == 1
    assert len(np.unique(np.diff(ds.lat).round(3))) == 1

    dlat = np.diff(ds.lat).mean()
    dlon = np.diff(ds.lon).mean()

    lat_b = np.concatenate((ds.lat - dlat/2, [ds.lat[-1]+dlat/2])).clip(-90,90)
    lon_b = np.concatenate((ds.lon - dlon/2, [ds.lon[-1]+dlon/2]))
    
    ds['lat_b'] = lat_b
    ds['lon_b'] = lon_b


def _quick_remove_bounds(ds):
    del ds['lat_b']
    del ds['lon_b']


def even_lat_lon(ds):
    return ds.isel(
            lat=slice(0, len(ds.lat)//2 * 2),
            lon=slice(0, len(ds.lon)//2 * 2)
        )


def construct_regridders(ds_a, ds_b, resolution_match='downscale', scale_method='bilinear', periodic=True):
    
    if resolution_match=='downscale':
        ds_out = xr.Dataset({'lat': min([ds_a.lat, ds_b.lat], key=lambda x: len(x)), 
                         'lon': min([ds_a.lon, ds_b.lon], key=lambda x: len(x))})
    elif resolution_match=='upscale':
        ds_out = xr.Dataset({'lat': max([ds_a.lat, ds_b.lat], key=lambda x: len(x)), 
                         'lon': max([ds_a.lon, ds_b.lon], key=lambda x: len(x))})
    else:
        raise ValueError("resolution_match must be one of ['upscale', 'downscale']")

    _quick_add_bounds(ds_out)
    _quick_add_bounds(ds_a)
    _quick_add_bounds(ds_b)

    if not ds_out[['lat', 'lon']].equals(ds_a[['lat', 'lon']]):
        regridder_a = xe.Regridder(ds_a, ds_out, scale_method, periodic=periodic)
        regridder_a.clean_weight_file()
    else: 
        regridder_a = None

    if not ds_out[['lat', 'lon']].equals(ds_b[['lat', 'lon']]):
        regridder_b = xe.Regridder(ds_b, ds_out, scale_method, periodic=periodic)
        regridder_b.clean_weight_file()
    else: 
        regridder_b = None

    _quick_remove_bounds(ds_a)
    _quick_remove_bounds(ds_b)
        
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

def z500_to_anomaly(ds):
    """Convert z500 to anomaly in 100m"""
    ds['z500'] = (ds['z500'] - 5500) / 100
    return ds

def z500_anomaly_to_z500(ds):
    """Convert z500 anomaly in 100m to z500 normal"""
    ds['z500'] = (ds['z500'] * 100) + 5500
    return ds

def temp_minmax_to_diff(ds):
    if 'tas' in ds.keys():
        for v in ['tasmin', 'tasmax']:
            if v in ds.keys():
                ds[v] = ds[v] - ds['tas']
    return ds

def temp_diff_to_minmax(ds):
    if 'tas' in ds.keys():
        for v in ['tasmin', 'tasmax']:
            if v in ds.keys():
                ds[v] = ds[v] + ds['tas']
    return ds

class Transformer(ABC):
    def __init__(self, conf):
        self.conf = conf
        self._fit = False
        self.ds_agg_a = None
        self.ds_agg_b = None
        self.rg_a = None
        self.rg_b = None
            
    def _check_fit(self):
        if not self._fit:
            raise ValueError("Need to call .fit() method first")
            
    def fit(self, ds_a, ds_b):
        periodic = self.conf['bbox'] is None
        # match to the coarsest resolution of the pair
        self.rg_a, self.rg_b = construct_regridders(ds_a, ds_b, self.conf['resolution_match'], self.conf['scale_method'], periodic)
        # modify aggregates since regridding is done before preprocessing
        # BEGIN DEPRECATED
        # agg data is now calculated after transform
        #if self.ds_agg_a is not None and self.rg_a is not None:
        #    self.ds_agg_a = self.rg_a(self.ds_agg_a).astype(np.float32)
        #if self.ds_agg_b is not None and self.rg_b is not None:
        #    self.ds_agg_b = self.rg_b(self.ds_agg_b).astype(np.float32)
        # END DEPRECATED
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
    def __init__(self, conf):
        super().__init__(conf)
        self.ds_agg_a = xr.load_dataset(conf['agg_data_a'])
        self.ds_agg_b = xr.load_dataset(conf['agg_data_b'])
        
    def _transform(self, ds, rg, ds_agg):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        ds = ds - ds_agg.sel(aggregate_statistic='mean').drop('aggregate_statistic')
        ds = ds / ds_agg.sel(aggregate_statistic='std').drop('aggregate_statistic')
        return ds
        
    def _inverse(self, ds, ds_agg):
        ds = ds * ds_agg.sel(aggregate_statistic='std').drop('aggregate_statistic')
        ds = ds + ds_agg.sel(aggregate_statistic='mean').drop('aggregate_statistic')
        return ds


class ZeroMeaniser(Normaliser):
    def __init__(self, conf):
        super().__init__(conf)

    def _transform(self, ds, rg, ds_agg):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        ds = ds - ds_agg.sel(aggregate_statistic='mean').drop('aggregate_statistic')
        return ds
        
    def _inverse(self, ds, ds_agg):
        ds = ds + ds_agg.sel(aggregate_statistic='mean').drop('aggregate_statistic')
        return ds


class UnitModifier(Transformer):
    def __init__(self, conf):
        super().__init__(conf)
        
    def _transform(self, ds, rg, *args):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        ds = kelvin_to_celcius(ds)
        ds = precip_kilograms_to_mm(ds)
        ds = z500_to_anomaly(ds)
        if self.conf['tas_diff']:
            ds = temp_minmax_to_diff(ds)
        return ds
        
    def _inverse(self, ds, *args):
        if self.conf['tas_diff']:
            ds = temp_diff_to_minmax(ds)
        ds = z500_anomaly_to_z500(ds)
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
    def __init__(self, conf, tas_field_norm=True, pr_field_norm=False):
        super().__init__(conf)
        self.tas_field_norm = tas_field_norm
        self.pr_field_norm = pr_field_norm

    def fit(self, ds_a, ds_b):
        super().fit(ds_a, ds_b)
        # same transforms to both datasets
        self.ds_agg = 0.5 * (self.ds_agg_a + self.ds_agg_b)
        if not self.tas_field_norm:
            all_other_vars = [k for k in self.ds_agg.keys() if k in ['tas', 'tasmin', 'tasmax', 'z500']]
            for k in all_other_vars:
                self.ds_agg[k] = self.ds_agg[k].mean(dim=('lat', 'lon'))
        if not self.pr_field_norm:
            self.ds_agg['pr_4root'] = self.ds_agg['pr_4root'].mean(dim=('lat', 'lon'))
            
        self.ds_agg_a = self.ds_agg_b = self.ds_agg
            
        
    def _transform(self, ds, rg, ds_agg):
        ds = ds if rg is None else rg(ds).astype(np.float32)
        
        # precipitation
        if 'pr' in ds.keys():
            # In some of the data numerical error means 0 -> O(1e-22). Therefore need to clip.
            ds['pr'] = ds['pr'].clip(0, None)**(1/4)
            ds['pr'] /= ds_agg['pr_4root'].sel(aggregate_statistic='std').drop('aggregate_statistic')
            
        # temperature
        ds = kelvin_to_celcius(ds)
        if self.conf['tas_diff']:
            ds = temp_minmax_to_diff(ds)
        temp_vars = [k for k in ds.keys() if k.startswith('tas')]
        for k in temp_vars:
            ds[k] /= ds_agg['tas'].sel(aggregate_statistic='std').drop('aggregate_statistic')
        
        # other
        other_vars = [k for k in ds.keys() if k in ['z500']]
        for k in other_vars:
            ds[k] -= ds_agg[k].sel(aggregate_statistic='mean').drop('aggregate_statistic')
            ds[k] /= ds_agg[k].sel(aggregate_statistic='std').drop('aggregate_statistic')
        return ds
        
    def _inverse(self, ds, ds_agg):
        # precipitation
        if 'pr' in ds.keys():
            ds['pr'] *= ds_agg['pr_4root'].sel(aggregate_statistic='std').drop('aggregate_statistic')
            ds['pr'] = ds['pr']**4
            
        # temperature
        temp_vars = [k for k in ds.keys() if k.startswith('tas')]
        for k in temp_vars:
            ds[k] *= ds_agg['tas'].sel(aggregate_statistic='std').drop('aggregate_statistic')
        if self.conf['tas_diff']:
            ds = temp_diff_to_minmax(ds)
        ds = celcius_to_kelvin(ds)
        
        # other
        other_vars = [k for k in ds.keys() if k in ['z500']]
        for k in other_vars:
            ds[k] *= ds_agg[k].sel(aggregate_statistic='std').drop('aggregate_statistic')
            ds[k] += ds_agg[k].sel(aggregate_statistic='mean').drop('aggregate_statistic')
        return ds

    
def get_land_mask(ds):
    lat = ds.lat.values.copy()
    lon = ds.lon.values.copy()
    lon[lon>180] = lon[lon>180]-360
    lon[lon<=-180] = lon[lon<=-180]+360
    lon_grid, lat_grid = np.meshgrid(lon,lat)
    land_mask = torch.from_numpy(
        globe.is_land(lat_grid, lon_grid)
            .astype(np.float32)).unsqueeze(0)
    return land_mask


class ModelRunsDataset(torch.utils.data.Dataset):
    def __init__(self, ds, use_land_mask=False):
        # Ensure even valued image sides
        self.ds = even_lat_lon(ds)
        self.use_land_mask = use_land_mask
        if use_land_mask:
            self.land_mask = get_land_mask(ds)
        else:
            self.land_mask = None

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
    
    def __init__(self, ds, allowed_indices, use_land_mask=False):
        super().__init__(ds, use_land_mask)
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
    train_dataset = SplitModelRunsDataset(dataset.ds, train_indices, dataset.use_land_mask)
    test_dataset = SplitModelRunsDataset(dataset.ds, test_indices, dataset.use_land_mask)
    return train_dataset, test_dataset


def get_dataset(zarr_path, level_vars=None, filter_bounds=True, split_at=360, bbox=None):
    """
    zarr_path
    reduce_height: {height: [variables],}
    filter_bounds: bool, optional
    split_at: int, [360, 180]
    bbox: {}
    """
    if split_at not in [360, 180]:
        raise ValueError("image must be split at 360 or 180")

    ds = xr.open_zarr(zarr_path, consolidated=True)
    ds = split_lon_at(ds, split_at)
    
    if bbox is not None:
        print(bbox)
        ds = ds.sel(lat=slice(bbox['S'], bbox['N']), lon=slice(bbox['W'], bbox['E']))
    if filter_bounds:
        ds = ds[[v for v in ds.data_vars if not 'bnds' in v]]
    if level_vars is not None:
        ds = reduce_height(ds, level_vars)
    return ds


def get_all_data_loaders(conf):

    # Parameters
    params = {'batch_size': conf['batch_size'],
              'num_workers': conf['num_workers']}
    
    ds_a = get_dataset(conf['data_zarr_a'], conf['level_vars'], filter_bounds=False, split_at=conf['split_at'], bbox=conf['bbox'])
    ds_b = get_dataset(conf['data_zarr_b'], conf['level_vars'], filter_bounds=False, split_at=conf['split_at'], bbox=conf['bbox'])
    
    if conf['time_range'] is not None:
        if conf['time_range'] == 'overlap':
            ds_a, ds_b = dataset_time_overlap([ds_a, ds_b])
        elif isinstance(conf['time_range'], dict):
            time_slice = slice(conf['time_range']['start_date'], conf['time_range']['end_date'])
            ds_a = ds_a.sel(time=time_slice)
            ds_b = ds_b.sel(time=time_slice)
        else:
            raise ValueError("time_range not valid : {}".format(conf['time_range']))
    
    if conf['preprocess_method']=='zeromean':
        trans = ZeroMeaniser(conf)
    elif conf['preprocess_method']=='normalise':
        trans = Normaliser(conf)
    elif conf['preprocess_method']=='units':
        trans = UnitModifier(conf)
    elif conf['preprocess_method']=='custom_allfield':
        trans = CustomTransformer(conf, tas_field_norm=True, pr_field_norm=True)
    elif conf['preprocess_method']=='custom_tasfield':
        trans = CustomTransformer(conf, tas_field_norm=True, pr_field_norm=False)
    elif conf['preprocess_method']=='custom_prfield':
        trans = CustomTransformer(conf, tas_field_norm=False, pr_field_norm=True)
    elif conf['preprocess_method']=='custom_nofield':
        trans = CustomTransformer(conf, tas_field_norm=False, pr_field_norm=False)
    else:
        raise ValueError(f"Unrecognised preprocess_method : {conf['preprocess_method']}")
    trans.fit(ds_a, ds_b)
    
    ds_a = filter_bounds(ds_a)
    ds_b = filter_bounds(ds_b)
    
    ds_a = trans.transform_a(ds_a)
    ds_b = trans.transform_b(ds_b)
        
    dataset_a_train, dataset_a_test = train_test_split(ModelRunsDataset(ds_a, conf['use_land_mask']), conf['test_size'])
    dataset_b_train, dataset_b_test = train_test_split(ModelRunsDataset(ds_b, conf['use_land_mask']), conf['test_size'])
    
    loaders = [torch.utils.data.DataLoader(d, **params) for d in 
                  [dataset_a_train, dataset_a_test, 
                   dataset_b_train, dataset_b_test]
              ]
    return loaders
