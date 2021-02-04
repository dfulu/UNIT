"""
Command line tool to translate data using pretrained UNIT network
"""

import xarray as xr
import numpy as np

from climatetranslation.unit.utils import get_config
from climatetranslation.unit.data import (
    get_dataset, 
    CustomTransformer, 
    UnitModifier, 
    ZeroMeaniser, 
    Normaliser,
    dataset_time_overlap,
    get_land_mask,
    even_lat_lon,
)
from climatetranslation.unit.trainer import UNIT_Trainer

import torch
    
    
def network_translate_constructor(config, checkpoint, x2x):
    
    # load model
    state_dict = torch.load(checkpoint)

    trainer = UNIT_Trainer(config)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.eval().cuda()
    encode = trainer.gen_a.encode if x2x[0]=='a' else trainer.gen_b.encode # encode function
    decode = trainer.gen_a.decode if x2x[-1]=='a' else trainer.gen_b.decode # decode function
    
    def network_translate(x):
        x = np.array(x)[np.newaxis, ...]
        x = torch.from_numpy(x).cuda()
        x, noise = encode(x)
        x = decode(x)
        x = x.cpu().detach().numpy()
        return x[0]
    return network_translate


def get_data_transformer(conf):
    # load pre/post processing transformer
    if conf['preprocess_method']=='zeromean':
        prepost_trans = ZeroMeaniser(conf)
    elif conf['preprocess_method']=='normalise':
        prepost_trans = Normaliser(conf)
    elif conf['preprocess_method']=='units':
        prepost_trans = UnitModifier(conf)
    elif conf['preprocess_method']=='custom_allfield':
        prepost_trans = CustomTransformer(conf, tas_field_norm=True, pr_field_norm=True)
    elif conf['preprocess_method']=='custom_tasfield':
        prepost_trans = CustomTransformer(conf, tas_field_norm=True, pr_field_norm=False)
    elif conf['preprocess_method']=='custom_prfield':
        prepost_trans = CustomTransformer(conf, tas_field_norm=False, pr_field_norm=True)
    elif conf['preprocess_method']=='custom_nofield':
        prepost_trans = CustomTransformer(conf, tas_field_norm=False, pr_field_norm=False)
    else:
        raise ValueError(f"Unrecognised preprocess_method : {conf['preprocess_method']}")
    return prepost_trans


if __name__=='__main__':
    
    import argparse
    import progressbar
    
    def check_x2x(x2x):
        x2x = str(x2x)
        if x2x not in ['a2a', 'a2b', 'b2a', 'b2b']:
            raise ValueError("Invalid x2x arg")
        return x2x
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--output_zarr', type=str, help="output zarr store path")
    parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
    parser.add_argument('--x2x', type=check_x2x, help="any of [a2a, a2b, b2a, b2b]")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load experiment setting
    conf = get_config(args.config)
    
    # load the datasets
    ds_a = get_dataset(conf['data_zarr_a'], conf['level_vars'], 
                       filter_bounds=False, split_at=conf['split_at'], 
                       bbox=conf['bbox'])
    ds_b = get_dataset(conf['data_zarr_b'], conf['level_vars'], 
                       filter_bounds=False, split_at=conf['split_at'], 
                       bbox=conf['bbox'])
    
    if conf['time_range'] is not None:
        if conf['time_range'] == 'overlap':
            ds_a, ds_b = dataset_time_overlap([ds_a, ds_b])
        elif isinstance(conf['time_range'], dict):
            time_slice = slice(conf['time_range']['start_date'], conf['time_range']['end_date'])
            ds_a = ds_a.sel(time=time_slice)
            ds_b = ds_b.sel(time=time_slice)
        else:
            raise ValueError("time_range not valid : {}".format(conf['time_range']))
    
    prepost_trans = get_data_transformer(conf)
    prepost_trans.fit(ds_a, ds_b)
    
    ds_a = even_lat_lon(prepost_trans.transform_a(ds_a))
    ds_b = even_lat_lon(prepost_trans.transform_b(ds_b))
    post_trans = prepost_trans.inverse_a if args.x2x[-1]=='a' else prepost_trans.inverse_b

    # load model 
    conf['input_dim_a'] = len(ds_a.keys())
    conf['input_dim_b'] = len(ds_b.keys())
    conf['land_mask_a'] = get_land_mask(ds_a)
    conf['land_mask_b'] = get_land_mask(ds_b)
    net_trans = network_translate_constructor(conf, args.checkpoint, args.x2x)
    
    ds = ds_a if args.x2x[0]=='a' else ds_b
    
    mode = 'w-'
    append_dim = None
    n_times = 100
    N_times = len(ds.time)
    
    
    with progressbar.ProgressBar(max_value=N_times) as bar:
        
        for i in range(0, N_times, n_times):
            
            # pre-rocess and convert to array
            da = (
                ds.isel(time=slice(i, min(i+n_times, N_times)))
                .to_array()
                .transpose('run', 'time', 'variable', 'lat', 'lon')
            )
            
            # transform through network 
            da = xr.apply_ufunc(
                net_trans, 
                da,
                vectorize=True,
                dask='allowed',
                output_dtypes=['float'],
                input_core_dims=[['variable', 'lat', 'lon']],
                output_core_dims=[['variable', 'lat', 'lon']]
            )
            
            # fix chunking
            da = da.chunk(dict(run=1, time=1, lat=-1, lon=-1))
            
            # post-process
            ds_translated = post_trans(da.to_dataset(dim='variable'))
            
            # append to zarr
            ds_translated.to_zarr(
                args.output_zarr, 
                mode=mode, 
                append_dim=append_dim,
                consolidated=True
            )
            
            # update progress bar and change modes so dat can be appended
            bar.update(i)
            mode, append_dim='a', 'time'
            
        bar.update(N_times)