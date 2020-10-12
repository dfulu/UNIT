"""
Command line tool to translate data using pretrained UNIT network
"""

import xarray as xr
import numpy as np

from utils import get_config, get_all_data_loaders, reduce_height
from data import construct_regridders
from trainer import UNIT_Trainer

import torch


def post_process_constructor(config, x2x):
    if config['preprocess_method'] in ['zeromean', 'normalise']:
        
        ds_agg_a = xr.load_dataset(config[f'agg_data_a'])
        ds_agg_b = xr.load_dataset(config[f'agg_data_b'])
        rg_a, rg_b = construct_regridders(ds_agg_a, ds_agg_b)

        ds_agg_a = ds_agg_a if rg_a is None else rg_a(ds_agg_a).astype(np.float32)
        ds_agg_b = ds_agg_b if rg_b is None else rg_b(ds_agg_b).astype(np.float32)
        
        ds_agg = ds_agg_a if x2x[-1]=='a' else ds_agg_b
        
        ds_agg = reduce_height(ds_agg, config['level_vars'])
        
        def undo_zeromean(x):
            return x + ds_agg.sel(aggregate_statistic='mean').to_array()
    
        def undo_normalise(x):
            x = x * ds_agg.sel(aggregate_statistic='std').to_array()
            x = undo_zeromean(x)
            return x
        
        if config['preprocess_method']=='zeromean':
            return undo_zeromean
        elif config['preprocess_method']=='normalise':
            return undo_normalise
    
    else:
        def unit_convert(x):
            i = 0
            for _, var_list in config['level_vars'].items():
                for v in var_list:
                    if 'tas' in v:
                        x[i] = x[i] + 273
                    elif v=='pr':
                        x[i] = x[i]/1000
                    i+=1
            return x
        
        return unit_convert
    
    
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
    


def complete_translate_constructor(config, checkpoint, x2x):
    
    network_translate = network_translate_constructor(config, checkpoint, x2x)
    post_process = post_process_constructor(config, x2x)
    
    def translate(x):
        x = network_translate(x)
        x = post_process(x)
        return x
    
    return translate


if __name__=='__main__':
    
    import argparse
    import progressbar
    
    def check_x2x(x2x):
        x2x = str(x2x)
        if x2x not in ['a2a', 'a2b', 'b2a', 'b2b']:
            raise ValueError("Invalid x2x arg")
        return x2x
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='~/model_outputs/outputs/hadgem3_to_cam5_nat-hist/config.yaml', help='Path to the config file.')
    parser.add_argument('--output_zarr', type=str, help="output zarr store path")
    parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
    parser.add_argument('--x2x', type=check_x2x, help="any of [a2a, a2b, b2a, b2b]")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load experiment setting
    config = get_config(args.config)

    # Setup model and data loader
    # By constructing loader and extracting dataset from this we make sure all preprocessing is
    # consistent
    loaders = get_all_data_loaders(config, downscale_consolidate=True)
    ds = loaders[0].dataset.ds if args.x2x[0]=='a' else loaders[2].dataset.ds
    da = ds.to_array().transpose('run', 'time', 'variable', 'lat', 'lon')#.chunk({'variable': -1})

    # append number of variables
    config['input_dim_a'] = loaders[0].dataset.shape[1]
    config['input_dim_b'] = loaders[2].dataset.shape[1]
    del loaders

    translate = complete_translate_constructor(config, args.checkpoint, args.x2x)
    
    mode = 'w-'
    append_dim = None
    n_times=100
    N_times = len(da.time)

    with progressbar.ProgressBar(max_value=N_times) as bar:
        
        for i in range(0, N_times, n_times):
                
            da_translated = xr.apply_ufunc(translate, 
                                        da.isel(
                                            time=slice(i, min(i+n_times, N_times))
                                        ),
                                        vectorize=True,
                                        dask='allowed',
                                        output_dtypes=['float'],
                                        input_core_dims=[['variable', 'lat', 'lon']],
                                        output_core_dims=[['variable', 'lat', 'lon']]
            )
            
            da_translated = da_translated.chunk(dict(run=1, time=1, lat=-1, lon=-1))

            da_translated.to_dataset(dim='variable').to_zarr(
                args.output_zarr, 
                mode=mode, 
                append_dim=append_dim,
                consolidated=True
            )
            bar.update(i)
            mode, append_dim='a', 'time'
        bar.update(N_times)