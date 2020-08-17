"""
Command line tool to translate data using pretrained UNIT network
"""

import xarray as xr
import numpy as np

from utils import get_config, get_all_data_loaders
from trainer import UNIT_Trainer

import torch


# load post-processing - opposite of preprocessing
def post_process_constructor(config, x2x):
    if config['preprocess_method']=='zeromean':
        
        ds_agg = xr.load_dataset(config[f'agg_data_{x2x[-1]}']).isel(height=0)
        
        def undo_zeromean(x):
            return x + ds_agg.sel(variable='mean').to_array()
        return undo_zeromean
    
    else:
        def celcius_to_kelvin(x):
            return x + 273
        return celcius_to_kelvin
    
    
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
        return x
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
    from dask.diagnostics import ProgressBar
    
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
    da = ds.to_array().transpose('run', 'time', 'variable', 'lat', 'lon')

    # append number of variables
    config['input_dim_a'] = loaders[0].dataset.shape[1]
    config['input_dim_b'] = loaders[2].dataset.shape[1]
    del loaders

    translate = complete_translate_constructor(config, args.checkpoint, args.x2x)
    
    da_translated = xr.apply_ufunc(translate, 
                                    da,
                                    vectorize=True,
                                    dask='parallelized', 
                                    output_dtypes=['float'],
                                    input_core_dims=[['variable', 'lat', 'lon']],
                                    output_core_dims=[['variable', 'lat', 'lon']])
    
    with ProgressBar():
        da_translated.to_dataset(dim='variable').to_zarr(args.output_zarr, consolidated=True, mode='w-')