"""
Command line tool to translate data using pretrained UNIT network
"""

import xarray as xr
import numpy as np

from utils import get_config
from data import get_dataset, Transformer
from trainer import UNIT_Trainer

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
    


class Translator:
    def __init__(self, config, checkpoint, x2x):
        self.x2x = x2x
        self.network_translate = network_translate_constructor(config, checkpoint, x2x)
        self.trans = 
        self._fit = False

    def _check_fit(self):
        is not self.self._fit:
            raise ValueError("Need to call .fit() method first")
            
    def fit(ds_a, ds_b):
        self.trans.fit(ds_a, ds_b)
        self._fit=True
        
    def translate(self, ds):
        self._check_fit()
        
        # preprocess
        if self.x2x[0] == 'a':
            ds = self.trans.transform_a(ds)
        else:
            ds = self.trans.transform_b(ds)
            
        # send through network
        xr.zeros_like(ds)
        ds.values = self.network_translate(ds.values)
        
        # undo preprocessing
        if self.x2x[-1] == 'a':
            ds = self.trans.inverse_a(ds)
        else:
            ds = self.trans.inverse_b(ds)
            
        return ds
        
            

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
    
    # load the datasets
    ds_a = get_dataset(config['data_zarr_a'], config['level_vars'])
    ds_b = get_dataset(config['data_zarr_b'], config['level_vars'])
    
    # load pre/post processing transformer
    prepost_trans = Transformer(config, downscale_consolidate=True)
    prepost_trans.fit(ds_a, ds_b)
    
    pre_trans = prepost_trans.transform_a if args.x2x[0]=='a' else prepost_trans.transform_b
    post_trans = prepost_trans.inverse_a if args.x2x[-1]=='a' else prepost_trans.inverse_b

    # load model 
    config['input_dim_a'] = len(ds_a.keys())
    config['input_dim_b'] = len(ds_b.keys())
    net_trans = network_translate_constructor(config, args.checkpoint, args.x2x)
    
    mode = 'w-'
    append_dim = None
    n_times=100
    N_times = len(da.time)

    with progressbar.ProgressBar(max_value=N_times) as bar:
        
        for i in range(0, N_times, n_times):
            
            # pre-rocess and convert to array
            da = (
                pre_trans(ds.isel(time=slice(i, min(i+n_times, N_times))))
                .to_array()
                .transpose('run', 'time', 'variable', 'lat', 'lon')
            )
            
            # transform through network 
            da = xr.apply_ufunc(translate, 
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