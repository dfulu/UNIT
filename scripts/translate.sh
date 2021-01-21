python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist/config.yaml --output_zarr /datastore/cam5/nat_hist_to_hadgem3_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist/checkpoints/gen_00160000.pt --a2b 0 --seed 98876

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-4channels/config.yaml --output_zarr /datadrive/cam5/nat_hist_to_hadgem3_4ch_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-4channels/checkpoints/gen_00110000.pt --x2x b2a --seed 9725432

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v3/config.yaml --output_zarr /datadrive/cam5/nat_hist_to_hadgem3_v3_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v3/checkpoints/gen_00550000.pt --x2x b2a --seed 237897

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/config.yaml --output_zarr /datadrive/cam5/nat_hist_to_hadgem3_v5_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/checkpoints/gen_00069000.pt --x2x b2a --seed 202002111018

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/config.yaml --output_zarr /datadrive/hadgem3/nat_hist_to_hadgem3_v5.1_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/checkpoints/gen_00056000.pt --x2x a2b --seed 202015121437