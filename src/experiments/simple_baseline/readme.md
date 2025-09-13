# Simple Baseline Super-Resolution Experiment


## Goal:
1. Generate data.
2. Explore ways to downsample / induce noise. 
3. Upscale with some simple SR (maybe bicubic). 
4. Evaluate vs HQ: using simple metric like PSNR.


## Components so far
- [builders.py](builders.py): to create scenes with different TX/RX configs
- [ex_data.ipynb](ex_data.ipynb): example usage of builders.py
- [ex_resolution.ipynb](ex_resolution.ipynb): example of downsampling / upscaling / explicitly modifying radio maps


## TODO
- [ ] Add method for inducing noise 
- [ ] Bicubic interpolation fn (torch.nn.functional.interpolate?)
- [ ] Add relevant metrics
- [ ] Configure JSON dumps
- [ ] Add option to create multiple scenes and save in appropriate format
