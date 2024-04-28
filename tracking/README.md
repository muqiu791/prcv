## Requirements

The lines below should set up a fresh environment with everything you need: 
```
conda create --name pips
source activate pips
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
conda install pip
pip install -r requirements.txt
```

## Demo

To download reference model, download the weights from [Hugging Face. ![](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/aharley/pips)

Alternatively, you can run this:

```
sh get_reference_model.sh
```

To run this model on a sample video, run this:
```
python demo.py
```

This will run the model on a sequence included in `demo_images/`.

For each 8-frame subsequence, the model will return `trajs_e`. This is estimated trajectory data for the particles, shaped `B,S,N,2`, where `S` is the sequence length and `N` is the number of particles, and `2` is the `x` and `y` coordinates. The script will also produce tensorboard logs with visualizations, which go into `logs_demo/`, as well as a few gifs in `./*.gif`. 



## FlyingThings++ dataset

To download our exact FlyingThings++ dataset, try [this link](https://drive.google.com/drive/folders/1zzWkGGFgJPyHpVaSA19zpYlux1Mf6wGC?usp=share_link). If the link doesn't work, create the data from the original FlyingThings, as described next. 

To create our FlyingThings++ dataset, first [download FlyingThings](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). The data should look like:

```
../flyingthings/optical_flow/
../flyingthings/object_index/
../flyingthings/frames_cleanpass_webp/
```

Once you have the flows and masks, you can run `python make_trajs.py`. This will put 80G of trajectory data into:
```
../flyingthings/trajs_ad/
```

In parallel, you can run `python make_occlusions.py`. This will put 537M of occlusion data into:

```
../flyingthings/occluders_al
```

This data will be loaded and joined with corresponding rgb by the `FlyingThingsDataset` class in `flyingthingsdataset.py`, when training and testing.

(The suffixes "ad" and "al" are version counters.)

Once loaded by the dataloader (`flyingthingsdataset.py`), the RGB will look like this:
<img src='https://particle-video-revisited.github.io/images/flt_rgbs.gif'>

The corresponding trajectories will look like this:
<img src='https://particle-video-revisited.github.io/images/flt_trajs.gif'>


## Training

To train a model on the flyingthings++ dataset:

```
python train.py
```

First it should print some diagnostic information about the model and data. Then, it should print a message for each training step, indicating the model name, progress, read time, iteration time, and loss. 

```
model_name 1_8_128_I6_3e-4_A_tb89_21:34:46
loading FlyingThingsDataset [...] found 13085 samples in ../flyingthings (dset=TRAIN, subset=all, version=ad)
loading occluders [...] found 7856 occluders in ../flyingthings (dset=TRAIN, subset=all, version=al)
not using augs in val
loading FlyingThingsDataset [...] found 2542 samples in ../flyingthings (dset=TEST, subset=all, version=ad)
loading occluders...found 1631 occluders in ../flyingthings (dset=TEST, subset=all, version=al)
warning: updated load_fails (on this worker): 1/13085...
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000001/100000; rtime 9.79; itime 20.24; loss = 40.30593
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000002/100000; rtime 0.01; itime 0.37; loss = 43.12448
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000003/100000; rtime 0.01; itime 0.36; loss = 36.60324
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000004/100000; rtime 0.01; itime 0.38; loss = 40.91223
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000005/100000; rtime 0.01; itime 0.35; loss = 79.32227
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000006/100000; rtime 0.01; itime 0.53; loss = 22.14469
1_8_128_I6_3e-4_A_tb89_21:34:46; step 000007/100000; rtime 0.04; itime 0.46; loss = 24.75386
[...]
```
Occasional `load_fails` warnings are typically harmless. They indicate when the dataloader fails to get `N` trajectories for a given video, which simply causes a retry. If you greatly increase `N` (the number of trajectories), or reduce the crop size, you can expect this warning to occur more frequently, since these constraints make it more difficult to find viable samples. As long as your `rtime` (read time) is small, then things are basically OK. 

To reproduce the result in the paper, you should train with 4 gpus, with horizontal and vertical flips, with a command like this:
```
python train.py --horz_flip=True --vert_flip=True --device_ids=[0,1,2,3]
```


## Acknowledgement

More details about pips can be found in:

[Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories](https://github.com/aharley/pips).


