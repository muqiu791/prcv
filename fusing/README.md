
## Usage
- Download all these files.
- If you want to test your own images on our model, "matlab_code_for_creating_base_and_detail_layers/main.m" is ready for you to generate the base and detail layers.

All the necessary parameter settings can be found at "args_fusion.py".

## Training
Training dataset can be found at this website: https://pjreddie.com/projects/coco-mirror/

Put the images at the "train2014" folder.

```
python train.py
```

## Environment
- Python 3.7.3
- torch 1.7.1
- scipy 1.2.0

## Acknowledgement
More details about fusinon can be found in:
[UNIFusion: A Lightweight Unified Image Fusion Network](hhttps://github.com/AWCXV/UNIFusion)

