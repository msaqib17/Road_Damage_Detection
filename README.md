<div align="center">
  <h1>ERDD: Efficient Road Damage Detection</h1>
</div>
An efficient approach for detecting multiple types of road damage using CSPDarknet with Attention4D.

## Introduction

This repository presents an efficient road damage detection framework using deep learning. Our approach incorporates Attention4D blocks within the CSPNeXtPAFPN neck to improve feature refinement across multiple scales, enabling better detection of various road damage types. The proposed methodology demonstrates superior performance in detecting large-sized road cracks while maintaining competitive overall detection capabilities.

## Installation

```bash
### Clone the repository
git clone https://github.com/yourusername/road-damage-detection.git
cd road-damage-detection

# Create and activate conda environment
conda create -n road python=3.8 -y
conda activate road

# Install PyTorch (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install MMEngine and MMCV
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install MMDetection
pip install -v -e .

```


## Proposed Dataset
Will be released very soon.

## Demo
Download weights from this Google Drive link https://drive.google.com/file/d/1XccDJlnS1PfmrHOUWfeRvRML9Hg6xsuj/view?usp=sharing
```bash
python demo/image_demo.py demo/86.JPG configs/rtmdet/rtmdet_l_8xb32-300e_coco.py --weights work_dirs/epoch_300.pth
```

## Training
```bash
python tools/train.py configs/rtmdet/rtmdet_l_8xb32-300e_coco.py
```
## Testing
```bash
python tools/test.py configs/rtmdet/rtmdet_l_8xb32-300e_coco.py work_dirs/rtmdet_l_8xb32-300e_coco/epoch_300.pth --cfg-options test_dataloader.dataset.ann_file=voc07_test.json test_dataloader.dataset.data_prefix.img=JPEGImages test_dataloader.dataset.data_prefix._delete_=True test_evaluator.format_only=True test_evaluator.ann_file=voc07_test.json test_evaluator.outfile_prefix=work_dirs/results
```
## Model Zoo

| Architecture | Backbone               | Input Size | AP    | AP_large | AP_small | Download                        |
|--------------|------------------------|------------|-------|----------|----------|---------------------------------|
| Ours         | CSPDarknet-Attention4D | 640x640    | 0.445 | 0.458    | 0.432    | [model](link) \| [config](link) |
| RTMDet       | CSPNeXt                | 640x640    | 0.423 | 0.435    | 0.411    | [model](link) \| [config](link) |
| PPYOLOE      | CSPResNet              | 640x640    | 0.411 | 0.425    | 0.398    | -                               |
| YOLOv8       | YOLOv8-backbone        | 640x640    | 0.402 | 0.418    | 0.386    | -                               |

## Performance Comparison 

| Method  | Backbone               | AP    | AP_50  | AP_75 | AP_small | AP_medium | AP_large |
|---------|------------------------|-------|--------|-------|----------|-----------|----------|
| Ours    | CSPDarknet-Attention4D | 0.445 | 0.675  | 0.482 | 0.432    | 0.446     | 0.458    |
| RTMDet  | CSPNeXt                | 0.423 | 0.654  | 0.461 | 0.411    | 0.425     | 0.435    |
| PPYOLOE | CSPResNet              | 0.411 | 0.642  | 0.448 | 0.398    | 0.412     | 0.425    |
| YOLOv8  | YOLOv8-backbone        | 0.402 | 0.635  | 0.442 | 0.386    | 0.405     | 0.418    |

# Citation
@article{yourarticle2024,
  title={Your Paper Title},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
