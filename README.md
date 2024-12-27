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
## Performance Evaluation on our dataset

## Detection Results

| Methods | Backbone | AP | AP₅₀ | AP₇₅ | APₛ | APₘ | APₗ | AR | ARₛ | ARₘ | ARₗ |
|:--------|:---------|:---|:-----|:-----|:----|:----|:----|:---|:----|:----|:----|
| YOLOV8 | YOLOv8CSPDarknet | 0.122 | 0.299 | 0.082 | 0.000 | 0.083 | 0.127 | 0.448 | 0.000 | 0.234 | 0.454 |
| YOLOV7 | YOLOv7Backbone | *0.255* | *0.498* | *0.233* | 0.000 | **0.127** | 0.263 | 0.547 | 0.000 | 0.351 | 0.553 |
| YOLOV6 | YOLOv6Backbone | 0.110 | 0.263 | 0.095 | 0.000 | 0.108 | 0.114 | *0.560* | 0.000 | **0.460** | 0.572 |
| PPYOLOE | PPYOLOECSPResNet | 0.112 | 0.463 | 0.062 | 0.000 | 0.079 | 0.117 | 0.322 | 0.000 | *0.388* | 0.325 |
| RTMDET | CSPNeXt | 0.268 | 0.527 | 0.229 | 0.000 | *0.123* | *0.280* | 0.517 | 0.000 | 0.373 | *0.623* |
| YOLOX | YOLOXCSPDarknet | 0.200 | 0.377 | 0.188 | 0.000 | 0.006 | 0.204 | 0.288 | 0.000 | 0.033 | 0.386 |
| **Ours** | CSPNeXt | **0.446** | **0.687** | **0.451** | 0.000 | 0.113 | **0.458** | **0.675** | 0.000 | 0.277 | **0.690** |

*Detection results (mAP) on road-crack Dataset. The best results are shown in **bold** and the second best in *italics*.*


## Scale-specific Performance Comparison (APₛ/APₘ/APₗ)
For comprehensive results. Please refer to the paper
## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{saqib2024road,
    title={Road Damage Detection Using Attention4D Blocks},
    author={Saqib, Muhammad and Author2, Name and Author3, Name},
    journal={arXiv preprint arXiv:xxxx.xxxx},
    year={2024}
}
```
