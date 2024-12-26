<div align="center">
  <h1>Efficient Road Damage Detection</h1>
</div>

## Introduction

This repository presents an efficient road damage detection framework using deep learning. Our approach incorporates Attention4D blocks within the CSPNeXtPAFPN neck to improve feature refinement across multiple scales, enabling better detection of various road damage types. The proposed methodology demonstrates superior performance in detecting large-sized road cracks while maintaining competitive overall detection capabilities.

## Major Features

- **Enhanced Feature Extraction**: Integration of Attention4D blocks in CSPNeXtPAFPN neck for better multi-scale feature refinement
- **Multi-type Damage Detection**: Effective detection of multiple types of road damage within single images
- **High Performance**: Superior detection of large-sized road cracks with AP of 0.458 and competitive overall AP of 0.445
- **Comprehensive Dataset**: Novel dataset capturing diverse road damage types in individual images

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/road-damage-detection.git
cd road-damage-detection

# Create and activate conda environment
conda create -n road python=3.8 -y
conda activate road

# Install requirements
pip install -r requirements.txt

## Model Zoo

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
