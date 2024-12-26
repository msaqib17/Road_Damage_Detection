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
