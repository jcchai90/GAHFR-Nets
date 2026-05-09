# GAHFR-Nets

**GAHFR-Nets** (Gated Adaptive Hybrid Fusion Networks) is an improved architecture based on [CVnets](https://github.com/LduIIPLab/CVnets). This project introduces **Gated Fusion mechanisms** to optimize feature integration in hybrid CNN-ViT models.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)

## 📖 Introduction

Hybrid architectures combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have shown great potential in computer vision tasks. However, simply concatenating features from these two distinct backbones often leads to suboptimal feature utilization.

To address this, **GAHFR-Nets** proposes a Gated Fusion strategy that dynamically weights and integrates features, allowing the network to adaptively select the most informative representations from both CNN and ViT branches.

## 🚀 Key Improvements

We introduce two major improvements upon the baseline **CVnets**:

### 1. Gated Fusion for CNN-ViT Feature Integration
Instead of direct concatenation, we insert a **Gated Fusion Module** at the intersection of CNN and ViT features.
- **Mechanism:** The module learns to control the flow of information, assigning adaptive weights to CNN and ViT features based on their relevance.
- **Benefit:** This prevents the dominance of one modality over the other and enhances the representation capability of the fused features.

### 2. Gated Fusion for Channel Attention
We further refine the feature processing within the ViT branch.
- **Mechanism:** A Gated Fusion mechanism is applied to the concatenation of **Left and Right Channel Attention** maps.
- **Benefit:** This ensures that the complementary information from different attention heads is fused effectively, rather than just being stacked.

## 🛠️ Getting Started

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.8
- CUDA 》-12.8
- Grad-cam == 1.5.7
- GPU 2x 5090

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/jchal90/GAHFR-Nets.git
   cd GAHFR-Nets
