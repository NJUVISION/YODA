# YODA: Yet Another One-step Diffusion-based Video Compressor

<div align="center">

**Xingchen Li, Junzhe Zhang, Junqi Shi, Ming Lu, Zhan Ma**

**Nanjing University**

[![Paper](https://img.shields.io/badge/Paper-TCSVT_2025-green)](https://github.com/NJUVISION/YODA) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
</div>

---

## 📢 Introduction

This repository is the official implementation of the paper **"YODA: Yet Another One-step Diffusion-based Video Compressor"**.

**YODA** is a novel neural video codec that achieves state-of-the-art perceptual quality while maintaining efficient inference. Unlike previous diffusion-based video codecs that rely on frozen 2D autoencoders and computationally expensive multi-step sampling, YODA introduces:

1.  **Temporal-Aware AutoEncoder (TA-AE):** A trainable frame autoencoder that embeds multiscale features from temporal references, yielding significantly more compact latent representations.
2.  **Conditional Latent Coder (CLC):** A motion-free latent codec that uses channel expansion (32→256) to implicitly model temporal correlations.
3.  **One-Step Linear DiT:** A lightweight Diffusion Transformer (DiT) with LoRA fine-tuning that performs denoising in a single step.

YODA consistently outperforms traditional standards (H.266/VVC) and recent neural baselines (DCVC-RT, PLVC, GLC-Video) on perceptual metrics like **LPIPS**, **DISTS**, **FID**, and **KID**.

---

## 🚀 Framework

<div align="center">
  <img src="assets/framework.png" width="800"/>
</div>
<br>

*Note: Please refer to Fig. 2 in the paper for the detailed architecture.*

The YODA pipeline consists of three main components:

1.  **Temporal-Aware AutoEncoder (TA-AE):** Instead of processing frames independently, TA-AE extracts multiscale temporal cues (5 scales) from the reference frame $\hat{x}_{t-1}$ to compress the current frame $x_t$ into a compact latent $l_t$.
2.  **Conditional Latent Coder (CLC):** Compresses the latent $l_t$ into bitstreams. It expands the channel dimension to capture rich temporal contexts, avoiding explicit optical flow estimation.
3.  **One-Step DiT Denoiser:** A linear DiT model processes the decoded (noisy) latent $\tilde{l}_t$. Using a "warm start" from the semantic latent, it restores high-frequency details in a single deterministic step.

---

## 🏆 Performance

YODA achieves superior perceptual quality compared to H.266/VVC (VTM-23.13) and SOTA neural codecs.

### Quantitative Results (BD-Rate vs. VTM-23.13)
*Lower BD-Rate is better.*

| Method | DISTS ↓ | LPIPS ↓ | KID ↓ | FID ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **DCVC-RT** | +0.62% | +4.53% | -21.05% | +23.91% |
| **PLVC** | -79.31% | -89.87% | -89.55% | -19.36% |
| **DiffVC** | -88.29% | -72.41% | -81.71% | N/A |
| **YODA (Ours)** | **-98.60%** | **-96.83%** | **-99.30%** | **-96.49%** |

*> Tested on UVG dataset. See paper for full tables on MCL-JCV and HEVC Class B.*

### Visual Comparison

<div align="center">
  <img src="assets/visual_comparison.png" width="800"/>
</div>

At low bitrates, YODA preserves fine textures (e.g., grass, building details) where VTM and DCVC-RT suffer from blurring or smoothing artifacts.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/NJUVISION/YODA.git](https://github.com/NJUVISION/YODA.git)
cd YODA

# Create environment (Requires PyTorch 2.x and CUDA)
conda create -n yoda python=3.9
conda activate yoda

# Install dependencies
pip install -r requirements.txt
