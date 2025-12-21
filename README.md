# YODA: Yet Another One-step Diffusion-based Video Compressor

<div align="center">

**Xingchen Li, Junzhe Zhang, Junqi Shi, Ming Lu, Zhan Ma**

**Nanjing University**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
</div>

---

## 📢 Introduction

**YODA** 是一个新的神经视频编解码器，旨在实现极致的感知质量与高效的推理速度。

虽然基于扩散模型（Diffusion Models）的方法在图像压缩领域表现出色，但将其应用于视频一直面临挑战。YODA 通过引入 **一步式扩散变换器（One-step Diffusion Transformer）** 和 **时域感知（Temporal-Awareness）** 机制，打破了传统方法和现有深度学习基线的限制。

**核心亮点：**
* [cite_start]**极致的感知质量：** 在 LPIPS, DISTS, FID, KID 等感知指标上不仅超越了 H.266/VVC，也优于 DCVC-RT, PLVC 等 SOTA 神经编解码器 [cite: 8]。
* [cite_start]**一步去噪 (One-Step Denoising)：** 使用轻量级的线性 DiT 模型，仅需一步即可完成去噪，大幅降低了扩散模型的推理延迟 [cite: 7, 57]。
* [cite_start]**时域感知设计：** 不同于以往使用冻结 2D 自编码器的做法，YODA 设计了可训练的时域感知自编码器，充分利用帧间相关性 [cite: 51]。

---

## 🚀 Framework | 架构概览

<div align="center">
  <img src="assets/framework.png" width="800" alt="YODA Framework"/>
</div>
<br>

> [cite_start]**图解说明：** YODA 的整体架构 (a) 与传统方法 (b) 的对比。YODA 采用了可训练的时域感知自编码器 (TA-AE)、隐式建模运动的 Latent Codec 以及基于 DiT 的一步去噪器 [cite: 36, 47]。

### 核心组件 (Key Components)

YODA 由三个关键模块组成，共同协作以实现高效压缩与高质量重建：

#### [cite_start]1. Temporal-Aware AutoEncoder (TA-AE) [cite: 132, 145]
传统的扩散视频压缩通常使用预训练的、独立的 2D 自编码器，忽略了时间相关性。
* **创新点：** YODA 引入了一个可训练的 **时域感知编码器**。
* [cite_start]**机制：** 它通过一个提取器（Extractor）从参考帧 $\hat{x}_{t-1}$ 中提取多尺度特征（5个尺度），并将这些特征嵌入到当前帧 $x_t$ 的编码过程中 [cite: 138, 211]。
* [cite_start]**效果：** 这种设计使得潜在表示（Latent Representation）更加紧凑，通常能将 Latent 大小减少一半 [cite: 52]。

#### [cite_start]2. Conditional Latent Coder (CLC) [cite: 140, 215]
* [cite_start]**机制：** 这是一个基于上下文的潜在空间编解码器。为了更好地挖掘帧间信息，CLC 在内部将通道维度从 32 扩展到 256 [cite: 55, 142]。
* [cite_start]**优势：** 通过在特征空间捕捉丰富的时空上下文，CLC 能够隐式地建模运动信息，无需显式的光流估计模块，从而简化了系统设计 [cite: 54]。

#### [cite_start]3. One-Step DiT Denoiser [cite: 143, 229]
* [cite_start]**模型：** 采用基于 Transformer 的线性扩散模型（Linear DiT）[cite: 57]。
* **热启动 (Warm Start)：** 去噪过程并非从纯高斯噪声开始，而是从解码后的潜在特征 $\tilde{l}_t$ 开始。
* [cite_start]**效率：** 结合 LoRA 微调技术，YODA 仅需 **一步 (One-step)** 确定性采样即可恢复高频细节和纹理，避免了传统扩散模型繁琐的多步迭代 [cite: 79, 230]。

---

## 👁️ Visual Comparison | 可视化对比

YODA 在低码率下展现出了惊人的细节保留能力，尤其是在纹理和动态场景中。

<div align="center">
  <img src="assets/visual_comparison.png" width="800" alt="Visual Comparison"/>
</div>

* [cite_start]**H.266/VVC (VTM) & DCVC-RT:** 在低比特率下容易出现模糊和块效应，丢失纹理细节 [cite: 465]。
* [cite_start]**YODA (Ours):** 得益于生成式扩散先验，能够重建出逼真的草地纹理、建筑结构和运动物体的边缘，视觉效果更接近 Ground Truth [cite: 465]。

---

## 📂 Data Preparation

[cite_start]我们在训练中使用了 **Vimeo-90k** 数据集，并在 **UVG**, **MCL-JCV**, 和 **HEVC Class B** 数据集上进行了评估 [cite: 313]。

建议的数据集目录结构：