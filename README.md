# YODA: Yet Another One-step Diffusion-based Video Compressor

<div align="center">

**[Xingchen Li](https://vision.nju.edu.cn/24/f6/c29471a730358/page.htm), [Junzhe Zhang](https://vision.nju.edu.cn/19/81/c29471a792961/page.htm), [Junqi Shi](https://vision.nju.edu.cn/ee/a0/c29471a585376/page.htm), [Ming Lu](https://vision.nju.edu.cn/fc/da/c29470a457946/page.htm), [Zhan Ma](https://vision.nju.edu.cn/fc/d3/c29470a457939/page.htm)**

**Nanjing University**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
</div>

---

## 📢 Introduction

**YODA** is a novel neural video codec designed to achieve extreme perceptual quality with efficient inference speed.

While one-step diffusion models have excelled in image compression, applying them to video remains a challenge. YODA overcomes the limitations of traditional methods and existing deep learning baselines by introducing a **One-step Diffusion Transformer** and a **Temporal-Awareness** mechanism.

**Core Highlights:**
* **Perceptual Quality:** YODA consistently outperforms H.266/VVC and SOTA neural codecs (such as DCVC-RT and PLVC) on perceptual metrics including LPIPS, DISTS, FID, and KID.
* **One-Step Denoising:** Utilizing a lightweight linear DiT model, YODA performs denoising in a single step, significantly reducing the inference latency associated with diffusion models.
* **Temporal-Aware Design:** Unlike prior efforts that rely on frozen 2D autoencoders, YODA employs a trainable Temporal-Aware AutoEncoder (TA-AE) to fully exploit inter-frame correlations.

---

## 🚀 Framework

<div align="center">
  <img src="assets/framework.png" width="800" alt="YODA Framework"/>
</div>
<br>

YODA proposes an end-to-end unified design consisting of three key components[cite: 132]:

* **Temporal-Aware AutoEncoder (TA-AE):** Extracts multiscale features from reference frames to generate a compact latent representation.
* **Conditional Latent Coder (CLC):** Implicitly models motion within the feature space to perform efficient entropy coding.
* **Linear DiT Model:** Adopts a linear DiT for efficient one-step denoising.
---

## 🏆 Performance

YODA demonstrates superior performance across multiple datasets (UVG, HEVC-B, MCL-JCV), surpassing both traditional standards (VTM) and recent neural video codecs (DCVC-RT, DiffVC, PLVC)


<div align="center">
  <img src="assets/metrics.png" width="95%" alt="Perceptual Quality RD Curves (LPIPS, DISTS, FID, KID)"/>
<p><i>Figure: Perceptual quality performance comparisons on UVG, HEVC-B, and MCL-JCV datasets. Lower is better.</i></p>
  
  <br>
</div>

---
## 👁️ Visual Comparison

We provide an **interactive video comparison** (with sliding view) on our project page to demonstrate the visual reconstruction quality of **YODA** against the **Ground Truth**.

<div align="center">
  <a href="https://staceyxc.github.io/Yoda_videos/">
    <img src="https://img.shields.io/badge/🎥_View-Interactive_Video_Demos-2ea44f?style=for-the-badge&logo=github" alt="Video Demos">
  </a>
  <br>
  <br>
  <a href="https://staceyxc.github.io/Yoda_videos/">
    <strong>👉 Click here to compare YODA with Ground Truth 👈</strong>
  </a>
</div>
---

## 📂 Data Preparation

We utilized the [Vimeo-90K](http://toflow.csail.mit.edu/) dataset for training and evaluated our model on the **UVG**, **MCL-JCV**, and **HEVC Class B** datasets.

---

## 🤝 Acknowledgment

We thank the authors of the following projects for their pioneering contributions and open-source efforts:

* **[DCVC-RT](https://github.com/microsoft/DCVC)**: Towards Practical Real-time Neural Video Compression.
* **[SANA](https://github.com/NVlabs/SANA)**: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.
* **[DC-AE](https://hanlab.mit.edu/projects/dc-ae)**: Deep Compression Autoencoder.

