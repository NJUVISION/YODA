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

Below is a video reconstruction comparison against H.266/VVC (VTM) and DCVC-RT across different scenarios:

<!-- | Ground Truth (Original) | VTM-23.13 | **YODA (Ours)** |
| :---: | :---: | :---: |
| <video src="https://github.com/NJUVISION/YODA/assets/videos/Beauty_original.mp4" width="100%" controls muted autoplay loop></video> | <video src="https://github.com/NJUVISION/YODA/raw/main/assets/demo_sports_vtm.mp4" width="100%" controls muted autoplay loop></video> | <video src="https://github.com/NJUVISION/YODA/raw/main/assets/demo_sports_yoda.mp4" width="100%" controls muted autoplay loop></video> |
| *Raw Sequence* | *Blurry boundaries* | *Sharp structure preserved* | -->


<table width="100%">
  <tr>
    <th width="50%">Ground Truth (Original)</th>
    <th width="50%">YODA (bpp 0.012)</th>
  </tr>
  <tr>
    <td><video src="assets/videos/Beauty_original.mp4" style="width:100%" controls muted autoplay loop></video></td>
    <td><video src="assets/videos/Beauty_yoda_0.012435.mp4" style="width:100%" controls muted autoplay loop></video></td>
  </tr>
</table>

<table width="100%">
  <tr>
    <th width="50%">Ground Truth (Original)</th>
    <th width="50%">YODA (bpp 0.023)</th>
  </tr>
  <tr>
    <td><video src="assets/videos/Kimono1_original.mp4" style="width:100%" controls muted autoplay loop></video></td>
    <td><video src="assets/videos/Kimono1_yoda_0.023113.mp4" style="width:100%" controls muted autoplay loop></video></td>
  </tr>
</table>

---

## 📂 Data Preparation

We utilized the [Vimeo-90K](http://toflow.csail.mit.edu/) dataset for training and evaluated our model on the **UVG**, **MCL-JCV**, and **HEVC Class B** datasets.

---

## 🤝 Acknowledgment

We thank the authors of the following projects for their pioneering contributions and open-source efforts:

* **[DCVC-RT](https://github.com/microsoft/DCVC)**: Towards Practical Real-time Neural Video Compression.
* **[SANA](https://github.com/NVlabs/SANA)**: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.
* **[DC-AE](https://hanlab.mit.edu/projects/dc-ae)**: Deep Compression Autoencoder.

