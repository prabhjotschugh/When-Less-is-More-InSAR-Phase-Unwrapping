# 🛰️ When Less Is More: Simplicity Beats Complexity for Physics-Constrained InSAR Phase Unwrapping

[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-ML4RS%20Workshop-blue)](https://ml4rs.github.io)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

### **⭐ Oral Presentation** at the 4th ICLR Workshop on Machine Learning for Remote Sensing (ML4RS), Rio de Janeiro, April 2026

**Authors:** [Prabhjot Singh](https://github.com/prabhjotschugh) (UT Austin · RediMinds Inc.) · Manmeet Singh (UT Austin · Western Kentucky University)

&nbsp;

### 📌 TL;DR

We challenge the trend of adopting complex computer vision architectures for InSAR phase unwrapping. Through a large-scale ablation study on a **global LiCSAR benchmark (20 frames, 39,724 patches, 651M pixels)**, we show that a **vanilla U-Net outperforms attention-based models by 34% in R²** with **2.5× faster inference**, because convolutional locality aligns better with the physics of smooth geophysical deformation than global attention.

### 🔍 Key Results

| Model | RMSE (cm) ↓ | R² ↑ | Latency (ms) ↓ | Params (M) |
|---|---|---|---|---|
| **✅ Vanilla U-Net** | **1.009** | **0.834** | **2.92 ± 0.06** | 7.76 |
| Enhanced U-Net | 1.149 | 0.786 | 6.35 ± 0.07 | 8.29 |
| Attention U-Net | 1.528 | 0.622 | 7.08 ± 0.07 | 11.37 |
| Hybrid Multi-Scale | 1.595 | 0.588 | 7.13 ± 0.17 | 17.21 |

⚡ Vanilla U-Net achieves **<1cm error in 88% of predictions** vs only 67.5% for the Hybrid.  
🏃 At **2.92ms inference latency**, it is the only architecture meeting sub-100ms requirements for real-time volcanic and seismic early-warning systems.



### 🌍 Global Benchmark

Our dataset spans **20 LiCSAR frames across 6 continents**, covering diverse volcanic, tectonic, and glacio-tectonic regimes (2020–2025). We use strict **frame-level stratified splitting** to prevent spatial leakage and evaluate true geographic generalization.

<img width="4778" height="1927" alt="map" src="https://github.com/user-attachments/assets/8dc55ba4-4ed2-41b6-92e0-4f5078c125c5" />



### 📈 Results Visualization

#### Representative Predictions Across Test Regimes

<img width="8025" height="899" alt="combined_sample_0" src="https://github.com/user-attachments/assets/57111e1c-e131-4e08-a197-d65953afd1cb" />

&nbsp;

<img width="8025" height="924" alt="combined_sample_1" src="https://github.com/user-attachments/assets/e1bd0aae-fdcd-4dea-b483-3558bda68b55" />

&nbsp;

<img width="8025" height="950" alt="combined_sample_2" src="https://github.com/user-attachments/assets/edd07c76-b310-40ab-838e-6ba699833e8c" />

&nbsp;

<img width="8025" height="924" alt="combined_sample_3" src="https://github.com/user-attachments/assets/0fefb554-b634-4b01-95df-9ad4d4f5d2e6" />

&nbsp;

<img width="8025" height="924" alt="combined_sample_4" src="https://github.com/user-attachments/assets/52e19e4e-b805-4dc6-ade4-1c4ad900f889" />



#### Power Spectral Density & Cumulative Error Distribution
Attention and Hybrid models inject spurious high-frequency power at >0.3 cycles/pixel — physically unphysical artifacts that violate the smoothness of elastic surface deformation.


### 🏗️ Models

We evaluate four U-Net variants of increasing complexity on an identical 4-level backbone:

- **V-UNet** - Vanilla U-Net (7.76M params) - *our best performer*
- **E-UNet** - Enhanced with Squeeze-Excitation blocks (8.29M params)
- **A-UNet** - Attention U-Net with bottleneck self-attention (11.37M params)
- **H-UNet** - Hybrid Multi-Scale with ASPP (17.21M params)


### 🤖 Pretrained Weights

Pre-trained model weights for all 4 architectures are available on Hugging Face:

👉 **[huggingface.co/Prabhjotschugh/InSAR-Phase-Unwrapping-Models](https://huggingface.co/Prabhjotschugh/InSAR-Phase-Unwrapping-Models)**

Download the `.pth` files and place them in the root directory before running evaluation scripts.

| Model | File | Size |
|---|---|---|
| ✅ Vanilla U-Net | `vanilla_unet_model.pth` | 93 MB |
| Enhanced U-Net | `enhanced_unet_model.pth` | 100 MB |
| Attention U-Net | `attention_unet_model.pth` | 137 MB |
| Hybrid Multi-Scale | `hybrid_model.pth` | 207 MB |


### 🚀 Getting Started

#### 1. Clone & install
```bash
git clone https://github.com/prabhjotschugh/When-Less-is-More-InSAR-Phase-Unwrapping.git
cd When-Less-is-More-InSAR-Phase-Unwrapping
pip install -r requirements.txt
```

#### 2. Download the dataset
```bash
python download_dataset.py
```
> ⚠️ Dataset is approximately **20GB**. Ensure sufficient disk space before downloading.

#### 3. Train
```bash
python train_vanilla_unet.py      # Vanilla U-Net (recommended)
python train_enhanced_unet.py     # Enhanced U-Net
python train_attention_unet.py    # Attention U-Net
python train_hybrid.py            # Hybrid Multi-Scale
```

#### 4. Evaluate & Visualize
```bash
python result_vanilla_unet.py      # Vanilla U-Net (recommended)
python result_enhanced_unet.py     # Enhanced U-Net
python result_attention_unet.py    # Attention U-Net
python result_hybrid.py            # Hybrid Multi-Scale
python result_combined.py          # Combined results 
```


### 🧠 Why Simpler Wins

PSD analysis reveals three failure mechanisms in complex models:

1. **Inductive bias mismatch** - Attention detects discrete boundaries; InSAR displacement has high spatial autocorrelation. Global attention introduces spurious long-range dependencies.
2. **Capacity-data mismatch** - Large models overfit frame-specific atmospheric noise rather than underlying physics.
3. **Multi-scale misapplication** - ASPP aggregation introduces aliasing artifacts in smooth-field regression.


### 📜 License
Code is licensed under the [MIT License](LICENSE). 
The paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


### 🙏 Acknowledgements

- LiCSAR dataset provided by the [COMET LiCSAR team](https://comet.nerc.ac.uk/COMET-LiCS-portal/)
- Training conducted on NVIDIA GH200 GPU (120GB VRAM)

--- 

<p align="center">
  <i>"Domain physics, not architectural sophistication, should guide ML4RS design. Less is more." 🛰️</i>
</p>
