# 🛰️ When Less Is More: Simplicity Beats Complexity for Physics-Constrained InSAR Phase Unwrapping

[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-ML4RS%20Workshop-blue)](https://ml4rs.github.io)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

### **⭐ Oral Presentation** at the 4th ICLR Workshop on Machine Learning for Remote Sensing (ML4RS), Rio de Janeiro, April 2026

**Authors:** [Prabhjot Singh](https://github.com/prabhjotschugh) (UT Austin · RediMinds Inc.) · Manmeet Singh (UT Austin · Western Kentucky University)

> **A note on the two result sets in this repo:** This project has two published versions - the original ICLR ML4RS 2026 workshop paper, and an extended IEEE GRSL journal version. During journal review, we retrained all four models under a single, fully standardized protocol (identical precision, dropout, weight decay, and loss across all models) to eliminate training confounds that existed in the workshop version. **Both code and results for both protocols are kept in this repo, clearly labeled, so nothing from either publication is lost.** The Key Results below are the standardized/GRSL numbers, since they reflect the final peer-reviewed protocol.

&nbsp;

### 📌 TL;DR

We challenge the trend of adopting complex computer vision architectures for InSAR phase unwrapping. Through a large-scale ablation study on a **global LiCSAR benchmark (20 frames, 39,724 patches, 651M pixels)**, we show that a **vanilla U-Net outperforms attention-based models by 34% in R²** with **2.5× faster inference**, because convolutional locality aligns better with the physics of smooth geophysical deformation than global attention.

### 🔍 Key Results - Standardized Protocol (IEEE GRSL)
 
All four models trained under an identical protocol: **FP32 precision, dropout=0.15, weight decay=1e-4, identical Huber+gradient physics loss, identical batch size/LR schedule/epochs**. Architecture is the only variable distinguishing the models.
 
| Model | RMSE (cm) ↓ | R² ↑ | P@1.0 (%) ↑ | Latency (ms) ↓ | Params (M) |
|---|---|---|---|---|---|
| **✅ Vanilla U-Net** | **1.070** | **0.810** | **83.73** | **2.74 ± 0.04** | 7.76 |
| Enhanced U-Net | 1.325 | 0.709 | 75.85 | 5.97 ± 0.04 | 8.29 |
| Attention U-Net | 1.439 | 0.656 | 76.57 | 6.81 ± 0.04 | 11.37 |
| Hybrid Multi-Scale | 1.639 | 0.555 | 69.52 | 6.74 ± 0.05 | 17.21 |
 
⚡ Vanilla U-Net achieves **<1cm error in 83.73% of predictions** vs only 69.52% for the Hybrid.  
🏃 At **2.74ms inference latency**, it is the only architecture that comfortably meets sub-100ms requirements for operational volcanic and seismic early-warning systems.


### 🔍 Results - ICLR ML4RS 2026 workshop version (mixed-precision protocol)</summary>

In this earlier protocol, Vanilla and Enhanced were trained in full FP32 while Attention and Hybrid used FP16 mixed precision, and dropout/weight-decay values were not identical across models - a training confound identified during GRSL peer review and fixed in the standardized protocol above. Kept here for transparency and to preserve the workshop record.


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
Attention and Hybrid models inject spurious high-frequency power at >0.3 cycles/pixel - physically unphysical artifacts that violate the smoothness of elastic surface deformation.


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

### 📂 Repository Structure
 
```
├── data/                       # LiCSAR frame metadata + dataset download script
├── figures/                    # Per-model + combined result figures, architecture diagrams
├── results/
│   ├── standardized/           # IEEE GRSL - standardized-protocol metrics
│   └── mixed_precision/        # ICLR ML4RS workshop - mixed-precision metrics
├── train/
│   ├── standardized/           # IEEE GRSL - shared base_config.py, identical protocol
│   └── mixed_precision/        # ICLR ML4RS workshop training scripts
├── visualize/                  # Evaluation + visualization scripts (per-model, combined,
│                                per-regime and per-frame breakdowns)
├── testing and resources/      # Exploratory notebooks, data preprocessing, download utilities
└── requirements.txt
```


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
**Standardized protocol (IEEE GRSL paper):**
```bash
python train/standardized/train_vanilla_unet.py      # Vanilla U-Net (recommended)
python train/standardized/train_enhanced_unet.py     # Enhanced U-Net
python train/standardized/train_attention_unet.py    # Attention U-Net
python train/standardized/train_hybrid.py            # Hybrid Multi-Scale
```
 
**Mixed-precision protocol (ICLR ML4RS workshop version):**
```bash
python train/mixed_precision/train_vanilla_unet.py
python train/mixed_precision/train_enhanced_unet.py
python train/mixed_precision/train_attention_unet.py
python train/mixed_precision/train_hybrid.py
```

#### 4. Evaluate & Visualize
```bash
python visualize/result_vanilla_unet.py      # Vanilla U-Net (recommended)
python visualize/result_enhanced_unet.py     # Enhanced U-Net
python visualize/result_attention_unet.py    # Attention U-Net
python visualize/result_hybrid.py            # Hybrid Multi-Scale
python visualize/result_combined.py          # Combined results
python visualize/per_regime_breakdown.py     # Breakdown by deformation regime
python visualize/combined_visualization.py   # Combined figure generation
```


### 🧠 Why Simpler Wins

PSD analysis reveals three failure mechanisms in complex models:

1. **Inductive bias mismatch** - Attention detects discrete boundaries; InSAR displacement has high spatial autocorrelation. Global attention introduces spurious long-range dependencies.
2. **Capacity-data mismatch** - Large models overfit frame-specific atmospheric noise rather than underlying physics.
3. **Multi-scale misapplication** - ASPP aggregation introduces aliasing artifacts in smooth-field regression.

### 📜 Citation
 
```bibtex
@inproceedings{
  singh2026when,
  title={When Less Is More: Simplicity Beats Complexity for Physics-Constrained In{SAR} Phase Unwrapping},
  author={Prabhjot Singh and Manmeet Singh},
  booktitle={4th ICLR Workshop on Machine Learning for Remote Sensing (Main Track)},
  year={2026},
  url={https://openreview.net/forum?id=liJldeR5ZX}
}
```

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
