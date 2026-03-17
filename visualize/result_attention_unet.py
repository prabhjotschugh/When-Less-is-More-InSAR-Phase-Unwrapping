import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy.signal import welch

# Import Attention U-Net
from train.train_attention_unet import AttentionInSAR_UNet, Config, prepare_datasets

# ==========================================
# 1. STYLE & PUBLICATION CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

cfg = Config()
cfg.FINAL_RESULTS_DIR = "final_results_attn_unet"

os.makedirs(cfg.FINAL_RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CORE PLOTTING FUNCTIONS (IDENTICAL)
# ==========================================

def plot_publication_psd(gt_list, pred_list, save_path):
    gt_flat = np.concatenate([g.flatten() for g in gt_list])
    pred_flat = np.concatenate([p.flatten() for p in pred_list])

    f_gt, p_gt = welch(gt_flat, fs=1.0, nperseg=1024)
    f_pred, p_pred = welch(pred_flat, fs=1.0, nperseg=1024)
    f_err, p_err = welch(gt_flat - pred_flat, fs=1.0, nperseg=1024)

    plt.figure(figsize=(7, 5))
    plt.semilogy(f_gt, p_gt, label="Ground Truth", linewidth=2)
    plt.semilogy(f_pred, p_pred, label="Attn-U-Net Prediction", linestyle="--", linewidth=2)
    plt.semilogy(f_err, p_err, label="Residual Error", alpha=0.6)

    plt.title("Spatial Power Spectral Density Comparison", fontweight="bold")
    plt.xlabel("Spatial Frequency [cycles/pixel]")
    plt.ylabel("Power Spectral Density [$m^2 / freq$]")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    sns.despine()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_inference_quad(X_sample, y_sample, pred_sample, sample_idx):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    phase = np.arctan2(X_sample[0], X_sample[1])
    im0 = axes[0].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("(a) Input Wrapped Phase")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    coh = X_sample[2]
    im1 = axes[1].imshow(coh, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("(b) Input Coherence ($\gamma$)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vmin = min(y_sample.min(), pred_sample.min())
    vmax = max(y_sample.max(), pred_sample.max())

    im2 = axes[2].imshow(y_sample, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[2].set_title("(c) Ground Truth LOS")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(pred_sample, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[3].set_title("(d) Attn-U-Net Prediction")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.FINAL_RESULTS_DIR, f"inference_sample_{sample_idx}.pdf"),
        bbox_inches="tight",
        dpi=400,
    )
    plt.close()


def plot_error_cdf(errors_cm, save_path):
    abs_errors = np.sort(np.abs(errors_cm))
    cdf = np.arange(len(abs_errors)) / float(len(abs_errors))

    plt.figure(figsize=(7, 5))
    plt.plot(abs_errors, cdf * 100, linewidth=2.5)

    for t in [1.0, 2.0]:
        pct = np.mean(abs_errors < t) * 100
        plt.axvline(t, linestyle=":", alpha=0.7)
        plt.text(t + 0.05, 20, f"{pct:.1f}% < {t}cm", rotation=90, fontweight="bold")

    plt.title("Error Cumulative Distribution Function (CDF)", fontweight="bold")
    plt.xlabel("Absolute Error [cm]")
    plt.ylabel("Percent of Pixels [%]")
    plt.xlim(0, 5)
    plt.ylim(0, 100)
    plt.grid(alpha=0.2)
    sns.despine()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# ==========================================
# 3. MAIN EVALUATION LOOP (MATCHED)
# ==========================================

def main():
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE)
    stats = checkpoint["stats"]

    model = AttentionInSAR_UNet(
        in_c=cfg.IN_CHANNELS,
        out_c=cfg.OUT_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    datasets, _ = prepare_datasets()
    test_loader = DataLoader(datasets["test"], batch_size=1, shuffle=False)

    all_gt_m, all_pred_m, all_errors_cm = [], [], []

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)

            pred_m = (pred * stats["y_std"].to(DEVICE) + stats["y_mean"].to(DEVICE)).cpu().numpy()[0, 0]
            target_m = (y * stats["y_std"].to(DEVICE) + stats["y_mean"].to(DEVICE)).cpu().numpy()[0, 0]

            all_gt_m.append(target_m)
            all_pred_m.append(pred_m)
            all_errors_cm.append((pred_m - target_m).flatten() * 100)

            if i < 5:
                plot_inference_quad(X.cpu().numpy()[0], target_m, pred_m, i)

    plot_publication_psd(
        all_gt_m,
        all_pred_m,
        os.path.join(cfg.FINAL_RESULTS_DIR, "figure_psd_comparison.png"),
    )

    plot_error_cdf(
        np.concatenate(all_errors_cm),
        os.path.join(cfg.FINAL_RESULTS_DIR, "figure_error_cdf.png"),
    )

    print("✔ Attention U-Net results generated successfully.")

if __name__ == "__main__":
    main()
