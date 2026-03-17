import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy.signal import welch

# Import your vanilla architecture and config
from train.train_vanilla_unet import VanillaInSAR_UNet, Config, prepare_datasets

# ==========================================
# 1. STYLE & PUBLICATION CONFIGURATION
# ==========================================
# Setting professional font styles (Serif/Times) - EXACT MATCH TO U-NET
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

# Update Config for new results directory
cfg = Config()
cfg.FINAL_RESULTS_DIR = "final_results_vanilla_unet"
os.makedirs(cfg.FINAL_RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. CORE PLOTTING FUNCTIONS
# ==========================================

def plot_publication_psd(gt_list, pred_list, save_path):
    """
    Standard PSD Comparison for ICML/ICLR. 
    Shows frequency-domain fidelity of the Vanilla emulator.
    EXACT MATCH to U-Net visualization style.
    """
    print("Generating Figure: PSD Comparison...")
    
    # Flatten and concatenate all test samples
    gt_flat = np.concatenate([g.flatten() for g in gt_list])
    pred_flat = np.concatenate([p.flatten() for p in pred_list])
    
    # Compute Welch PSD
    # Note: fs=1.0 assumes pixel-space frequency
    f_gt, p_gt = welch(gt_flat, fs=1.0, nperseg=1024)
    f_pred, p_pred = welch(pred_flat, fs=1.0, nperseg=1024)
    
    plt.figure(figsize=(7, 5))
    plt.semilogy(f_gt, p_gt, label='Ground Truth (Target)', color='#1f77b4', linewidth=2)
    plt.semilogy(f_pred, p_pred, label='Vanilla U-Net Prediction', color='#ff7f0e', linestyle='--', linewidth=2)
    
    # Plot residual error PSD (Standard in geophysical papers)
    f_err, p_err = welch(gt_flat - pred_flat, fs=1.0, nperseg=1024)
    plt.semilogy(f_err, p_err, label='Residual Error', color='#d62728', alpha=0.5, linewidth=1.5)

    plt.title("Spatial Power Spectral Density Comparison", fontweight='bold')
    plt.xlabel("Spatial Frequency [cycles/pixel]")
    plt.ylabel("Power Spectral Density [$m^2 / \\text{freq}$]")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, loc='upper right')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_inference_quad(X_sample, y_sample, pred_sample, sample_idx):
    """
    Generates the requested 4-panel inference results:
    Wrapped Phase | Coherence | GT LOS | Pred LOS
    EXACT MATCH to U-Net visualization style.
    """
    print(f"Generating Figure: Inference Quad (Sample {sample_idx})...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    # 1. Wrapped Phase (Channel 0/1 are sin/cos) - Use Twilight (Cyclic Colormap)
    phase = np.arctan2(X_sample[0], X_sample[1])
    im0 = axes[0].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("(a) Input Wrapped Phase", pad=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label('rad')
    
    # 2. Coherence (Channel 2)
    coh = X_sample[2]
    im1 = axes[1].imshow(coh, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("(b) Input Coherence ($\\gamma$)", pad=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Shared Scale for GT vs Pred (Critical for scientific honesty)
    vmin = min(y_sample.min(), pred_sample.min())
    vmax = max(y_sample.max(), pred_sample.max())
    
    # 3. Ground Truth LOS
    im2 = axes[2].imshow(y_sample, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2].set_title("(c) Ground Truth LOS", pad=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label('m')
    
    # 4. Vanilla U-Net Prediction
    im3 = axes[3].imshow(pred_sample, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[3].set_title("(d) Vanilla U-Net Prediction", pad=10)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04).set_label('m')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(cfg.FINAL_RESULTS_DIR, f"inference_sample_{sample_idx}.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()

def plot_error_cdf(errors_cm, save_path):
    """
    Bonus: Cumulative Distribution Function of Error.
    Highly effective for proving "Precision @ 1cm" and "95th percentile" metrics.
    EXACT MATCH to U-Net visualization style.
    """
    print("Generating Figure: Cumulative Error Distribution...")
    abs_errors = np.sort(np.abs(errors_cm))
    cdf = np.arange(len(abs_errors)) / float(len(abs_errors))
    
    plt.figure(figsize=(7, 5))
    plt.plot(abs_errors, cdf * 100, color='teal', linewidth=2.5)
    
    # Annotate key thresholds (0.5cm, 1.0cm, 2.0cm from your results)
    for threshold, color in zip([1.0, 2.0], ['#e74c3c', '#2ecc71']):
        pct = np.sum(abs_errors < threshold) / len(abs_errors) * 100
        plt.axvline(threshold, color=color, linestyle=':', alpha=0.8)
        plt.text(threshold+0.05, 20, f'{pct:.1f}% < {threshold}cm', color=color, fontweight='bold', rotation=90)
    
    plt.title("Error Cumulative Distribution Function (CDF)", fontweight='bold')
    plt.xlabel("Absolute Error [cm]")
    plt.ylabel("Percent of Pixels [%]")
    plt.xlim(0, 5) 
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.2)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================

def main():
    print(f"\n{'='*60}")
    print("VANILLA U-NET - PUBLICATION VISUALIZATION")
    print(f"{'='*60}\n")
    
    print(f"Loading best model and normalization stats...")
    
    # Load stats and best model
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE)
    stats = checkpoint['stats']
    
    model = VanillaInSAR_UNet(cfg.IN_CHANNELS, cfg.OUT_CHANNELS, 
                              base_channels=cfg.BASE_CHANNELS, dropout=0.0).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.5f}\n")
    
    # Prepare datasets (uses your existing prepare_datasets logic)
    datasets, _ = prepare_datasets()
    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)
    
    all_gt_m = []
    all_pred_m = []
    all_errors_cm = []
    
    print("Running inference on test set...")
    # Evaluation loop
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            
            # Denormalize to meters
            pred_m = (pred * stats['y_std'].to(DEVICE) + stats['y_mean'].to(DEVICE)).cpu().numpy()[0, 0]
            target_m = (y * stats['y_std'].to(DEVICE) + stats['y_mean'].to(DEVICE)).cpu().numpy()[0, 0]
            
            # Collect for PSD and CDF
            all_gt_m.append(target_m)
            all_pred_m.append(pred_m)
            all_errors_cm.append((pred_m - target_m).flatten() * 100)
            
            # Generate Inference Quad for the first 5 samples
            if i < 5:
                # Need original X (normalized but we can still extract sin/cos/coh)
                X_orig = X.cpu().numpy()[0]
                plot_inference_quad(X_orig, target_m, pred_m, i)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} samples...")
    
    print(f"✓ Inference complete on {len(test_loader)} samples\n")

    # 4. Generate Paper Figures
    print("Generating publication-quality figures...\n")
    
    plot_publication_psd(all_gt_m, all_pred_m, 
                        os.path.join(cfg.FINAL_RESULTS_DIR, "figure_psd_comparison.png"))
    
    errors_flat = np.concatenate(all_errors_cm)
    plot_error_cdf(errors_flat, 
                  os.path.join(cfg.FINAL_RESULTS_DIR, "figure_error_cdf.png"))
    
    # Compute and save final metrics
    rmse = np.sqrt(np.mean(errors_flat**2))
    mae = np.mean(np.abs(errors_flat))
    
    pred_flat = np.concatenate([p.flatten() for p in all_pred_m])
    target_flat = np.concatenate([t.flatten() for t in all_gt_m])
    
    from sklearn.metrics import r2_score
    r2 = r2_score(target_flat, pred_flat)
    
    # Save summary metrics
    summary_path = os.path.join(cfg.FINAL_RESULTS_DIR, "publication_metrics_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VANILLA U-NET - PUBLICATION METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: Vanilla U-Net (No SE blocks, No ASPP)\n")
        f.write(f"Training epochs: {checkpoint['epoch']+1}\n")
        f.write(f"Best validation loss: {checkpoint['best_val_loss']:.6f}\n\n")
        f.write("PRIMARY METRICS:\n")
        f.write(f"  RMSE:       {rmse:.3f} cm\n")
        f.write(f"  MAE:        {mae:.3f} cm\n")
        f.write(f"  R² Score:   {r2:.4f}\n\n")
        f.write("ERROR STATISTICS:\n")
        f.write(f"  Std Error:  {np.std(errors_flat):.3f} cm\n")
        f.write(f"  Median Err: {np.median(np.abs(errors_flat)):.3f} cm\n")
        f.write(f"  95th %ile:  {np.percentile(np.abs(errors_flat), 95):.3f} cm\n")
        f.write(f"  99th %ile:  {np.percentile(np.abs(errors_flat), 99):.3f} cm\n\n")
        f.write("PRECISION METRICS:\n")
        f.write(f"  Precision @ 0.5cm: {np.sum(np.abs(errors_flat) < 0.5) / len(errors_flat) * 100:.2f}%\n")
        f.write(f"  Precision @ 1.0cm: {np.sum(np.abs(errors_flat) < 1.0) / len(errors_flat) * 100:.2f}%\n")
        f.write(f"  Precision @ 2.0cm: {np.sum(np.abs(errors_flat) < 2.0) / len(errors_flat) * 100:.2f}%\n\n")
        f.write(f"Total test samples: {len(test_loader)}\n")
        f.write(f"Total pixels evaluated: {len(errors_flat):,}\n")

    print(f"\n{'='*60}")
    print(f"PUBLICATION RESULTS GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nLocation: {cfg.FINAL_RESULTS_DIR}")
    print(f"\nGenerated files:")
    print(f"  • figure_psd_comparison.png - Frequency domain analysis")
    print(f"  • figure_error_cdf.png - Cumulative error distribution")
    print(f"  • inference_sample_0-4.pdf - High-res quad visualizations")
    print(f"  • publication_metrics_summary.txt - All metrics for paper")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()