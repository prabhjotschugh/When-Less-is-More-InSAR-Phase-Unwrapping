import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from scipy.signal import welch
import time

# Import all models
from train.train_enhanced_unet import EnhancedInSAR_UNet, Config as ConfigUNet, prepare_datasets as prep_unet
from train.train_attention_unet import AttentionInSAR_UNet, Config as ConfigAttn, prepare_datasets as prep_attn
from train.train_hybrid import HybridMultiScaleUNet, Config as ConfigHybrid, prepare_datasets as prep_hybrid
from train.train_vanilla_unet import VanillaInSAR_UNet, Config as ConfigVanilla, prepare_datasets as prep_vanilla

# ==========================================
# PUBLICATION STYLE CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
})

OUTPUT_DIR = "ICLR_Final_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL CONFIGURATION
# ==========================================
MODEL_CONFIGS = {
    "Vanilla U-Net": {
        "config": ConfigVanilla(),
        "color": "#9467bd",  # Purple
        "linestyle": ":",
        "linewidth": 2.0,
        "alpha": 0.9,
        "prepare_fn": prep_vanilla,
        "order": 1,
    },
    "Attention U-Net": {
        "config": ConfigAttn(),
        "color": "#ff7f0e",  # Orange
        "linestyle": "--",
        "linewidth": 2.2,
        "alpha": 0.9,
        "prepare_fn": prep_attn,
        "order": 2,
    },
    "Enhanced U-Net": {
        "config": ConfigUNet(),
        "color": "#2ca02c",  # Green
        "linestyle": "-",
        "linewidth": 2.5,
        "alpha": 0.95,
        "prepare_fn": prep_unet,
        "order": 3,
    },
    "Hybrid Multi-Scale": {
        "config": ConfigHybrid(),
        "color": "#d62728",  # Red
        "linestyle": "-.",
        "linewidth": 2.2,
        "alpha": 0.9,
        "prepare_fn": prep_hybrid,
        "order": 4,
    },
}

# ==========================================
# LOAD ALL MODELS
# ==========================================
def load_models():
    models = {}
    stats_dict = {}
    
    for name, config in MODEL_CONFIGS.items():
        print(f"Loading {name}...")
        cfg = config["config"]
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE)
        
        if name == "Enhanced U-Net":
            model = EnhancedInSAR_UNet(
                cfg.IN_CHANNELS, cfg.OUT_CHANNELS,
                base_channels=cfg.BASE_CHANNELS, dropout=0.0
            )
        elif name == "Attention U-Net":
            model = AttentionInSAR_UNet(
                in_c=cfg.IN_CHANNELS, out_c=cfg.OUT_CHANNELS,
                base_channels=cfg.BASE_CHANNELS
            )
        elif name == "Hybrid Multi-Scale":
            model = HybridMultiScaleUNet(
                cfg.IN_CHANNELS, cfg.OUT_CHANNELS,
                base_channels=cfg.BASE_CHANNELS, dropout=0.0
            )
        else:  # Vanilla
            model = VanillaInSAR_UNet(
                cfg.IN_CHANNELS, cfg.OUT_CHANNELS,
                base_channels=cfg.BASE_CHANNELS, dropout=0.0
            )
        
        model.load_state_dict(checkpoint["model"])
        model.to(DEVICE)
        model.eval()
        
        models[name] = model
        stats_dict[name] = checkpoint["stats"]
        print(f"✓ {name} loaded")
    
    return models, stats_dict

# ==========================================
# EFFICIENCY METRICS
# ==========================================
def compute_efficiency_metrics(models):
    print("\n" + "="*70)
    print("COMPUTING EFFICIENCY METRICS")
    print("="*70 + "\n")
    
    metrics = {}
    
    # Get input channels from first config
    first_cfg = list(MODEL_CONFIGS.values())[0]["config"]
    in_channels = first_cfg.IN_CHANNELS
    print(f"Using {in_channels} input channels for efficiency analysis\n")
    
    dummy_input = torch.randn(1, in_channels, 256, 256).to(DEVICE)
    
    for name, model in models.items():
        print(f"Analyzing {name}...")
        
        # Model size
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = param_count * 4 / (1024**2)
        
        if param_count >= 1e6:
            params_str = f"{param_count/1e6:.2f}M"
        elif param_count >= 1e3:
            params_str = f"{param_count/1e3:.2f}K"
        else:
            params_str = f"{param_count}"
        
        # Manual FLOPs estimation
        flops = 2 * param_count * 256 * 256
        if flops >= 1e9:
            flops_str = f"{flops/1e9:.2f}G"
        elif flops >= 1e6:
            flops_str = f"{flops/1e6:.2f}M"
        else:
            flops_str = f"{flops/1e3:.2f}K"
        
        # Inference time
        times = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Measure
            for _ in range(100):
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(dummy_input)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time_ms = np.mean(times) * 1000
        std_time_ms = np.std(times) * 1000
        
        metrics[name] = {
            "parameters": param_count,
            "size_mb": param_size_mb,
            "flops": flops,
            "flops_str": flops_str,
            "params_str": params_str,
            "inference_time_ms": avg_time_ms,
            "inference_std_ms": std_time_ms,
        }
        
        print(f"  Parameters: {params_str}")
        print(f"  Size: {param_size_mb:.2f} MB")
        print(f"  FLOPs: {flops_str}")
        print(f"  Inference: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms\n")
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "efficiency_metrics.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("MODEL EFFICIENCY COMPARISON - ICLR ML4RS WORKSHOP\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<25} {'Params':<12} {'Size (MB)':<12} {'FLOPs':<12} {'Time (ms)':<15}\n")
        f.write("-"*80 + "\n")
        
        for name, m in metrics.items():
            f.write(f"{name:<25} {m['params_str']:<12} {m['size_mb']:<12.2f} "
                   f"{m['flops_str']:<12} {m['inference_time_ms']:<7.2f} ± {m['inference_std_ms']:<5.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Efficiency metrics saved\n")
    return metrics

# ==========================================
# INFERENCE ON TEST SET
# ==========================================
def run_inference(models, stats_dict):
    print("="*70)
    print("RUNNING INFERENCE ON TEST SET")
    print("="*70 + "\n")
    
    # Use Enhanced U-Net's dataset
    datasets, _ = prep_unet()
    test_loader = DataLoader(datasets["test"], batch_size=1, shuffle=False)
    
    results = {name: {"gt": [], "pred": [], "errors_cm": []} for name in models.keys()}
    vis_samples = []
    
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            sample_data = {"input": X.cpu().numpy()[0]}
            
            for name, model in models.items():
                pred = model(X)
                stats = stats_dict[name]
                
                pred_m = (pred * stats["y_std"].to(DEVICE) + 
                         stats["y_mean"].to(DEVICE)).cpu().numpy()[0, 0]
                target_m = (y * stats["y_std"].to(DEVICE) + 
                           stats["y_mean"].to(DEVICE)).cpu().numpy()[0, 0]
                
                results[name]["gt"].append(target_m)
                results[name]["pred"].append(pred_m)
                results[name]["errors_cm"].append((pred_m - target_m).flatten() * 100)
                
                sample_data[name] = pred_m
            
            sample_data["gt"] = target_m
            
            if i < 5:
                vis_samples.append(sample_data)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} samples...")
    
    print(f"\n✓ Inference complete on {len(test_loader)} samples\n")
    return results, vis_samples

# ==========================================
# COMBINED PSD PLOT 
# ==========================================
def plot_combined_psd(results):
    print("Generating combined PSD plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    
    # Ground Truth
    gt_flat = np.concatenate([g.flatten() for g in results["Enhanced U-Net"]["gt"]])
    f_gt, p_gt = welch(gt_flat, fs=1.0, nperseg=2048, noverlap=1024)
    
    # Plot with smoothing for better visualization
    ax.semilogy(f_gt, p_gt, label="Ground Truth", color="#000000", 
                linewidth=3.0, zorder=10, alpha=1.0)
    
    # Sort models by order for consistent legend
    sorted_models = sorted(MODEL_CONFIGS.items(), key=lambda x: x[1]["order"])
    
    for name, config in sorted_models:
        pred_flat = np.concatenate([p.flatten() for p in results[name]["pred"]])
        f_pred, p_pred = welch(pred_flat, fs=1.0, nperseg=2048, noverlap=1024)
        
        ax.semilogy(
            f_pred, p_pred,
            label=name,
            color=config["color"],
            linestyle=config["linestyle"],
            linewidth=config["linewidth"],
            alpha=config["alpha"],
            zorder=5
        )
    
    ax.set_xlabel("Spatial Frequency [cycles/pixel]", fontsize=13, fontweight='bold')
    ax.set_ylabel("Power Spectral Density [$m^2$/freq]", fontsize=13, fontweight='bold')
    
    # Set limits for better visualization
    ax.set_xlim(0, 0.5)
    ax.set_ylim(1e-7, 5e-2)
    
    # Grid
    ax.grid(True, which="both", alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.15, linestyle=':', linewidth=0.3)
    
    # Legend
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, 
              edgecolor='black', fancybox=False, ncol=1,
              fontsize=10)
    
    # Remove top and right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_psd.pdf"), bbox_inches="tight", dpi=400)
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_psd.png"), bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Combined PSD saved\n")

# ==========================================
# COMBINED CDF PLOT 
# ==========================================
def plot_combined_cdf(results):
    print("Generating combined CDF plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    
    # Sort models by order
    sorted_models = sorted(MODEL_CONFIGS.items(), key=lambda x: x[1]["order"])
    
    for name, config in sorted_models:
        errors_cm = np.concatenate(results[name]["errors_cm"])
        abs_errors = np.sort(np.abs(errors_cm))
        cdf = np.arange(len(abs_errors)) / float(len(abs_errors)) * 100
        
        ax.plot(
            abs_errors, cdf,
            label=name,
            color=config["color"],
            linestyle=config["linestyle"],
            linewidth=config["linewidth"],
            alpha=config["alpha"]
        )
    
    ax.set_xlabel("Absolute Error [cm]", fontsize=13, fontweight='bold')
    ax.set_ylabel("Cumulative Percentage [%]", fontsize=13, fontweight='bold')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 100)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.15, linestyle=':', linewidth=0.3)
    
    # Add reference lines at key thresholds
    for threshold in [1.0, 2.0]:
        ax.axvline(threshold, color='gray', linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)
    
    # Legend
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor='black', fancybox=False, ncol=1,
              fontsize=10)
    
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_cdf.pdf"), bbox_inches="tight", dpi=400)
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_cdf.png"), bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Combined CDF saved\n")

# ==========================================
# COMBINED VISUALIZATION (7 PANELS)
# ==========================================
def plot_combined_visualization(vis_samples):
    print("Generating combined visualization plots...")
    
    for idx, sample in enumerate(vis_samples):
        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        
        X = sample["input"]
        
        # 1. Wrapped Phase
        phase = np.arctan2(X[0], X[1])
        im0 = axes[0].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[0].set_title("(a) Wrapped Phase", fontsize=12, fontweight='bold')
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label('rad', fontsize=10)
        
        # 2. Coherence
        coh = X[2]
        im1 = axes[1].imshow(coh, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("(b) Coherence", fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label('γ', fontsize=10)
        
        # 3. Ground Truth
        gt = sample["gt"]
        vmin = gt.min()
        vmax = gt.max()
        
        for name in MODEL_CONFIGS.keys():
            vmin = min(vmin, sample[name].min())
            vmax = max(vmax, sample[name].max())
        
        im2 = axes[2].imshow(gt, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[2].set_title("(c) Ground Truth", fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_label('m', fontsize=10)
        
        # 4-7. Predictions (in order)
        model_order = ["Vanilla U-Net", "Attention U-Net", "Enhanced U-Net", "Hybrid Multi-Scale"]
        labels = ["(d) Vanilla", "(e) Attention", "(f) Enhanced", "(g) Hybrid"]
        
        for i, (name, label) in enumerate(zip(model_order, labels)):
            pred = sample[name]
            im = axes[3 + i].imshow(pred, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[3 + i].set_title(label, fontsize=12, fontweight='bold')
            cbar = plt.colorbar(im, ax=axes[3 + i], fraction=0.046, pad=0.04)
            cbar.set_label('m', fontsize=10)
        
        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"combined_sample_{idx}.pdf"),
            bbox_inches="tight", dpi=400
        )
        plt.close()
        
        print(f"  ✓ Sample {idx} saved")
    
    print(f"\n✓ All visualization plots saved\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "="*70)
    print("ICLR ML4RS WORKSHOP - COMBINED PUBLICATION RESULTS")
    print("="*70 + "\n")
    
    models, stats_dict = load_models()
    efficiency_metrics = compute_efficiency_metrics(models)
    results, vis_samples = run_inference(models, stats_dict)
    
    plot_combined_psd(results)
    plot_combined_cdf(results)
    plot_combined_visualization(vis_samples)
    
    print("Computing performance metrics...")
    
    # Sort models by order for consistent output
    sorted_models = sorted(MODEL_CONFIGS.items(), key=lambda x: x[1]["order"])
    
    with open(os.path.join(OUTPUT_DIR, "performance_metrics.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("MODEL PERFORMANCE COMPARISON - ICLR ML4RS WORKSHOP\n")
        f.write("="*80 + "\n\n")
        
        for name, config in sorted_models:
            errors_cm = np.concatenate(results[name]["errors_cm"])
            rmse = np.sqrt(np.mean(errors_cm**2))
            mae = np.mean(np.abs(errors_cm))
            
            f.write(f"\n{name}:\n")
            f.write(f"  RMSE:          {rmse:.3f} cm\n")
            f.write(f"  MAE:           {mae:.3f} cm\n")
            f.write(f"  Median Error:  {np.median(np.abs(errors_cm)):.3f} cm\n")
            f.write(f"  Std Dev:       {np.std(errors_cm):.3f} cm\n")
            f.write(f"  95th %ile:     {np.percentile(np.abs(errors_cm), 95):.3f} cm\n")
            f.write(f"  99th %ile:     {np.percentile(np.abs(errors_cm), 99):.3f} cm\n")
            f.write(f"  Precision@0.5: {np.sum(np.abs(errors_cm) < 0.5) / len(errors_cm) * 100:.2f}%\n")
            f.write(f"  Precision@1.0: {np.sum(np.abs(errors_cm) < 1.0) / len(errors_cm) * 100:.2f}%\n")
            f.write(f"  Precision@2.0: {np.sum(np.abs(errors_cm) < 2.0) / len(errors_cm) * 100:.2f}%\n")
    
    print(f"✓ Performance metrics saved\n")
    
    print("="*70)
    print("ALL RESULTS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  • combined_psd.pdf/png - Power spectral density comparison")
    print("  • combined_cdf.pdf/png - Cumulative error distribution")
    print("  • combined_sample_0-4.pdf - 7-panel visualizations")
    print("  • efficiency_metrics.txt - Model efficiency comparison")
    print("  • performance_metrics.txt - Model performance comparison")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()