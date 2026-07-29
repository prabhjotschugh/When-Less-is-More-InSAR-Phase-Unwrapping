"""
PER-REGIME PERFORMANCE BREAKDOWN
Addresses Reviewer 2, comment #1:
  "The authors selected patches from multiple types of scenarios, but the
   visualized results are relatively limited... The authors may consider
   categorizing the selected patches according to scene type and conducting
   more detailed discussions for each category."

Your test set (per the paper) is geographically held out by frame:
  Hudson            -> glacio-tectonic
  Viedma            -> glacio-tectonic
  Deception Island  -> glacio-tectonic / volcanic (subduction-zone volcanic island)

This script re-runs inference for all four models but keeps each patch tagged
with its source frame_id, then aggregates RMSE/MAE/R2 PER FRAME and per
deformation-regime bucket. This gives you the table/figure Reviewer 2 is
asking for, and also directly supports Reviewer 2's comment #1 about large-
deformation regimes (frames with bigger LOS displacement ranges act as a
proxy for "large-scale deformation" patches).

Run this AFTER all four models have been trained (needs all 4 best.pth files).
"""

import os
import sys
import json
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_vanilla_unet import VanillaInSAR_UNet, Config as ConfigVanilla
from train.train_enhanced_unet import EnhancedInSAR_UNet, Config as ConfigEnhanced
from train.train_attention_unet import AttentionInSAR_UNet, Config as ConfigAttn
from train.train_hybrid import HybridMultiScaleUNet, Config as ConfigHybrid

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "per_regime_breakdown"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map frame_id -> (display name, deformation regime) based on the frames listed
# in the paper / your FRAMES_TO_DOWNLOAD dict. Adjust display names freely;
# the regime label is what matters for the reviewer-facing table.
FRAME_REGIME_MAP = {
    '001A_05031_131313': ('Calatrava', 'volcanic'),
    '002A_05136_020502': ('Azores (Pico/SaoJorge/Terceira)', 'volcanic'),
    '005A_07021_131313': ('Pico de Orizaba', 'volcanic'),
    '008A_12731_060000': ('Mayor Island', 'volcanic'),
    '008A_12836_151207': ('Okataina/White Island', 'volcanic'),
    '011A_04472_131313': ('Middle Gobi', 'continental_tectonic'),
    '010A_09915_111413': ('Sangeang Api / Ranakah', 'volcanic'),
    '010A_16318_111313': ('Ritmann / Melbourne', 'volcanic'),
    '020A_05163_131313': ('Black Rock Desert', 'continental_tectonic'),
    '020A_05362_131313': ('San Francisco Volcanic Field', 'volcanic'),
    '006D_05111_131313': ('Sabalan', 'volcanic'),
    '006D_05310_131313': ('Sahand', 'continental_tectonic'),
    '003D_09757_111111': ('Kelut/Semeru/Lawu', 'volcanic'),
    '007D_05293_151310': ('Methana', 'volcanic'),
    '009D_15291_161402': ('Deception Island', 'glacio_tectonic'),
    '010D_13610_131313': ('Cerro Hudson', 'glacio_tectonic'),
    '010D_13986_131310': ('Viedma/Lautaro/Aguilera', 'glacio_tectonic'),
    '012D_06537_131313': ('Durango Volcanic Field', 'volcanic'),
    '015D_02942_110000': ('Nunivak Island', 'volcanic'),
    '086D_04090_131308': ('Garibaldi Lake', 'glacio_tectonic'),
}

REGIME_COLORS = {
    'volcanic': '#d62728',
    'continental_tectonic': '#2ca02c',
    'glacio_tectonic': '#1f77b4',
}

MODEL_REGISTRY = {
    "Vanilla":   {"cls": VanillaInSAR_UNet,   "cfg": ConfigVanilla()},
    "Enhanced":  {"cls": EnhancedInSAR_UNet,  "cfg": ConfigEnhanced()},
    "Attention": {"cls": AttentionInSAR_UNet, "cfg": ConfigAttn()},
    "Hybrid":    {"cls": HybridMultiScaleUNet,"cfg": ConfigHybrid()},
}


def load_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        return np.nan_to_num(data, 0.0)


def rebuild_test_patches_with_frame_ids(cfg):
    """Re-extract test-split patches, but this time keep frame_id per patch
    (the training scripts don't persist this, so we recompute deterministically
    using the SAME split file -- patch order during extraction is the same
    walk order, so indices line up with dataset_splits_v2.json)."""
    all_patches_X, all_patches_y, patch_frame_ids = [], [], []

    downloaded_frames = sorted([d for d in os.listdir(cfg.DATA_DIR) if os.path.isdir(os.path.join(cfg.DATA_DIR, d))])

    for frame_id in downloaded_frames:
        frame_dir = os.path.join(cfg.DATA_DIR, frame_id)
        ifg_root = os.path.join(frame_dir, 'interferograms')
        meta_dir = os.path.join(frame_dir, 'metadata')
        if not os.path.exists(ifg_root):
            continue
        try:
            e = load_tif(os.path.join(meta_dir, f"{frame_id}.geo.E.tif"))
            n = load_tif(os.path.join(meta_dir, f"{frame_id}.geo.N.tif"))
            u = load_tif(os.path.join(meta_dir, f"{frame_id}.geo.U.tif"))
            mag = np.sqrt(e**2 + n**2 + u**2); mag[mag < 1e-6] = 1.0
            e, n, u = e/mag, n/mag, u/mag
        except Exception:
            continue

        for ifg_id in sorted(os.listdir(ifg_root)):
            ifg_dir = os.path.join(ifg_root, ifg_id)
            if not os.path.isdir(ifg_dir):
                continue
            try:
                wrap = load_tif(os.path.join(ifg_dir, f"{ifg_id}.geo.diff_pha.tif"))
                unw = load_tif(os.path.join(ifg_dir, f"{ifg_id}.geo.unw.tif"))
                coh = load_tif(os.path.join(ifg_dir, f"{ifg_id}.geo.cc.tif"))
                if coh.max() > 1.0:
                    coh = coh / 255.0
                los_gt = (unw * cfg.WAVELENGTH) / (4 * np.pi)

                H, W = wrap.shape
                for r in range(0, H - cfg.PATCH_SIZE, cfg.STRIDE):
                    for c in range(0, W - cfg.PATCH_SIZE, cfg.STRIDE):
                        p_coh = coh[r:r+cfg.PATCH_SIZE, c:c+cfg.PATCH_SIZE]
                        if np.mean(p_coh) < cfg.MIN_COHERENCE:
                            continue
                        p_wrap = wrap[r:r+cfg.PATCH_SIZE, c:c+cfg.PATCH_SIZE]
                        p_e = e[r:r+cfg.PATCH_SIZE, c:c+cfg.PATCH_SIZE]
                        p_n = n[r:r+cfg.PATCH_SIZE, c:c+cfg.PATCH_SIZE]
                        p_u = u[r:r+cfg.PATCH_SIZE, c:c+cfg.PATCH_SIZE]
                        p_los = los_gt[r:r+cfg.PATCH_SIZE, c:c+cfg.PATCH_SIZE]

                        x_tensor = np.stack([np.sin(p_wrap), np.cos(p_wrap), p_coh, p_e, p_n, p_u],
                                             axis=0).astype(np.float32)
                        y_tensor = p_los[None, :, :].astype(np.float32)

                        all_patches_X.append(x_tensor)
                        all_patches_y.append(y_tensor)
                        patch_frame_ids.append(frame_id)
            except Exception:
                continue

    return all_patches_X, all_patches_y, patch_frame_ids


def main():
    print("=" * 70)
    print("PER-REGIME / PER-FRAME PERFORMANCE BREAKDOWN (Reviewer 2, comment #1)")
    print("=" * 70)

    base_cfg = ConfigVanilla()  # all configs share DATA_DIR/SPLIT_FILE/patch params

    print("\nRe-extracting patches with frame_id tracking (deterministic order)...")
    all_X, all_y, frame_ids = rebuild_test_patches_with_frame_ids(base_cfg)
    print(f"✓ {len(all_X)} total patches re-extracted")

    with open(base_cfg.SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    test_idx = split_data['test_indices']

    test_frame_ids = [frame_ids[i] for i in test_idx]
    X_test = torch.tensor(np.stack([all_X[i] for i in test_idx]), dtype=torch.float32)
    y_test = torch.tensor(np.stack([all_y[i] for i in test_idx]), dtype=torch.float32)
    print(f"✓ Test set: {len(test_idx)} patches across "
          f"{len(set(test_frame_ids))} unique frames: {sorted(set(test_frame_ids))}")

    results_per_model = {}

    for model_name, reg in MODEL_REGISTRY.items():
        print(f"\nEvaluating {model_name}...")
        cfg = reg["cfg"]
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=DEVICE)
        stats = checkpoint['stats']

        model = reg["cls"](cfg.IN_CHANNELS, cfg.OUT_CHANNELS, base_channels=cfg.BASE_CHANNELS, dropout=0.0).to(DEVICE)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        X_norm = (X_test.to(DEVICE) - stats['X_mean'].to(DEVICE)) / stats['X_std'].to(DEVICE)

        per_patch_rmse_cm = []
        with torch.no_grad():
            for i in range(0, len(X_norm), 32):
                xb = X_norm[i:i+32]
                yb = y_test[i:i+32].to(DEVICE)
                pred_norm = model(xb)
                pred_m = pred_norm * stats['y_std'].to(DEVICE) + stats['y_mean'].to(DEVICE)
                diff_cm = (pred_m - yb).cpu().numpy() * 100
                for j in range(diff_cm.shape[0]):
                    per_patch_rmse_cm.append(np.sqrt(np.mean(diff_cm[j] ** 2)))

        results_per_model[model_name] = np.array(per_patch_rmse_cm)
        print(f"  ✓ {model_name}: overall patch-mean RMSE = {np.mean(per_patch_rmse_cm):.3f} cm")

    # ---- Aggregate per frame ----
    print("\n" + "=" * 70)
    print("PER-FRAME RMSE (cm) — for rebuttal table")
    print("=" * 70)

    frame_arr = np.array(test_frame_ids)
    unique_frames = sorted(set(test_frame_ids))

    table_rows = []
    for frame_id in unique_frames:
        mask = frame_arr == frame_id
        display_name, regime = FRAME_REGIME_MAP.get(frame_id, (frame_id, 'unknown'))
        row = {'frame_id': frame_id, 'display_name': display_name, 'regime': regime, 'n_patches': int(mask.sum())}
        for model_name, rmse_arr in results_per_model.items():
            row[f'{model_name}_rmse_cm'] = float(np.mean(rmse_arr[mask])) if mask.sum() > 0 else None
        table_rows.append(row)

    header = f"{'Frame':<30} {'Regime':<22} {'N':>6}" + "".join(f" {m:>12}" for m in MODEL_REGISTRY)
    print(header)
    print("-" * len(header))
    for row in table_rows:
        line = f"{row['display_name']:<30} {row['regime']:<22} {row['n_patches']:>6}"
        for model_name in MODEL_REGISTRY:
            val = row[f'{model_name}_rmse_cm']
            line += f" {val:>12.3f}" if val is not None else f" {'--':>12}"
        print(line)

    # ---- Aggregate per regime ----
    print("\n" + "=" * 70)
    print("PER-REGIME RMSE (cm) — the table to put directly in the rebuttal")
    print("=" * 70)

    regime_rows = []
    unique_regimes = sorted(set(r for _, r in FRAME_REGIME_MAP.values() if r in
                                 [FRAME_REGIME_MAP.get(f, (None, 'unknown'))[1] for f in unique_frames]))
    for regime in sorted(set(FRAME_REGIME_MAP.get(f, (f, 'unknown'))[1] for f in unique_frames)):
        regime_frames = [f for f in unique_frames if FRAME_REGIME_MAP.get(f, (f, 'unknown'))[1] == regime]
        mask = np.isin(frame_arr, regime_frames)
        row = {'regime': regime, 'n_patches': int(mask.sum())}
        for model_name, rmse_arr in results_per_model.items():
            row[f'{model_name}_rmse_cm'] = float(np.mean(rmse_arr[mask])) if mask.sum() > 0 else None
        regime_rows.append(row)

    header = f"{'Regime':<22} {'N':>6}" + "".join(f" {m:>12}" for m in MODEL_REGISTRY)
    print(header)
    print("-" * len(header))
    for row in regime_rows:
        line = f"{row['regime']:<22} {row['n_patches']:>6}"
        for model_name in MODEL_REGISTRY:
            val = row[f'{model_name}_rmse_cm']
            line += f" {val:>12.3f}" if val is not None else f" {'--':>12}"
        print(line)

    # ---- Save everything ----
    with open(os.path.join(OUTPUT_DIR, 'per_frame_breakdown.json'), 'w') as f:
        json.dump(table_rows, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'per_regime_breakdown.json'), 'w') as f:
        json.dump(regime_rows, f, indent=2)

    # ---- Bar chart: RMSE by regime, grouped by model ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    regimes_present = [row['regime'] for row in regime_rows]
    x = np.arange(len(regimes_present))
    width = 0.2
    for i, model_name in enumerate(MODEL_REGISTRY):
        vals = [row[f'{model_name}_rmse_cm'] for row in regime_rows]
        ax.bar(x + i * width, vals, width, label=model_name)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([r.replace('_', ' ').title() for r in regimes_present])
    ax.set_ylabel('Mean Patch RMSE (cm)')
    ax.set_title('Test-Set RMSE by Deformation Regime (Reviewer 2, comment #1)', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rmse_by_regime.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved: {OUTPUT_DIR}/per_frame_breakdown.json")
    print(f"✓ Saved: {OUTPUT_DIR}/per_regime_breakdown.json")
    print(f"✓ Saved: {OUTPUT_DIR}/rmse_by_regime.png")
    print("\nUse the per-regime table directly in the rebuttal letter and/or as a")
    print("new supplementary table in the revised manuscript (Reviewer 2, comment #1).")


if __name__ == "__main__":
    main()
