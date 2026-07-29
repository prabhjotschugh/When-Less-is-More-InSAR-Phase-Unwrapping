# ==========================================
# ENHANCED UNET TRAINING SCRIPT
# When Less Is More: Simplicity Beats Complexity for Physics-Constrained InSAR Phase Unwrapping
# IEEE GRSL Revision — standardized protocol (see base_config.py)
# ==========================================

import os
import sys
import json
import random
import warnings
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.base_config import BaseConfig, print_protocol_banner

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. CONFIGURATION (inherits standardized protocol from BaseConfig)
# ==========================================
class InSARDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Config(BaseConfig):
    """Enhanced U-Net config. ONLY paths/names differ from BaseConfig."""
    MODEL_PATH = os.path.join(BaseConfig.BASE_DIR, "insar_enhanced_unet_best.pth")
    CHECKPOINT_PATH = os.path.join(BaseConfig.BASE_DIR, "insar_enhanced_checkpoint.pth")
    RESULTS_DIR = os.path.join(BaseConfig.BASE_DIR, "results_enhanced")
    VIZ_DIR = os.path.join(RESULTS_DIR, "visualizations")
    TRAIN_VIZ_DIR = os.path.join(RESULTS_DIR, "training_viz")


cfg = Config()
set_seed(cfg.SEED)
for d in [cfg.BASE_DIR, cfg.DATA_DIR, cfg.RESULTS_DIR, cfg.VIZ_DIR, cfg.TRAIN_VIZ_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================================
# 2. EARLY STOPPING
# ==========================================
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-5, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = score < (self.best_score - self.min_delta) if self.mode == 'min' \
            else score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"  Best validation loss: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
        return False

    def state_dict(self):
        return {'counter': self.counter, 'best_score': self.best_score,
                'best_epoch': self.best_epoch, 'early_stop': self.early_stop}

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']


# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def save_training_visualization(X, y, pred, epoch, batch_idx, stats, save_dir):
    device = X.device
    pred_denorm = pred * stats['y_std'].to(device) + stats['y_mean'].to(device)
    y_denorm = y * stats['y_std'].to(device) + stats['y_mean'].to(device)

    X_np = X[0].cpu().numpy()
    y_np = y_denorm[0, 0].cpu().numpy()
    pred_np = pred_denorm[0, 0].cpu().numpy()

    wrapped_phase = np.arctan2(X_np[0], X_np[1])
    coherence = X_np[2]
    u_comp = X_np[5]
    error = y_np - pred_np

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(wrapped_phase, cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
    ax1.set_title('Input: Wrapped Phase', fontsize=12, fontweight='bold'); ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(coherence, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Input: Coherence', fontsize=12, fontweight='bold'); ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(u_comp, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_title('Input: LOS Up Component', fontsize=12, fontweight='bold'); ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    vmin_disp = min(y_np.min(), pred_np.min())
    vmax_disp = max(y_np.max(), pred_np.max())

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(y_np, cmap='RdBu_r', vmin=vmin_disp, vmax=vmax_disp)
    ax4.set_title('Ground Truth LOS Displacement', fontsize=12, fontweight='bold'); ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).set_label('Displacement (m)', fontsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(pred_np, cmap='RdBu_r', vmin=vmin_disp, vmax=vmax_disp)
    ax5.set_title('Enhanced U-Net Prediction', fontsize=12, fontweight='bold'); ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).set_label('Displacement (m)', fontsize=9)

    ax6 = fig.add_subplot(gs[1, 2])
    error_max = max(abs(error.min()), abs(error.max()))
    im6 = ax6.imshow(error, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
    ax6.set_title('Error (GT - Pred)', fontsize=12, fontweight='bold'); ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04).set_label('Error (m)', fontsize=9)

    rmse = np.sqrt(np.mean(error**2)); mae = np.mean(np.abs(error))
    fig.suptitle(f'Enhanced InSAR UNet - Epoch {epoch+1} | RMSE: {rmse*100:.2f} cm | MAE: {mae*100:.2f} cm',
                 fontsize=14, fontweight='bold', y=0.98)

    save_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}_batch_{batch_idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    return rmse, mae


def plot_training_curves(history, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(history['train']) + 1)
    ax.plot(epochs, history['train'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val'], 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (Huber + Gradient)')
    ax.set_title('Training and Validation Loss', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()


def create_multi_sample_visualization(model, dataloader, stats, epoch, save_dir, n_samples=5):
    device = next(model.parameters()).device
    model.eval()
    samples_collected = 0
    all_data = []

    with torch.no_grad():
        for X, y in dataloader:
            if samples_collected >= n_samples:
                break
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_denorm = pred * stats['y_std'].to(device) + stats['y_mean'].to(device)
            y_denorm = y * stats['y_std'].to(device) + stats['y_mean'].to(device)

            for i in range(X.shape[0]):
                if samples_collected >= n_samples:
                    break
                X_np = X[i].cpu().numpy()
                y_np = y_denorm[i, 0].cpu().numpy()
                pred_np = pred_denorm[i, 0].cpu().numpy()
                all_data.append({
                    'wrapped': np.arctan2(X_np[0], X_np[1]), 'coherence': X_np[2], 'u_comp': X_np[5],
                    'gt': y_np, 'pred': pred_np, 'error': y_np - pred_np
                })
                samples_collected += 1

    fig = plt.figure(figsize=(18, 4 * n_samples))
    gs = GridSpec(n_samples, 6, figure=fig, hspace=0.4, wspace=0.3)

    for row, data in enumerate(all_data):
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(data['wrapped'], cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
        if row == 0: ax1.set_title('Wrapped Phase', fontweight='bold')
        ax1.axis('off'); plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(data['coherence'], cmap='gray', vmin=0, vmax=1)
        if row == 0: ax2.set_title('Coherence', fontweight='bold')
        ax2.axis('off'); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(data['u_comp'], cmap='RdBu_r', vmin=-1, vmax=1)
        if row == 0: ax3.set_title('LOS Up', fontweight='bold')
        ax3.axis('off'); plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        vmin = min(data['gt'].min(), data['pred'].min()); vmax = max(data['gt'].max(), data['pred'].max())

        ax4 = fig.add_subplot(gs[row, 3])
        im4 = ax4.imshow(data['gt'], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        if row == 0: ax4.set_title('Ground Truth', fontweight='bold')
        ax4.axis('off'); plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[row, 4])
        im5 = ax5.imshow(data['pred'], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        if row == 0: ax5.set_title('Enhanced U-Net Prediction', fontweight='bold')
        ax5.axis('off'); plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[row, 5])
        error_max = max(abs(data['error'].min()), abs(data['error'].max()))
        im6 = ax6.imshow(data['error'], cmap='RdBu_r', vmin=-error_max, vmax=error_max)
        if row == 0: ax6.set_title('Error (GT - Pred)', fontweight='bold')
        ax6.axis('off'); plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        ax1.text(-0.1, 0.5, f'Sample {row+1}', transform=ax1.transAxes, fontweight='bold', rotation=90, va='center')

    fig.suptitle(f'Enhanced InSAR UNet - Multi-Frame - Epoch {epoch+1}', fontweight='bold', y=0.995)
    save_path = os.path.join(save_dir, f'multi_sample_epoch_{epoch+1:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()


# ==========================================
# 4. DATA PROCESSING (identical pipeline across all 4 scripts)
# ==========================================
def load_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        return np.nan_to_num(data, 0.0)


def extract_all_patches_with_metadata():
    print(f"\n{'='*60}\nSTEP 2: EXTRACTING ALL PATCHES\n{'='*60}")
    all_patches_X, all_patches_y, patch_metadata = [], [], []

    downloaded_frames = [d for d in os.listdir(cfg.DATA_DIR) if os.path.isdir(os.path.join(cfg.DATA_DIR, d))]
    print(f"Found {len(downloaded_frames)} frames")

    for frame_id in tqdm(downloaded_frames, desc="Extracting patches from all frames"):
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

        for ifg_id in os.listdir(ifg_root):
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
                        patch_metadata.append({'frame_id': frame_id, 'ifg_id': ifg_id, 'row_start': r, 'col_start': c})
            except Exception:
                continue

    print(f"✓ Extracted {len(all_patches_X)} valid patches")
    return all_patches_X, all_patches_y, patch_metadata


def create_stratified_splits(all_patches_X, all_patches_y, patch_metadata):
    print(f"\n{'='*60}\nSTEP 3: CREATING STRATIFIED SPLITS\n{'='*60}")
    n_total = len(all_patches_X)
    indices = np.arange(n_total)
    activities = np.array([np.abs(y).max() for y in all_patches_y])

    activity_bins = np.percentile(activities, [33, 67])
    low_activity = indices[activities <= activity_bins[0]]
    med_activity = indices[(activities > activity_bins[0]) & (activities <= activity_bins[1])]
    high_activity = indices[activities > activity_bins[1]]

    np.random.shuffle(low_activity); np.random.shuffle(med_activity); np.random.shuffle(high_activity)

    def split_indices(idx_arr):
        n = len(idx_arr); n_train = int(n * 0.70); n_val = int(n * 0.15)
        return {'train': idx_arr[:n_train], 'val': idx_arr[n_train:n_train+n_val], 'test': idx_arr[n_train+n_val:]}

    low_splits, med_splits, high_splits = split_indices(low_activity), split_indices(med_activity), split_indices(high_activity)

    train_idx = np.concatenate([low_splits['train'], med_splits['train'], high_splits['train']])
    val_idx = np.concatenate([low_splits['val'], med_splits['val'], high_splits['val']])
    test_idx = np.concatenate([low_splits['test'], med_splits['test'], high_splits['test']])

    np.random.shuffle(train_idx); np.random.shuffle(val_idx); np.random.shuffle(test_idx)

    splits = {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}
    with open(cfg.SPLIT_FILE, 'w') as f:
        json.dump({'train_indices': splits['train'], 'val_indices': splits['val'], 'test_indices': splits['test'],
                   'metadata': {'total_patches': n_total, 'train_count': len(train_idx),
                                'val_count': len(val_idx), 'test_count': len(test_idx)}}, f, indent=2)

    print(f"✓ Splits saved | Train: {len(train_idx)} Val: {len(val_idx)} Test: {len(test_idx)}")
    return splits


def prepare_datasets():
    if cfg.RESUME_TRAINING and os.path.exists(cfg.CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location='cpu')
        stats_loaded = checkpoint.get('stats', None)
    else:
        stats_loaded = None

    all_patches_X, all_patches_y, patch_metadata = extract_all_patches_with_metadata()
    if len(all_patches_X) == 0:
        print("⚠ ERROR: No patches extracted!")
        return {}, {}

    if not os.path.exists(cfg.SPLIT_FILE):
        splits = create_stratified_splits(all_patches_X, all_patches_y, patch_metadata)
    else:
        print(f"Loading existing splits from {cfg.SPLIT_FILE}")
        with open(cfg.SPLIT_FILE, 'r') as f:
            split_data = json.load(f)
            splits = {'train': split_data['train_indices'], 'val': split_data['val_indices'], 'test': split_data['test_indices']}

    X_all = torch.tensor(np.stack(all_patches_X), dtype=torch.float32)
    y_all = torch.tensor(np.stack(all_patches_y), dtype=torch.float32)
    print(f"✓ X shape {X_all.shape}, y shape {y_all.shape}")

    if stats_loaded is not None:
        stats = stats_loaded
    else:
        train_idx = splits['train']
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        stats = {'X_mean': X_train.mean(dim=(0, 2, 3), keepdim=True),
                 'X_std': X_train.std(dim=(0, 2, 3), keepdim=True) + 1e-8,
                 'y_mean': y_train.mean(), 'y_std': y_train.std() + 1e-8}
        print(f"  Computed normalization stats: y_mean={stats['y_mean'].item():.6f}, y_std={stats['y_std'].item():.6f}")

    datasets = {}
    for split_name, indices in splits.items():
        X_split, y_split = X_all[indices], y_all[indices]
        X_norm = (X_split - stats['X_mean']) / stats['X_std']
        y_norm = (y_split - stats['y_mean']) / stats['y_std']
        datasets[split_name] = InSARDataset(X_norm, y_norm)
        print(f"  {split_name}: {len(X_norm)} patches")

    return datasets, stats


# ==========================================
# 5. ENHANCED MODEL ARCHITECTURE (SE blocks + attention gates — the architectural variable)
# ==========================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(self.bn2(self.conv2(out)))
        return self.act2(out + identity)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, max(channels // reduction, 4), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(channels // reduction, 4), channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(s))))
        return x * s


class AttentionGate(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gating_ch, inter_ch, kernel_size=1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class EnhancedInSAR_UNet(nn.Module):
    """Enhanced U-Net: adds Squeeze-Excitation blocks for channel-wise recalibration
    after each encoder stage, plus attention gates at skip connections. 8.29M params
    at base_channels=32. Dropout is now identical to Vanilla/Attention/Hybrid (cfg.DROPOUT)."""
    def __init__(self, in_c=6, out_c=1, base_channels=32, dropout=0.15):
        super().__init__()
        b = base_channels

        self.enc1 = ResidualConvBlock(in_c, b, dropout=dropout)
        self.enc2 = ResidualConvBlock(b, b*2, dropout=dropout)
        self.enc3 = ResidualConvBlock(b*2, b*4, dropout=dropout)
        self.enc4 = ResidualConvBlock(b*4, b*8, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(b*8, b*16, dropout=dropout),
            SEBlock(b*16, reduction=8)
        )

        self.up4 = nn.ConvTranspose2d(b*16, b*8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(in_ch=b*8, gating_ch=b*8, inter_ch=b*4)
        self.dec4 = ResidualConvBlock(b*16, b*8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(b*8, b*4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(in_ch=b*4, gating_ch=b*4, inter_ch=b*2)
        self.dec3 = ResidualConvBlock(b*8, b*4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(b*4, b*2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(in_ch=b*2, gating_ch=b*2, inter_ch=b)
        self.dec2 = ResidualConvBlock(b*4, b*2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(b*2, b, kernel_size=2, stride=2)
        self.att1 = AttentionGate(in_ch=b, gating_ch=b, inter_ch=b//2)
        self.dec1 = ResidualConvBlock(b*2, b, dropout=dropout)

        self.se1 = SEBlock(b, reduction=4)
        self.se2 = SEBlock(b*2, reduction=8)
        self.se3 = SEBlock(b*4, reduction=8)
        self.se4 = SEBlock(b*8, reduction=16)

        self.out_conv = nn.Sequential(
            nn.Conv2d(b, b//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(b//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(b//2, out_c, kernel_size=1)
        )
        self.residual_head = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x1 = self.se1(self.enc1(x))
        x2 = self.se2(self.enc2(self.pool(x1)))
        x3 = self.se3(self.enc3(self.pool(x2)))
        x4 = self.se4(self.enc4(self.pool(x3)))

        b = self.bottleneck(self.pool(x4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(x4, d4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(x3, d3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(x2, d2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(x1, d1)], dim=1))

        out = self.out_conv(d1)
        res = self.residual_head(x)
        if res.shape[2:] != out.shape[2:]:
            res = F.interpolate(res, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + res


# ==========================================
# 6. PHYSICS LOSS (IDENTICAL formulation for ALL 4 models)
# ==========================================
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def calc_physics_loss(pred, target, cfg):
    huber = nn.HuberLoss(delta=cfg.HUBER_DELTA)(pred, target)
    device = pred.device
    grad_x = F.conv2d(pred, SOBEL_X.to(device), padding=1)
    grad_y = F.conv2d(pred, SOBEL_Y.to(device), padding=1)
    t_grad_x = F.conv2d(target, SOBEL_X.to(device), padding=1)
    t_grad_y = F.conv2d(target, SOBEL_Y.to(device), padding=1)
    grad_loss = F.mse_loss(grad_x, t_grad_x) + F.mse_loss(grad_y, t_grad_y)
    return huber + cfg.LAMBDA_GRAD * grad_loss


# ==========================================
# 7. TRAINING WITH CHECKPOINT RESUMING
# ==========================================
def save_checkpoint(epoch, model, optimizer, scheduler, early_stopping, history, best_loss, stats, total_steps):
    checkpoint = {
        'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'early_stopping': early_stopping.state_dict(),
        'history': history, 'best_loss': best_loss, 'stats': stats,
        'config': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        'total_steps': total_steps
    }
    torch.save(checkpoint, cfg.CHECKPOINT_PATH)
    print(f"  ✓ Checkpoint saved (Total steps: {total_steps})")


def train_model(datasets, stats):
    print_protocol_banner("ENHANCED U-NET", cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader = DataLoader(datasets['train'], batch_size=cfg.BATCH_SIZE, shuffle=True,
                               pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(datasets['val'], batch_size=cfg.BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=2, persistent_workers=True)

    model = EnhancedInSAR_UNet(cfg.IN_CHANNELS, cfg.OUT_CHANNELS,
                                base_channels=cfg.BASE_CHANNELS, dropout=cfg.DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
                             betas=(0.9, 0.999), eps=1e-8)

    start_epoch, best_loss, total_steps = 0, float('inf'), 0
    history = {'train': [], 'val': [], 'lr': []}
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE, min_delta=cfg.MIN_DELTA)

    if cfg.RESUME_TRAINING and os.path.exists(cfg.CHECKPOINT_PATH):
        try:
            print(f"\n{'='*60}\nRESUMING FROM CHECKPOINT\n{'='*60}")
            checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            history = checkpoint['history']
            total_steps = checkpoint.get('total_steps', start_epoch * len(train_loader))
            if 'early_stopping' in checkpoint:
                early_stopping.load_state_dict(checkpoint['early_stopping'])
            print(f"✓ Resumed from epoch {start_epoch} | Best val: {best_loss:.5f} | Steps: {total_steps}")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}\n  Starting fresh...")
            model.apply(init_weights)
            start_epoch, best_loss, history, total_steps = 0, float('inf'), {'train': [], 'val': [], 'lr': []}, 0
    else:
        print("\nStarting fresh training...")
        model.apply(init_weights)

    total_training_steps = cfg.EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, total_steps=total_training_steps,
        pct_start=0.1, anneal_strategy='cos', last_epoch=total_steps - 1
    )

    if cfg.RESUME_TRAINING and os.path.exists(cfg.CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"✓ Scheduler resumed. LR: {scheduler.get_last_lr()[0]:.6f}\n")
        except Exception as e:
            print(f"⚠ Could not load scheduler state: {e}\n")

    criterion_val = nn.HuberLoss(delta=cfg.HUBER_DELTA)

    print(f"Training: {start_epoch} -> {cfg.EPOCHS} epochs | Patience: {cfg.EARLY_STOP_PATIENCE}\n")

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}", leave=False)

        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = calc_physics_loss(pred, y, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_steps += 1

            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

            if (epoch + 1) % cfg.VIZ_FREQUENCY == 0 and batch_idx == 0:
                with torch.no_grad():
                    model.eval()
                    save_training_visualization(X, y, pred, epoch, batch_idx, stats, cfg.TRAIN_VIZ_DIR)
                    model.train()

        avg_train = t_loss / len(train_loader)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                v_loss += criterion_val(pred, y).item()
        avg_val = v_loss / len(val_loader)

        history['train'].append(avg_train); history['val'].append(avg_val); history['lr'].append(scheduler.get_last_lr()[0])

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train: {avg_train:.5f} | Val: {avg_val:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'stats': stats, 'config': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
                        'best_val_loss': best_loss}, cfg.MODEL_PATH)
            print(f"  ✓ Best model saved (Val Loss: {best_loss:.5f})")

        if (epoch + 1) % cfg.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping, history, best_loss, stats, total_steps)

        if (epoch + 1) % cfg.VIZ_FREQUENCY == 0:
            create_multi_sample_visualization(model, val_loader, stats, epoch, cfg.TRAIN_VIZ_DIR, n_samples=cfg.N_VIZ_SAMPLES)

        if (epoch + 1) % 50 == 0 or (epoch + 1) == cfg.EPOCHS:
            plot_training_curves(history, os.path.join(cfg.RESULTS_DIR, 'training_curves.png'))

        if early_stopping(avg_val, epoch):
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping, history, best_loss, stats, total_steps)
            print(f"\n✓ Training stopped early at epoch {epoch+1}")
            break

    save_checkpoint(epoch, model, optimizer, scheduler, early_stopping, history, best_loss, stats, total_steps)
    plot_training_curves(history, os.path.join(cfg.RESULTS_DIR, 'training_curves_final.png'))
    print(f"\n✓ Training completed! Best Val Loss: {best_loss:.5f}")
    return history


# ==========================================
# 8. EVALUATION
# ==========================================
def compute_power_spectrum_density(signal):
    try:
        if signal.ndim == 2:
            signal = signal.flatten()
        nperseg = len(signal) // 4 if len(signal) < 256 else 256
        freqs, psd = welch(signal, nperseg=nperseg)
        return freqs, psd
    except Exception:
        return None, None


def plot_psd_comparison(gt_list, pred_list, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gt_all = np.concatenate([g.flatten() for g in gt_list])
    pred_all = np.concatenate([p.flatten() for p in pred_list])
    error_all = gt_all - pred_all

    ax = axes[0, 0]
    freqs_gt, psd_gt = compute_power_spectrum_density(gt_all)
    if freqs_gt is not None:
        ax.semilogy(freqs_gt, psd_gt, 'b-', linewidth=2, label='Ground Truth')
        ax.set_title('Ground Truth PSD', fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    freqs_pred, psd_pred = compute_power_spectrum_density(pred_all)
    if freqs_pred is not None:
        ax.semilogy(freqs_pred, psd_pred, 'r-', linewidth=2, label='Prediction')
        ax.set_title('Prediction PSD', fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 0]
    if freqs_gt is not None and freqs_pred is not None:
        ax.semilogy(freqs_gt, psd_gt, 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax.semilogy(freqs_pred, psd_pred, 'r--', linewidth=2, label='Prediction', alpha=0.7)
        ax.set_title('PSD Comparison', fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 1]
    freqs_err, psd_err = compute_power_spectrum_density(error_all)
    if freqs_err is not None:
        ax.semilogy(freqs_err, psd_err, 'g-', linewidth=2, label='Error')
        ax.set_title('Error PSD', fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()


def evaluate(datasets, stats):
    print(f"\n{'='*60}\nSTEP 8: COMPREHENSIVE EVALUATION — ENHANCED U-NET\n{'='*60}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    model = EnhancedInSAR_UNet(cfg.IN_CHANNELS, cfg.OUT_CHANNELS,
                                base_channels=cfg.BASE_CHANNELS, dropout=0.0).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded best model from epoch {checkpoint['epoch']+1} | Best val loss: {checkpoint['best_val_loss']:.5f}")

    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)
    errors_cm, all_predictions, all_targets = [], [], []

    print("\nGenerating predictions on test set...")
    with torch.no_grad():
        for idx, (X, y) in enumerate(tqdm(test_loader)):
            X, y = X.to(device), y.to(device)
            pred_norm = model(X)
            pred_m = pred_norm * stats['y_std'].to(device) + stats['y_mean'].to(device)
            target_m = y * stats['y_std'].to(device) + stats['y_mean'].to(device)
            diff_cm = (pred_m - target_m).cpu().numpy() * 100
            errors_cm.extend(diff_cm.flatten())
            all_predictions.append(pred_m.cpu().numpy())
            all_targets.append(target_m.cpu().numpy())
            if idx < 10:
                save_training_visualization(X, y, pred_norm, idx, 0, stats, cfg.VIZ_DIR)

    errors = np.array(errors_cm)
    pred_flat = np.concatenate([p.flatten() for p in all_predictions])
    target_flat = np.concatenate([t.flatten() for t in all_targets])

    rmse = np.sqrt(np.mean(errors**2)); mae = np.mean(np.abs(errors)); r2 = r2_score(target_flat, pred_flat)
    precision = np.sum(np.abs(errors) < 1.0) / len(errors)

    print(f"\n{'='*60}\nFINAL TEST METRICS — ENHANCED U-NET\n{'='*60}")
    print(f"RMSE: {rmse:.3f} cm | MAE: {mae:.3f} cm | R²: {r2:.4f}")
    print(f"Std: {np.std(errors):.3f} cm | Median: {np.median(np.abs(errors)):.3f} cm")
    print(f"95th %: {np.percentile(np.abs(errors), 95):.3f} cm | Precision@1cm: {precision*100:.2f}%")
    print(f"{'='*60}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(errors, bins=100, range=(-10, 10), density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_title(f'Error Distribution\nRMSE: {rmse:.2f} cm | MAE: {mae:.2f} cm | R²: {r2:.3f}', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    sorted_errors = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, linewidth=2, color='darkgreen')
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50th %ile')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90th %ile')
    ax.set_title('Cumulative Error Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_xlim(0, min(20, sorted_errors.max()))

    plt.tight_layout(); plt.savefig(os.path.join(cfg.RESULTS_DIR, 'error_analysis.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(pred_flat) > 50000:
        idxs = np.random.choice(len(pred_flat), 50000, replace=False)
        pred_sample, target_sample = pred_flat[idxs], target_flat[idxs]
    else:
        pred_sample, target_sample = pred_flat, target_flat
    ax.hexbin(target_sample, pred_sample, gridsize=50, cmap='Blues', mincnt=1)
    min_val, max_val = min(target_sample.min(), pred_sample.min()), max(target_sample.max(), pred_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_title(f'Prediction vs Ground Truth\nRMSE: {rmse:.2f} cm | R²: {r2:.3f}', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    plt.tight_layout(); plt.savefig(os.path.join(cfg.RESULTS_DIR, 'scatter_pred_vs_gt.png'), dpi=150); plt.close()

    plot_psd_comparison(all_targets, all_predictions, os.path.join(cfg.RESULTS_DIR, 'power_spectrum_density.png'))
    create_multi_sample_visualization(model, test_loader, stats, -1, cfg.VIZ_DIR, n_samples=min(10, len(datasets['test'])))

    metrics_file = os.path.join(cfg.RESULTS_DIR, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\nEnhanced InSAR U-Net - Test Metrics\n" + "="*60 + "\n\n")
        f.write(f"Epochs: {checkpoint['epoch']+1}\nBest val loss: {checkpoint['best_val_loss']:.6f}\n\n")
        f.write(f"RMSE: {rmse:.3f} cm\nMAE: {mae:.3f} cm\nR²: {r2:.4f}\n")
        f.write(f"Std: {np.std(errors):.3f} cm\nMedian: {np.median(np.abs(errors)):.3f} cm\n")
        f.write(f"95th %: {np.percentile(np.abs(errors), 95):.3f} cm\nPrecision@1cm: {precision*100:.2f}%\n")
        f.write(f"Test patches: {len(datasets['test'])}\n")
    print(f"✓ Metrics saved to {metrics_file}\n")

    np.savez(os.path.join(cfg.RESULTS_DIR, 'raw_test_outputs.npz'),
             errors_cm=errors, pred_flat=pred_flat, target_flat=target_flat)


if __name__ == "__main__":
    print("\n" + "="*60 + "\nENHANCED InSAR U-Net (Standardized Protocol)\n" + "="*60)
    datasets, stats = prepare_datasets()
    if 'train' in datasets and len(datasets['train']) > 0:
        history = train_model(datasets, stats)
        evaluate(datasets, stats)
        print("\n" + "="*60 + "\nPIPELINE COMPLETED!\n" + "="*60)
        print(f"\nResults: {cfg.RESULTS_DIR}\nModel: {cfg.MODEL_PATH}")
    else:
        print("\n⚠ ERROR: No training data!")
        sys.exit(1)
