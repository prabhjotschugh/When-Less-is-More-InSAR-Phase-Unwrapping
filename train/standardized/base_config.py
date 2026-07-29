# ==========================================
# SHARED BASE CONFIGURATION
# When Less Is More: Simplicity Beats Complexity for Physics-Constrained InSAR Phase Unwrapping
# IEEE GRSL Revision
#
# CRITICAL: This file is the SINGLE SOURCE OF TRUTH for every hyperparameter
# that must be identical across Vanilla / Enhanced / Attention / Hybrid.
# Reviewer 1 (initial + revision) flagged that the four models were NOT
# trained under a standardized protocol:
#   - Vanilla/Enhanced used FP32, Attention/Hybrid used FP16 (AMP)
#   - Enhanced had dropout=0.10, Vanilla had dropout=0.0
#   - Weight decay differed between Attention/Hybrid and Vanilla/Enhanced
#
# Fix: every shared hyperparameter lives here ONCE. Each model-specific
# script imports BaseConfig and only overrides path/name fields. No script
# is allowed to silently redefine LR, dropout, weight decay, precision,
# batch size, epochs, patience, or loss function. If you need to change
# any of these for the ablation, change it here so all four scripts move
# together and the "standardized protocol" claim in the paper stays true.
# ==========================================

import os


class BaseConfig:
    # ---- Data ----
    BASE_DIR = "insar_icml_project"
    DATA_DIR = os.path.join(BASE_DIR, "raw_frames")
    SPLIT_FILE = os.path.join(BASE_DIR, "dataset_splits_v2.json")

    PATCH_SIZE = 128
    STRIDE = 64
    MIN_COHERENCE = 0.5
    MIN_LOS_MAGNITUDE = 0.01
    WAVELENGTH = 0.056

    # ---- Training protocol (IDENTICAL FOR ALL 4 MODELS) ----
    BATCH_SIZE = 32
    EPOCHS = 1000
    LR = 8e-5
    WEIGHT_DECAY = 1e-4          # unified (was 5e-5 vs 1e-4 split before)
    GRAD_CLIP = 1.0              # unified (was 1.0 vs 0.5 split before)

    # ---- Precision (IDENTICAL: full FP32, no AMP, for any model) ----
    USE_AMP = False               # unified: AMP removed entirely (was on for Attn/Hybrid only)

    # ---- Regularization (IDENTICAL dropout base value for all 4) ----
    DROPOUT = 0.15                # unified (was 0.0 / 0.10 / 0.20 / 0.20 split before)
    LABEL_SMOOTHING = 0.0

    # ---- Early stopping (IDENTICAL) ----
    EARLY_STOP_PATIENCE = 100
    MIN_DELTA = 1e-5

    # ---- Checkpointing / resuming ----
    RESUME_TRAINING = True
    SAVE_CHECKPOINT_FREQ = 10

    # ---- Visualization ----
    VIZ_FREQUENCY = 20
    N_VIZ_SAMPLES = 5

    # ---- Model I/O shape (IDENTICAL) ----
    IN_CHANNELS = 6
    OUT_CHANNELS = 1
    BASE_CHANNELS = 32

    # ---- Loss function (IDENTICAL: Huber + Sobel-gradient physics loss for all 4) ----
    HUBER_DELTA = 1.0
    LAMBDA_GRAD = 0.1

    # ---- Reproducibility ----
    SEED = 42


def get_base_kwargs():
    """Return a dict of every field above, used for sanity-printing at
    the start of each training run so it's visible in logs / rebuttal
    appendix that the protocol truly matches across scripts."""
    return {k: v for k, v in vars(BaseConfig).items() if not k.startswith('_')}


def print_protocol_banner(model_name, cfg):
    """Print a banner confirming which fields are shared vs model-specific.
    Paste this output into the rebuttal letter as evidence of standardization."""
    shared_fields = [
        'BATCH_SIZE', 'EPOCHS', 'LR', 'WEIGHT_DECAY', 'GRAD_CLIP', 'USE_AMP',
        'DROPOUT', 'EARLY_STOP_PATIENCE', 'MIN_DELTA', 'IN_CHANNELS',
        'OUT_CHANNELS', 'BASE_CHANNELS', 'HUBER_DELTA', 'LAMBDA_GRAD', 'SEED'
    ]
    print(f"\n{'='*70}")
    print(f"STANDARDIZED PROTOCOL CHECK — {model_name}")
    print(f"{'='*70}")
    for field in shared_fields:
        print(f"  {field:<20} = {getattr(cfg, field)}")
    print(f"{'='*70}\n")
