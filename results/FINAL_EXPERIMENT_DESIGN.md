# Final Experiment Design: Combined Exp 2+3 (Spatial RGB + Fine-Tuning)

## Rationale for Combining

Running Exp 2 and Exp 3 separately would cost ~100 hours of training. By combining them:
- **One preprocessing run** (~30 min) for spatial RGB images
- **One training run** (~48 hrs) with fine-tuned InceptionV3
- **Total savings**: ~48 hours

More importantly, fine-tuning InceptionV3 on *averaged* spectrograms (Exp 2 alone) would learn 
better spectrogram features, but would still be spatially blind. The synergy of combining them 
means InceptionV3 can learn **cross-regional spatial patterns** like Frontal Alpha Asymmetry — 
which is exactly what we need to fix the 5 outlier subjects.

---

## Step 1: Spatial RGB Preprocessing (`preprocess_spatial.py`)

### Channel Grouping (based on actual EDF channel names)

The 20 channels from the 10-20 system map naturally into 3 brain regions:

| RGB Channel | Brain Region | EDF Channels | Count |
|:-----------:|:------------:|:-------------|:-----:|
| **Red** | Frontal | Fp1, F3, F7, Fz, Fp2, F4, F8 | 7 |
| **Green** | Central/Temporal | C3, T3, C4, T4, Cz | 5 |
| **Blue** | Parietal/Occipital | P3, O1, T5, P4, O2, T6, Pz | 7 |

> Note: A2-A1 (reference channel) is excluded.
> Total: 19 channels across 3 groups.

### Processing Pipeline (per epoch)
```
For each group (R, G, B):
    1. Average channels within the group → 1D signal
    2. DWT (db4, 5 levels) → extract cA
    3. CWT (Morlet, 64 scales) → 2D scalogram
    4. Resize to 224×224 → one channel of the RGB image

Stack R, G, B → (224, 224, 3) TRUE RGB image
```

### Why This Fixes the Outliers
Currently: All 20 channels → average → 1 grayscale → copy to RGB → model sees NO spatial info.

With Spatial RGB:
- **Frontal Alpha Asymmetry** (LEFT vs RIGHT frontal power) → visible as R channel intensity patterns
- **Frontal-Posterior Gradient** → visible as R vs B contrast
- **Temporal Lobe Activity** → isolated in G channel

A healthy person with "generally low alpha" will show uniform reduction across R, G, B.
A depressed person typically shows **asymmetric frontal reduction** (left > right frontal alpha loss)
→ The model can now distinguish them via spatial patterns in the RGB image.

---

## Step 2: Training Configuration (`train_exp_final.py`)

### Fine-Tuning Strategy
```python
# FROZEN: InceptionV3 layers up to Mixed_6e (early feature extractors)
# UNFROZEN: Mixed_7a, Mixed_7b, Mixed_7c (high-level feature combining)

# Differential learning rates:
optimizer = torch.optim.Adam([
    {'params': inception_frozen_params, 'lr': 0},          # frozen
    {'params': inception_unfrozen_params, 'lr': 1e-5},     # slow adaptation
    {'params': lstm_params, 'lr': 1e-4},                   # normal
    {'params': classifier_params, 'lr': 1e-4},             # normal
])
```

### All Other Settings (Same as Exp 1)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sequence length | 10 | Matches paper, proven in Exp 1 |
| LSTM hidden | 128 | Matches paper |
| pos_weight | n_mdd/n_h (~1.7) | Corrected, proven in Exp 1 |
| Batch size | 8 | Reduced from 16 (more params unfrozen = more memory) |
| Max epochs | 50 | Same as Exp 1 |
| Early stopping | patience=10 | Same as Exp 1 |
| Threshold | Per-fold optimization | Same as Exp 1 |
| Seed | 42 | Reproducibility |

### Additional Improvements
1. **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
2. **Gradient Clipping**: max_norm=1.0 (prevents exploding gradients during fine-tuning)
3. **Subject-Level Majority Voting**: After sequence-level predictions, aggregate to subject-level via majority vote

---

## Expected Impact

| Metric | Exp 1 | **Final Exp (Exp 2+3)** | Target |
|--------|:-----:|:-----------------------:|:------:|
| Accuracy | 80.5% | **≥82-85%** | >75% |
| Sensitivity | 90.1% | **≥90%** (maintain) | >85% |
| Specificity | 63.6% | **≥72-78%** | >75% |

### Where the gains come from:
- **Spatial RGB** → Fix 2-3 of 5 outliers → +8-12% Specificity
- **Fine-tuning** → Better learned features → +2-5% across all metrics
- **LR Scheduler** → More stable convergence → reduces noise in results
- **Majority Voting** → +1-3% subject-level accuracy

---

## Execution Plan

```
Phase 1: Preprocessing (~30 min)
    python scripts/preprocess_spatial.py
    → Generates new data/spectrograms_spatial/{spectrograms,labels,subject_ids}.npy

Phase 2: Training (~48 hrs)
    python scripts/train_exp_final.py
    → Results saved to results/exp_final/

Phase 3: Analysis (~5 min)
    python scripts/visualize_exp_final.py
    → Comparison charts, confusion matrix, per-subject breakdown
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Fine-tuning overfits | Medium | Differential LR + early stopping + grad clipping |
| Spatial RGB doesn't help outliers | Low | Even partial improvement is valuable; majority voting as backup |
| Training time increases | Low | Batch size reduced to 8; epoch time similar |
| Memory issues (more unfrozen params) | Low | MPS handles ~23M params fine; batch=8 keeps memory manageable |
