# Experiment 1 Report: Architecture Fix (Sequence-based InceptionV3 + LSTM)

## Summary

| Metric | Baseline (seq=1) | **Exp 1 (seq=10)** | Target | Status |
|--------|:-----------------:|:-------------------:|:------:|:------:|
| **Accuracy** | 68.0% | **80.5%** | >75% | ✅ PASSED |
| **Sensitivity (MDD)** | 75.6% | **90.1%** | >85% | ✅ PASSED |
| **Specificity (H)** | 56.4% | **63.6%** | >75% | ⚠️ GAP |
| **F1 Score** | 72.0% | **85.5%** | — | — |

## Global Confusion Matrix
```
                Pred Healthy    Pred MDD
True Healthy       1,375          787
True MDD             377        3,431
```
- Total test sequences: 5,970

## Key Architectural Changes
1. **Sequence length**: 1 → **10** consecutive spectrograms (7 seconds of brain activity)
2. **pos_weight**: Fixed from `n_h/n_mdd` (0.56) to `n_mdd/n_h` (~1.7)
3. **LSTM hidden**: 256 → **128** (matching paper)
4. **Threshold optimization**: Per-fold validation sweep (0.1–0.9)
5. **Random seed**: Set to 42 for reproducibility

## Per-Subject Analysis

### Healthy Subjects (14 total)
| Tier | Subjects | Count | Mean Specificity |
|------|----------|-------|-----------------|
| Tier 1 (>70%) | H_19, H_22, H_23, H_28, H_4, H_5, H_6, H_8, H_9 | 9 | **84.4%** |
| Tier 2 (10-70%) | H_14, H_26 | 2 | 67.9% |
| **Tier 3 (<10%)** | **H_16, H_24, H_27** | **3** | **0.7%** |
| **Without outliers** | — | **11** | **81.4%** |

### MDD Subjects (26 total)
| Tier | Subjects | Count | Mean Sensitivity |
|------|----------|-------|-----------------|
| Tier 1 (>90%) | 24 subjects | 24 | **97.5%** |
| Tier 2 (20-90%) | — | 0 | — |
| **Tier 3 (<20%)** | **MDD_19, MDD_5** | **2** | **11.4%** |
| **Without outliers** | — | **24** | **97.5%** |

## Outlier Root Cause
Spectral analysis confirmed:
- **Outlier Healthy** subjects have reduced Alpha energy (~0.37) matching MDD profiles
- **Outlier MDD** subjects have higher Alpha energy (~0.38) matching Healthy profiles
- These individuals sit on the **wrong side of the feature boundary**
- The single-channel averaged preprocessing cannot distinguish them

## Conclusion
Experiment 1 successfully validated that the architecture fix (seq_len=10) was the #1 priority.
The remaining gap (Specificity 63.6% vs 75% target) is attributable to 5 outlier subjects
whose averaged spectral profiles overlap with the opposite class.

## Figures
See `results/exp1_seq_lstm/figures/` for:
1. Per-subject performance bar chart
2. Global confusion matrix
3. Class-wise box plots
4. Training loss curves
5. Threshold distribution
6. Baseline vs Experiment 1 comparison

## Training Details
- Device: Apple M-Series GPU (MPS)
- Total training time: **48 hours** (2,880 minutes)
- Average fold time: ~72 minutes
- Epochs per fold: 16–50 (early stopping with patience=10)
