"""
train_exp_final.py
──────────────────
Final Experiment: Spatial RGB + Fine-Tuned InceptionV3 + LSTM.

COMBINED from Experiment 2 (Fine-Tuning) + Experiment 3 (Spatial RGB):

KEY CHANGES vs train_exp1.py (Experiment 1):
  1. Uses SPATIAL RGB spectrograms (Frontal=R, Central=G, Posterior=B)
     → Model can now see WHERE brain activity is happening
  2. PARTIAL fine-tuning of InceptionV3 (last 3 Inception blocks unfrozen)
     → Learns spectrogram-specific features instead of ImageNet shapes
  3. Differential learning rates (InceptionV3: 1e-5, LSTM/classifier: 1e-4)
     → Slow adaptation of pretrained layers, fast learning of new layers
  4. LR Scheduler (ReduceLROnPlateau) for stable convergence
  5. Gradient clipping to prevent exploding gradients during fine-tuning
  6. Subject-level majority voting for final evaluation

KEPT FROM Experiment 1:
  - Sequence-based input (seq_len=10)
  - Corrected pos_weight (n_mdd/n_h)
  - LSTM hidden=128, 1 layer
  - Per-fold threshold optimization
  - Random seed for reproducibility

HOW TO RUN:
    conda activate eeg_train
    python scripts/train_exp_final.py
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import csv
from prettytable import PrettyTable

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms_spatial")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "exp_final")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Model hyperparameters
SEQ_LEN            = 10       # consecutive spectrograms per sequence (paper: 10)
INCEPTION_FEATURES = 2048     # InceptionV3 output features
LSTM_HIDDEN        = 128      # paper uses 128
LSTM_LAYERS        = 1
DROPOUT            = 0.3

# Training hyperparameters
LR_INCEPTION     = 1e-5     # slow learning for fine-tuned InceptionV3 layers
LR_NEW           = 1e-4     # fast learning for LSTM + classifier
BATCH_SIZE       = 8        # reduced from 16 (more params = more memory)
MAX_EPOCHS       = 50
PATIENCE         = 10       # early stopping patience
MAX_SEQS_PER_SUBJECT = 200  # cap sequences per subject
GRAD_CLIP        = 1.0      # gradient clipping max norm
LR_SCHED_FACTOR  = 0.5      # LR scheduler reduction factor
LR_SCHED_PATIENCE = 5       # LR scheduler patience

# Log file
LOG_FILE  = os.path.join(RESULTS_DIR, "training_log.txt")
CSV_FILE  = os.path.join(RESULTS_DIR, "fold_results.csv")

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════
def log_msg(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def log_separator():
    log_msg("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET: Sequence-based (same as Exp 1)
# ═══════════════════════════════════════════════════════════════════════════
class EEGSequenceDataset(Dataset):
    """
    Each sample is a sequence of SEQ_LEN consecutive spectrograms.
    Now uses SPATIAL RGB images from preprocess_spatial.py.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences  # (N, SEQ_LEN, 224, 224, 3)
        self.labels    = labels     # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]   # (SEQ_LEN, 224, 224, 3)
        lbl = self.labels[idx]

        # Convert to tensor: (SEQ_LEN, 3, H, W) for PyTorch
        seq_tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).float()
        lbl_tensor = torch.tensor(lbl, dtype=torch.float32)
        return seq_tensor, lbl_tensor


def build_sequences(indices, spectrograms, labels, subject_ids,
                    seq_len=SEQ_LEN, max_seqs=None):
    """Build sequences of consecutive spectrograms. Same as Exp 1."""
    sequences = []
    seq_labels = []

    sids = subject_ids[indices]
    for sid in np.unique(sids):
        sid_mask = np.where(sids == sid)[0]
        sid_global = indices[sid_mask]
        label = labels[sid_global[0]]

        n_seqs = len(sid_global) // seq_len
        if max_seqs is not None and n_seqs > max_seqs:
            starts = np.random.choice(
                np.arange(len(sid_global) - seq_len + 1), max_seqs, replace=False
            )
            starts = sorted(starts)
        else:
            starts = [i * seq_len for i in range(n_seqs)]

        for start in starts:
            seq_idx = sid_global[start:start + seq_len]
            if len(seq_idx) < seq_len:
                continue
            seq = spectrograms[seq_idx]
            sequences.append(seq)
            seq_labels.append(label)

    if len(sequences) == 0:
        return np.array([]).reshape(0, seq_len, 224, 224, 3), np.array([])

    return np.array(sequences, dtype=np.float32), np.array(seq_labels, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL: Fine-Tuned InceptionV3 + LSTM
# ═══════════════════════════════════════════════════════════════════════════
class InceptionV3_LSTM_FineTuned(nn.Module):
    """
    Hybrid model: Partially Fine-Tuned InceptionV3 + LSTM.

    Changes from Exp 1:
      - InceptionV3 layers Mixed_7a through Mixed_7c are UNFROZEN
        → Can learn spectrogram-specific features
      - All earlier layers remain FROZEN
        → Preserve general visual features, prevent catastrophic forgetting

    Input:  (batch, seq_len, 3, 224, 224)
    Output: (batch,) logits
    """
    def __init__(self):
        super().__init__()

        # Load pretrained InceptionV3
        self.inception = models.inception_v3(weights='IMAGENET1K_V1')
        self.inception.aux_logits = False
        self.inception.AuxLogits = None
        self.inception.fc = nn.Identity()

        # FREEZE everything first
        for param in self.inception.parameters():
            param.requires_grad = False

        # UNFREEZE only Mixed_7c (last Inception block)
        # Fewer trainable params → less overfitting risk
        unfreeze_layers = ['Mixed_7c']
        for name, module in self.inception.named_children():
            if name in unfreeze_layers:
                for param in module.parameters():
                    param.requires_grad = True

        # Upsample 224→299 for InceptionV3
        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

        # LSTM: sees sequence of InceptionV3 features
        self.lstm = nn.LSTM(
            input_size=INCEPTION_FEATURES,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=0.0
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN, 1)
        )

    def get_param_groups(self):
        """Return parameter groups for differential learning rates."""
        # Inception unfrozen parameters (Mixed_7c only)
        inception_unfrozen = []
        inception_unfrozen_names = ['Mixed_7c']
        for name, module in self.inception.named_children():
            if name in inception_unfrozen_names:
                inception_unfrozen.extend(
                    p for p in module.parameters() if p.requires_grad
                )

        # LSTM + classifier parameters
        new_params = list(self.lstm.parameters()) + list(self.classifier.parameters())

        return [
            {'params': inception_unfrozen, 'lr': LR_INCEPTION, 'label': 'inception_unfrozen'},
            {'params': new_params,         'lr': LR_NEW,       'label': 'lstm_classifier'},
        ]

    def forward(self, x):
        """
        x: (batch, seq_len, 3, 224, 224)
        """
        batch_size, seq_len, C, H, W = x.shape

        # Process each timestep through InceptionV3
        x_flat = x.view(batch_size * seq_len, C, H, W)
        x_flat = self.upsample(x_flat)

        features_flat = self.inception(x_flat)
        features = features_flat.view(batch_size, seq_len, INCEPTION_FEATURES)

        # LSTM over the sequence
        _, (h_n, _) = self.lstm(features)

        # Classify from final hidden state
        out = self.classifier(h_n.squeeze(0))
        return out.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids,
                   fold_num, subject_name, total_folds):
    """Train and evaluate model for one LOSO fold."""

    log_separator()
    log_msg(f"FOLD {fold_num}/{total_folds} — Test Subject: {subject_name}")
    log_separator()

    t_fold_start = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # 1. Build sequences
    # ─────────────────────────────────────────────────────────────────────
    t_prep = time.time()

    X_tr_seq, y_tr = build_sequences(
        train_indices, spectrograms, labels, subject_ids,
        seq_len=SEQ_LEN, max_seqs=MAX_SEQS_PER_SUBJECT
    )
    X_te_seq, y_te = build_sequences(
        test_indices, spectrograms, labels, subject_ids,
        seq_len=SEQ_LEN, max_seqs=None
    )

    if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
        log_msg(f"  SKIP: insufficient sequences (train={len(X_tr_seq)}, test={len(X_te_seq)})")
        return None

    # Stratified validation split (15%)
    h_idx   = np.where(y_tr == 0)[0]
    mdd_idx = np.where(y_tr == 1)[0]
    n_val_h   = max(1, int(0.15 * len(h_idx)))
    n_val_mdd = max(1, int(0.15 * len(mdd_idx)))

    val_h   = np.random.choice(h_idx,   n_val_h,   replace=False)
    val_mdd = np.random.choice(mdd_idx, n_val_mdd, replace=False)
    val_rel = np.concatenate([val_h, val_mdd])
    tr_rel  = np.setdiff1d(np.arange(len(y_tr)), val_rel)

    X_val, y_val = X_tr_seq[val_rel], y_tr[val_rel]
    X_tr,  y_tr  = X_tr_seq[tr_rel],  y_tr[tr_rel]

    # Class balance
    n_h   = int((y_tr == 0).sum())
    n_mdd = int((y_tr == 1).sum())
    # pos_weight boosts the POSITIVE class (label=1 = MDD)
    # Since MDD is majority, we want pos_weight < 1 to REDUCE MDD bias
    # This makes the model pay MORE attention to getting Healthy right
    pos_weight = n_h / max(n_mdd, 1)

    log_msg(f"  Sequences — Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te_seq)}")
    log_msg(f"  Class balance (train): H={n_h}, MDD={n_mdd} | pos_weight={pos_weight:.3f}")

    # Datasets and loaders
    train_ds = EEGSequenceDataset(X_tr,     y_tr)
    val_ds   = EEGSequenceDataset(X_val,    y_val)
    test_ds  = EEGSequenceDataset(X_te_seq, y_te)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log_msg(f"  Data prep: {time.time()-t_prep:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Initialize model with fine-tuning
    # ─────────────────────────────────────────────────────────────────────
    model = InceptionV3_LSTM_FineTuned().to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_msg(f"  Model: FT-InceptionV3+LSTM(seq={SEQ_LEN}) | "
            f"Trainable: {trainable_params:,}/{total_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )

    # Differential learning rates
    param_groups = model.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHED_FACTOR,
        patience=LR_SCHED_PATIENCE
    )

    log_msg(f"  Optimizer: Adam | LR_inception={LR_INCEPTION}, LR_new={LR_NEW}")
    log_msg(f"  Scheduler: ReduceLROnPlateau(factor={LR_SCHED_FACTOR}, patience={LR_SCHED_PATIENCE})")
    log_msg(f"  Gradient clipping: max_norm={GRAD_CLIP}")

    # ─────────────────────────────────────────────────────────────────────
    # 3. Training loop
    # ─────────────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None
    train_losses  = []
    val_losses    = []

    log_msg("  Starting training...")

    for epoch in range(MAX_EPOCHS):
        t_epoch = time.time()

        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0

        for batch_seqs, batch_labels in train_loader:
            batch_seqs   = batch_seqs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_seqs)
            loss    = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping (prevents exploding gradients during fine-tuning)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        epoch_train_loss /= max(n_batches, 1)
        train_losses.append(epoch_train_loss)

        # --- Validate ---
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches  = 0

        with torch.no_grad():
            for batch_seqs, batch_labels in val_loader:
                batch_seqs   = batch_seqs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                outputs      = model(batch_seqs)
                loss         = criterion(outputs, batch_labels)
                epoch_val_loss += loss.item()
                n_val_batches  += 1

        epoch_val_loss /= max(n_val_batches, 1)
        val_losses.append(epoch_val_loss)

        # LR scheduler step
        scheduler.step(epoch_val_loss)

        # Get current LR for logging
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        lr_str = f"lr=[{current_lrs[0]:.1e},{current_lrs[1]:.1e}]"

        log_msg(f"  Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
                f"train={epoch_train_loss:.4f} | val={epoch_val_loss:.4f} | "
                f"{lr_str} | {time.time()-t_epoch:.1f}s")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log_msg(f"  Early stopping at epoch {epoch+1}")
                break

    # ─────────────────────────────────────────────────────────────────────
    # 4. Threshold optimization on validation set
    # ─────────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    val_probs_list = []
    val_true_list  = []
    with torch.no_grad():
        for batch_seqs, batch_labels in val_loader:
            batch_seqs = batch_seqs.to(DEVICE)
            outputs    = model(batch_seqs)
            probs      = torch.sigmoid(outputs).cpu().numpy()
            val_probs_list.extend(probs)
            val_true_list.extend(batch_labels.numpy())

    val_probs = np.array(val_probs_list)
    val_true  = np.array(val_true_list).astype(int)

    best_thresh = 0.5
    best_f1_val = 0.0
    for thresh in np.arange(0.1, 0.91, 0.05):
        val_preds = (val_probs >= thresh).astype(int)
        f1_val = f1_score(val_true, val_preds, zero_division=0)
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_thresh = thresh

    log_msg(f"  Optimal threshold: {best_thresh:.2f} (val F1={best_f1_val:.3f})")

    # ─────────────────────────────────────────────────────────────────────
    # 5. Evaluation on test set
    # ─────────────────────────────────────────────────────────────────────
    all_probs = []
    all_true  = []

    with torch.no_grad():
        for batch_seqs, batch_labels in test_loader:
            batch_seqs = batch_seqs.to(DEVICE)
            outputs    = model(batch_seqs)
            probs      = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(batch_labels.numpy())

    y_true = np.array(all_true).astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= best_thresh).astype(int)

    # Sequence-level metrics
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)

    h_mask   = (y_true == 0)
    mdd_mask = (y_true == 1)
    h_recall   = accuracy_score(y_true[h_mask],   y_pred[h_mask])   if h_mask.any()   else 0.0
    mdd_recall = accuracy_score(y_true[mdd_mask], y_pred[mdd_mask]) if mdd_mask.any() else 0.0

    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_score = float('nan')

    # Subject-level majority vote
    if len(y_pred) > 0:
        vote_counts = np.bincount(y_pred.astype(int), minlength=2)
        majority_vote = 1 if vote_counts[1] >= vote_counts[0] else 0
        actual_label = y_true[0]  # all same in LOSO single-class test
        vote_correct = int(majority_vote == actual_label)
    else:
        majority_vote = -1
        vote_correct = 0

    log_msg(f"  RESULTS: acc={acc:.3f} | f1={f1:.3f} | auc={auc_score:.3f} | thresh={best_thresh:.2f}")
    log_msg(f"           Sensitivity(MDD)={mdd_recall:.3f} | Specificity(H)={h_recall:.3f}")
    log_msg(f"           Majority Vote: pred={majority_vote}, correct={vote_correct}")
    log_msg(f"  Fold time: {(time.time()-t_fold_start)/60:.1f} min")

    return {
        'subject':       subject_name,
        'accuracy':      acc,
        'f1':            f1,
        'precision':     precision,
        'recall':        recall,
        'auc':           auc_score,
        'sensitivity':   mdd_recall,
        'specificity':   h_recall,
        'threshold':     best_thresh,
        'majority_vote': majority_vote,
        'vote_correct':  vote_correct,
        'y_true':        y_true,
        'y_pred':        y_pred,
        'y_prob':        y_prob,
        'train_loss':    train_losses,
        'val_loss':      val_losses,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    log_separator()
    log_msg("FINAL EXPERIMENT: Spatial RGB + Fine-Tuned InceptionV3 + LSTM")
    log_msg(f"  Device:         {DEVICE}")
    log_msg(f"  Data:           Spatial RGB (Frontal=R, Central=G, Posterior=B)")
    log_msg(f"  Seq length:     {SEQ_LEN} spectrograms per sample")
    log_msg(f"  LSTM:           hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS}")
    log_msg(f"  Fine-tuning:    Mixed_7c only (1 block, reduced overfitting)")
    log_msg(f"  LR inception:   {LR_INCEPTION}")
    log_msg(f"  LR new:         {LR_NEW}")
    log_msg(f"  Batch size:     {BATCH_SIZE}")
    log_msg(f"  Grad clipping:  {GRAD_CLIP}")
    log_msg(f"  pos_weight:     n_mdd/n_h (corrected)")
    log_msg(f"  Threshold:      per-fold optimization")
    log_separator()

    # Load spatial RGB data (mmap_mode='r' to prevent swap thrashing)
    log_msg("Loading spatial RGB data...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"), mmap_mode='r')
    labels       = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids  = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    log_msg(f"  Loaded (mmap): {spectrograms.shape}, labels={labels.shape}")

    # Verify these are true RGB (not replicated grayscale)
    sample = spectrograms[0]
    is_spatial = not np.allclose(sample[:,:,0], sample[:,:,1])
    log_msg(f"  Spatial RGB check: {'PASS ✓' if is_spatial else 'FAIL ✗ (grayscale detected!)'}")
    if not is_spatial:
        log_msg("  ERROR: Data appears to be replicated grayscale, not spatial RGB!")
        log_msg("  Run preprocess_spatial.py first to generate spatial RGB data.")
        sys.exit(1)

    # Get unique subjects
    unique_sids = sorted(np.unique(subject_ids))
    total_folds = len(unique_sids)
    log_msg(f"  Subjects: {total_folds} | LOSO folds: {total_folds}")

    # Build subject name map
    cleaned_dir = os.path.join(PROJECT_ROOT, "data", "cleaned")
    sid_to_name = {}
    for i, folder in enumerate(sorted(os.listdir(cleaned_dir))):
        if os.path.isdir(os.path.join(cleaned_dir, folder)):
            sid_to_name[i] = folder

    # LOSO loop
    fold_results = []
    all_y_true   = []
    all_y_pred   = []
    vote_results = []

    # Resume logic: Check completed subjects in CSV_FILE
    completed_subjects = set()
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed_subjects.add(row['subject'])
            if len(completed_subjects) > 0:
                log_msg(f"  Resuming: Found {len(completed_subjects)} subjects already completed.")
        except Exception as e:
            log_msg(f"  Resume Warning: Could not read {CSV_FILE}: {e}")

    for fold_num, test_sid in enumerate(unique_sids, start=1):
        subject_name = sid_to_name.get(test_sid, f"Subject_{test_sid}")

        if subject_name in completed_subjects:
            log_msg(f"FOLD {fold_num}/{total_folds} — Skipping {subject_name} (already in CSV)")
            continue

        test_indices  = np.where(subject_ids == test_sid)[0]
        train_indices = np.where(subject_ids != test_sid)[0]

        result = train_one_fold(
            train_indices, test_indices,
            spectrograms, labels, subject_ids,
            fold_num, subject_name, total_folds
        )

        if result is None:
            continue

        fold_results.append(result)
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['y_pred'])
        vote_results.append({
            'subject': subject_name,
            'vote': result['majority_vote'],
            'correct': result['vote_correct'],
            'true_label': result['y_true'][0] if len(result['y_true']) > 0 else -1,
        })

        # Save to CSV (mode='a' if file exists, else 'w')
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, "a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                'subject', 'accuracy', 'f1', 'precision', 'recall',
                'auc', 'sensitivity', 'specificity', 'threshold',
                'majority_vote', 'vote_correct'
            ])
            if not file_exists:
                writer.writeheader()
            
            # Note: We only append the LAST result here to avoid re-writing everything
            # or if we want to overwrite everything we can keep 'w' and write from fold_results
            # But appending is safer for resumption.
            writer.writerow({k: result[k] for k in writer.fieldnames})

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    log_separator()
    log_msg("FINAL RESULTS")
    log_separator()

    # Global confusion matrix
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred)
    log_msg(f"  Global Confusion Matrix:\n{cm}")

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    global_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    global_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    global_accuracy    = (tp + tn) / (tp + tn + fp + fn)
    global_f1          = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # Subject-level majority vote accuracy
    vote_correct_count = sum(1 for v in vote_results if v['correct'] == 1)
    vote_total = len(vote_results)
    vote_accuracy = vote_correct_count / vote_total if vote_total > 0 else 0

    # Per-fold table
    table = PrettyTable()
    table.field_names = ["Subject", "Acc", "F1", "Sensitivity", "Specificity",
                         "AUC", "Thresh", "Vote"]
    for r in fold_results:
        table.add_row([
            r['subject'],
            f"{r['accuracy']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['sensitivity']:.3f}",
            f"{r['specificity']:.3f}",
            f"{r['auc']:.3f}" if not np.isnan(r['auc']) else "nan",
            f"{r['threshold']:.2f}",
            "✓" if r['vote_correct'] else "✗",
        ])
    log_msg(f"\n{table}")

    # Summary stats
    accs  = [r['accuracy']    for r in fold_results]
    f1s   = [r['f1']          for r in fold_results]
    senss = [r['sensitivity'] for r in fold_results]
    specs = [r['specificity'] for r in fold_results]

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Mean", "Std", "Min", "Max"]
    for name, vals in [("Accuracy", accs), ("F1", f1s),
                        ("Sensitivity", senss), ("Specificity", specs)]:
        summary_table.add_row([
            name,
            f"{np.mean(vals):.3f}",
            f"{np.std(vals):.3f}",
            f"{np.min(vals):.3f}",
            f"{np.max(vals):.3f}",
        ])
    log_msg(f"\n{summary_table}")

    log_msg(f"\n  GLOBAL (aggregated across all folds):")
    log_msg(f"    Accuracy:       {global_accuracy:.3f}")
    log_msg(f"    Sensitivity:    {global_sensitivity:.3f}")
    log_msg(f"    Specificity:    {global_specificity:.3f}")
    log_msg(f"    F1 Score:       {global_f1:.3f}")
    log_msg(f"\n  SUBJECT-LEVEL MAJORITY VOTE:")
    log_msg(f"    Correct: {vote_correct_count}/{vote_total} ({vote_accuracy:.1%})")

    # Comparison with Exp 1
    log_msg(f"\n  COMPARISON WITH EXPERIMENT 1:")
    log_msg(f"    {'Metric':<20} {'Exp 1':<10} {'Final':<10} {'Change':<10}")
    log_msg(f"    {'─' * 50}")
    exp1 = {'Accuracy': 0.805, 'Sensitivity': 0.901, 'Specificity': 0.636, 'F1': 0.855}
    final = {'Accuracy': global_accuracy, 'Sensitivity': global_sensitivity,
             'Specificity': global_specificity, 'F1': global_f1}
    for metric in exp1:
        diff = final[metric] - exp1[metric]
        sign = '+' if diff >= 0 else ''
        log_msg(f"    {metric:<20} {exp1[metric]:.3f}     {final[metric]:.3f}     {sign}{diff:.3f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Final Experiment: Spatial RGB + Fine-Tuned InceptionV3 + LSTM\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total subjects: {total_folds}\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"LSTM hidden: {LSTM_HIDDEN}\n")
        f.write(f"Fine-tuned layers: Mixed_7a, Mixed_7b, Mixed_7c\n")
        f.write(f"LR inception: {LR_INCEPTION}, LR new: {LR_NEW}\n\n")
        f.write(str(summary_table) + "\n\n")
        f.write(f"Global Accuracy:    {global_accuracy:.4f}\n")
        f.write(f"Global Sensitivity: {global_sensitivity:.4f}\n")
        f.write(f"Global Specificity: {global_specificity:.4f}\n")
        f.write(f"Global F1:          {global_f1:.4f}\n\n")
        f.write(f"Subject-Level Vote: {vote_correct_count}/{vote_total} ({vote_accuracy:.1%})\n")
        f.write(f"\nTotal time: {(time.time()-t_start)/60:.1f} minutes\n")

    log_msg(f"\n  Results saved to: {RESULTS_DIR}")
    log_msg(f"  Total time: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
