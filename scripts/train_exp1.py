"""
train_exp1.py
─────────────
Experiment 1: Fix the architecture to match the paper.

KEY CHANGES vs train_inceptionv3_loso.py:
  1. Sequence-based input: 10 consecutive spectrograms per sample (seq_len=10)
     → InceptionV3 extracts features from each, LSTM sees the temporal sequence
  2. Fixed pos_weight: n_mdd / n_h (>1) to penalize missed MDD cases more
  3. LSTM hidden_size=128 (matches paper), 1 layer
  4. Random seed set for reproducibility
  5. Comprehensive metrics: Sensitivity, Specificity, F1, AUC per fold
  6. Threshold optimization on validation set before test evaluation

HOW TO RUN:
    conda activate eeg_train
    python scripts/train_exp1.py
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "exp1_seq_lstm")
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

# Model hyperparameters (matching paper)
SEQ_LEN          = 10       # consecutive spectrograms per sequence (paper: 10 timesteps)
INCEPTION_FEATURES = 2048   # InceptionV3 output features
LSTM_HIDDEN      = 128      # paper uses 128
LSTM_LAYERS      = 1
DROPOUT          = 0.3

# Training hyperparameters
LEARNING_RATE    = 1e-4     # paper: 0.0001
BATCH_SIZE       = 16       # paper: 16
MAX_EPOCHS       = 50       # paper: 50
PATIENCE         = 10       # early stopping patience
MAX_SEQS_PER_SUBJECT = 200  # cap sequences per subject to limit training time

# Log file
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.txt")

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
# DATASET: Sequence-based
# ═══════════════════════════════════════════════════════════════════════════
class EEGSequenceDataset(Dataset):
    """
    Each sample is a sequence of SEQ_LEN consecutive spectrograms from the
    same subject. Label is the subject's class (0=Healthy, 1=MDD).

    This matches the paper's input_shape = (10, 224, 224, 3).
    """
    def __init__(self, sequences, labels):
        """
        sequences: np.array of shape (N, SEQ_LEN, 224, 224, 3)
        labels:    np.array of shape (N,)
        """
        self.sequences = sequences
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]   # (SEQ_LEN, 224, 224, 3)
        lbl = self.labels[idx]

        # Convert to tensor: (SEQ_LEN, 3, H, W) for PyTorch
        seq_tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).float()  # (SEQ_LEN, 3, 224, 224)
        lbl_tensor = torch.tensor(lbl, dtype=torch.float32)
        return seq_tensor, lbl_tensor


def build_sequences(indices, spectrograms, labels, subject_ids, seq_len=SEQ_LEN, max_seqs=None):
    """
    Build sequences of consecutive spectrograms from the given indices.
    Since spectrograms are stored contiguously per subject, we can safely
    take consecutive windows within each subject's block.

    Returns:
        sequences: np.array (N, seq_len, 224, 224, 3)
        seq_labels: np.array (N,)
    """
    sequences = []
    seq_labels = []

    # Group indices by subject
    sids = subject_ids[indices]
    for sid in np.unique(sids):
        sid_mask = np.where(sids == sid)[0]
        sid_global = indices[sid_mask]  # global indices for this subject
        label = labels[sid_global[0]]

        # Build non-overlapping sequences of length seq_len
        n_seqs = len(sid_global) // seq_len
        if max_seqs is not None and n_seqs > max_seqs:
            # Randomly sample max_seqs starting positions
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
            seq = spectrograms[seq_idx]  # (seq_len, 224, 224, 3)
            sequences.append(seq)
            seq_labels.append(label)

    if len(sequences) == 0:
        return np.array([]).reshape(0, seq_len, 224, 224, 3), np.array([])

    return np.array(sequences, dtype=np.float32), np.array(seq_labels, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL: InceptionV3 + LSTM (matching paper architecture)
# ═══════════════════════════════════════════════════════════════════════════
class InceptionV3_LSTM_Seq(nn.Module):
    """
    Hybrid model: TimeDistributed InceptionV3 + LSTM.

    Input:  (batch, seq_len, 3, 224, 224)
    For each timestep: InceptionV3 → 2048-dim feature vector
    LSTM sees: (batch, seq_len, 2048) → final hidden state
    Classifier: Linear(128, 1) → sigmoid
    """
    def __init__(self, freeze_inception=True):
        super().__init__()

        # Load pretrained InceptionV3
        self.inception = models.inception_v3(weights='IMAGENET1K_V1')
        self.inception.aux_logits = False
        self.inception.AuxLogits = None
        self.inception.fc = nn.Identity()  # Remove classification head

        if freeze_inception:
            for param in self.inception.parameters():
                param.requires_grad = False

        # Upsample 224→299 for InceptionV3
        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

        # LSTM: sees sequence of InceptionV3 features
        self.lstm = nn.LSTM(
            input_size=INCEPTION_FEATURES,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=0.0  # no dropout with 1 layer
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 3, 224, 224)
        """
        batch_size, seq_len, C, H, W = x.shape

        # Process each timestep through InceptionV3
        # Merge batch and seq_len dims: (batch*seq_len, 3, 224, 224)
        x_flat = x.view(batch_size * seq_len, C, H, W)
        x_flat = self.upsample(x_flat)              # (batch*seq_len, 3, 299, 299)

        features_flat = self.inception(x_flat)      # (batch*seq_len, 2048)

        # Reshape back to sequence: (batch, seq_len, 2048)
        features = features_flat.view(batch_size, seq_len, INCEPTION_FEATURES)

        # LSTM over the sequence
        _, (h_n, _) = self.lstm(features)           # h_n: (1, batch, 128)

        # Classify from final hidden state
        out = self.classifier(h_n.squeeze(0))       # (batch, 1)
        return out.squeeze(-1)                       # (batch,)


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

    # Build training sequences (capped per subject)
    X_tr_seq, y_tr = build_sequences(
        train_indices, spectrograms, labels, subject_ids,
        seq_len=SEQ_LEN, max_seqs=MAX_SEQS_PER_SUBJECT
    )

    # Build test sequences (all of them, no cap)
    X_te_seq, y_te = build_sequences(
        test_indices, spectrograms, labels, subject_ids,
        seq_len=SEQ_LEN, max_seqs=None
    )

    if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
        log_msg(f"  SKIP: insufficient sequences (train={len(X_tr_seq)}, test={len(X_te_seq)})")
        return None

    # Stratified validation split (15% of training sequences)
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

    # Class balance for pos_weight
    n_h   = int((y_tr == 0).sum())
    n_mdd = int((y_tr == 1).sum())

    # FIX: pos_weight = n_mdd / n_h  (>1 = penalize missing MDD more)
    # This is the CORRECT direction: we want to catch MDD (reduce false negatives)
    pos_weight = n_mdd / max(n_h, 1)

    log_msg(f"  Sequences — Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te_seq)}")
    log_msg(f"  Class balance (train): H={n_h}, MDD={n_mdd} | pos_weight={pos_weight:.3f}")

    # Create datasets and loaders
    train_ds = EEGSequenceDataset(X_tr,     y_tr)
    val_ds   = EEGSequenceDataset(X_val,    y_val)
    test_ds  = EEGSequenceDataset(X_te_seq, y_te)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log_msg(f"  Data prep: {time.time()-t_prep:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Initialize model
    # ─────────────────────────────────────────────────────────────────────
    model = InceptionV3_LSTM_Seq(freeze_inception=True).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_msg(f"  Model: InceptionV3+LSTM(seq={SEQ_LEN}) | Trainable: {trainable_params:,}/{total_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

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

        log_msg(f"  Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
                f"train={epoch_train_loss:.4f} | val={epoch_val_loss:.4f} | "
                f"{time.time()-t_epoch:.1f}s")

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

    # Collect validation probabilities
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

    # Sweep threshold to maximize F1 on validation set
    best_thresh = 0.5
    best_f1_val = 0.0
    for thresh in np.arange(0.1, 0.91, 0.05):
        val_preds = (val_probs >= thresh).astype(int)
        f1_val = f1_score(val_true, val_preds, zero_division=0)
        if f1_val > best_f1_val:
            best_f1_val  = f1_val
            best_thresh  = thresh

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

    # Metrics
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)

    # Per-class recall (Sensitivity = MDD recall, Specificity = H recall)
    h_mask   = (y_true == 0)
    mdd_mask = (y_true == 1)
    h_recall   = accuracy_score(y_true[h_mask],   y_pred[h_mask])   if h_mask.any()   else 0.0
    mdd_recall = accuracy_score(y_true[mdd_mask], y_pred[mdd_mask]) if mdd_mask.any() else 0.0

    # AUC
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_score = float('nan')

    log_msg(f"  RESULTS: acc={acc:.3f} | f1={f1:.3f} | auc={auc_score:.3f} | thresh={best_thresh:.2f}")
    log_msg(f"           Sensitivity(MDD)={mdd_recall:.3f} | Specificity(H)={h_recall:.3f}")
    log_msg(f"  Fold time: {(time.time()-t_fold_start)/60:.1f} min")

    return {
        'subject':     subject_name,
        'accuracy':    acc,
        'f1':          f1,
        'precision':   precision,
        'recall':      recall,
        'auc':         auc_score,
        'sensitivity': mdd_recall,
        'specificity': h_recall,
        'threshold':   best_thresh,
        'y_true':      y_true,
        'y_pred':      y_pred,
        'y_prob':      y_prob,
        'train_loss':  train_losses,
        'val_loss':    val_losses,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    log_separator()
    log_msg("EXPERIMENT 1: Sequence-based InceptionV3 + LSTM")
    log_msg(f"  Device:     {DEVICE}")
    log_msg(f"  Seq length: {SEQ_LEN} spectrograms per sample")
    log_msg(f"  LSTM:       hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS}")
    log_msg(f"  pos_weight: n_mdd/n_h (corrected — penalizes missing MDD)")
    log_msg(f"  Threshold:  optimized on validation set per fold")
    log_separator()

    # Load data
    log_msg("Loading data...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids  = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    log_msg(f"  Loaded: {spectrograms.shape}, labels={labels.shape}")

    # Get unique subjects in sorted order
    unique_sids = sorted(np.unique(subject_ids))
    total_folds = len(unique_sids)
    log_msg(f"  Subjects: {total_folds} | LOSO folds: {total_folds}")

    # Build subject name map (sorted alphabetically = same as preprocess.py)
    import os as _os
    cleaned_dir = os.path.join(PROJECT_ROOT, "data", "cleaned")
    sid_to_name = {}
    for i, folder in enumerate(sorted(_os.listdir(cleaned_dir))):
        if _os.path.isdir(_os.path.join(cleaned_dir, folder)):
            sid_to_name[i] = folder

    # LOSO loop
    fold_results = []
    all_y_true   = []
    all_y_pred   = []

    for fold_num, test_sid in enumerate(unique_sids, start=1):
        subject_name = sid_to_name.get(test_sid, f"Subject_{test_sid}")

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

        # Save intermediate CSV after each fold
        csv_path = os.path.join(RESULTS_DIR, "fold_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                'subject', 'accuracy', 'f1', 'precision', 'recall',
                'auc', 'sensitivity', 'specificity', 'threshold'
            ])
            writer.writeheader()
            for r in fold_results:
                writer.writerow({k: r[k] for k in writer.fieldnames})

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

    # Per-fold table
    table = PrettyTable()
    table.field_names = ["Subject", "Acc", "F1", "Sensitivity", "Specificity", "AUC", "Thresh"]
    for r in fold_results:
        table.add_row([
            r['subject'],
            f"{r['accuracy']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['sensitivity']:.3f}",
            f"{r['specificity']:.3f}",
            f"{r['auc']:.3f}" if not np.isnan(r['auc']) else "nan",
            f"{r['threshold']:.2f}",
        ])
    log_msg(f"\n{table}")

    # Summary stats
    accs   = [r['accuracy']    for r in fold_results]
    f1s    = [r['f1']          for r in fold_results]
    senss  = [r['sensitivity'] for r in fold_results]
    specs  = [r['specificity'] for r in fold_results]

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Mean", "Std", "Min", "Max"]
    for name, vals in [("Accuracy", accs), ("F1", f1s), ("Sensitivity", senss), ("Specificity", specs)]:
        summary_table.add_row([
            name,
            f"{np.mean(vals):.3f}",
            f"{np.std(vals):.3f}",
            f"{np.min(vals):.3f}",
            f"{np.max(vals):.3f}",
        ])
    log_msg(f"\n{summary_table}")

    log_msg(f"\n  GLOBAL (aggregated across all folds):")
    log_msg(f"    Accuracy:    {global_accuracy:.3f}")
    log_msg(f"    Sensitivity: {global_sensitivity:.3f}")
    log_msg(f"    Specificity: {global_specificity:.3f}")
    log_msg(f"    F1 Score:    {global_f1:.3f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment 1: Sequence-based InceptionV3 + LSTM\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total subjects: {total_folds}\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"LSTM hidden: {LSTM_HIDDEN}\n\n")
        f.write(str(summary_table) + "\n\n")
        f.write(f"Global Accuracy:    {global_accuracy:.4f}\n")
        f.write(f"Global Sensitivity: {global_sensitivity:.4f}\n")
        f.write(f"Global Specificity: {global_specificity:.4f}\n")
        f.write(f"Global F1:          {global_f1:.4f}\n")
        f.write(f"\nTotal time: {(time.time()-t_start)/60:.1f} minutes\n")

    log_msg(f"\n  Results saved to: {RESULTS_DIR}")
    log_msg(f"  Total time: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
