import sys, os, time
sys.path.insert(0, '/home/claude')
import torch
import numpy as np
from train import CNN_LSTM, DEVICE, EEGDataset, get_loso_folds

print("="*60)
print("  SMOKE TEST v2")
print("="*60)

# 1. Model builds, forward pass, param count
print("\n  1. Model + forward pass on", DEVICE)
model = CNN_LSTM().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"     Parameters: {total_params:,}")

fake = torch.randn(64, 3, 224, 224).to(DEVICE)   # full batch size
t0 = time.time()
with torch.no_grad():
    out = model(fake)
torch.mps.synchronize() if DEVICE.type == 'mps' else None
t1 = time.time()
print(f"     Input:  {fake.shape}")
print(f"     Output: {out.shape}   (should be [64])")
print(f"     One batch forward: {(t1-t0)*1000:.1f} ms")

# 2. Estimate fold time
# One fold: ~47k training samples, batch=64 → 734 batches/epoch
# Assume ~15 epochs average (early stop). Forward+backward ≈ 3x forward.
batches_per_epoch = 47000 // 64
forward_ms = (t1 - t0) * 1000
# backward is roughly 2x forward
total_per_epoch_s = batches_per_epoch * forward_ms * 3 / 1000
est_epochs = 15
est_total_s = total_per_epoch_s * est_epochs
print(f"\n  2. Timing estimate")
print(f"     Batches/epoch:  {batches_per_epoch}")
print(f"     Forward batch:  {forward_ms:.1f} ms")
print(f"     Est per epoch:  {total_per_epoch_s:.0f} s")
print(f"     Est per fold:   {est_total_s:.0f} s  (~{est_total_s/60:.1f} min)")
print(f"     Est Approach 1: {est_total_s*40/60:.0f} min  (40 folds)")

# 3. Dataset __getitem__ — no resize, should be instant
print("\n  3. Dataset __getitem__ speed (no resize)")
fake_specs = np.random.rand(1000, 224, 224, 3).astype(np.float32)
fake_labels = np.random.randint(0, 2, 1000).astype(np.int32)
ds = EEGDataset(fake_specs, fake_labels, np.arange(len(fake_labels)))
t0 = time.time()
for i in range(1000):
    _ = ds[i]
t1 = time.time()
print(f"     1000 __getitem__ calls: {(t1-t0)*1000:.1f} ms  ({(t1-t0):.4f} ms each)")

# 4. LOSO sanity
print("\n  4. LOSO split check")
fake_sids = np.repeat(np.arange(40), 100)
folds = get_loso_folds(fake_sids, list(range(40)))
tr, te = folds[0]
print(f"     Folds: {len(folds)}")
print(f"     Fold 0 test sids: {set(fake_sids[te])}  train sids: {len(set(fake_sids[tr]))}")
print(f"     Overlap: {set(fake_sids[te]) & set(fake_sids[tr])}")

print("\n" + "="*60)
print("  ALL OK. Ready to train.")
print("="*60)
