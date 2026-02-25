"""
preprocess_spatial.py
─────────────────────
Spatial RGB preprocessing for EEG Depression Detection.

WHAT IT DOES:
    Instead of averaging ALL 20 channels into a single grayscale image
    (then replicating to fake RGB), this script creates TRUE RGB images
    where each colour channel represents a different brain region:

        RED   = Frontal   (Fp1, F3, F7, Fz, Fp2, F4, F8)  — 7 channels
        GREEN = Central   (C3, T3, C4, T4, Cz)             — 5 channels
        BLUE  = Posterior  (P3, O1, T5, P4, O2, T6, Pz)    — 7 channels

    Channel A2-A1 (reference) is excluded.

    For each epoch:
        1. Group the 19 EEG channels into 3 spatial regions
        2. Average channels within each region → 3 separate 1D signals
        3. Each signal → DWT (db4, 5 levels) → CWT (Morlet, 64 scales) → scalogram
        4. Stack 3 scalograms as R, G, B → (224, 224, 3) TRUE RGB

    This preserves spatial information that is CRITICAL for distinguishing:
        - Frontal Alpha Asymmetry (depression biomarker)
        - Frontal vs Posterior gradients
        - Temporal lobe activity patterns

HOW TO RUN:
    conda activate edf_reader
    python scripts/preprocess_spatial.py

OUTPUT:
    data/spectrograms_spatial/
        spectrograms.npy   — (N, 224, 224, 3) TRUE spatial RGB images
        labels.npy         — (N,) 0=Healthy, 1=MDD
        subject_ids.npy    — (N,) integer subject ID
"""

import os
import numpy as np
import mne
import pywt
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =========================================================================
# SECTION 0: CONFIG
# =========================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLEANED_DIR  = os.path.join(PROJECT_ROOT, "data", "cleaned")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "spectrograms_spatial")

# Bandpass filter
FILTER_LOW  = 0.5
FILTER_HIGH = 40.0

# Artifact rejection
ARTIFACT_THRESHOLD_UV = 100.0

# Epoching
EPOCH_DURATION_SEC = 0.7

# DWT
DWT_WAVELET = 'db4'
DWT_LEVELS  = 5

# CWT Scalogram
CWT_WAVELET  = 'morl'
CWT_N_SCALES = 64

# Final image
IMG_SIZE = 224

# =========================================================================
# CHANNEL GROUPINGS
# =========================================================================
# Based on 10-20 system channels found in our EDF files:
#   ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE',
#    'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE',
#    'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE',
#    'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1']
#
# Map channel names to their standard 10-20 electrode name (strip prefix/suffix)
# Then group by brain region:

FRONTAL_ELECTRODES  = ['Fp1', 'F3', 'F7', 'Fz', 'Fp2', 'F4', 'F8']   # 7 channels
CENTRAL_ELECTRODES  = ['C3', 'T3', 'C4', 'T4', 'Cz']                  # 5 channels
POSTERIOR_ELECTRODES = ['P3', 'O1', 'T5', 'P4', 'O2', 'T6', 'Pz']     # 7 channels
EXCLUDED_ELECTRODES  = ['A2-A1']                                        # reference, excluded

# Total: 7 + 5 + 7 = 19 channels (20th is reference)


def get_electrode_name(ch_name):
    """Extract standard electrode name from EDF channel name.
    'EEG Fp1-LE' → 'Fp1'
    'EEG A2-A1'  → 'A2-A1'
    """
    name = ch_name.replace('EEG ', '').replace('-LE', '')
    return name


def get_channel_groups(ch_names):
    """Map channel names to spatial group indices.
    
    Returns:
        frontal_idx: list of channel indices for frontal region
        central_idx: list of channel indices for central region
        posterior_idx: list of channel indices for posterior region
    """
    frontal_idx  = []
    central_idx  = []
    posterior_idx = []

    for i, ch in enumerate(ch_names):
        electrode = get_electrode_name(ch)
        if electrode in FRONTAL_ELECTRODES:
            frontal_idx.append(i)
        elif electrode in CENTRAL_ELECTRODES:
            central_idx.append(i)
        elif electrode in POSTERIOR_ELECTRODES:
            posterior_idx.append(i)
        elif electrode in EXCLUDED_ELECTRODES:
            pass  # skip reference
        else:
            print(f"  WARNING: Unknown electrode '{electrode}' (channel '{ch}'), skipping")

    return frontal_idx, central_idx, posterior_idx


# =========================================================================
# SECTION 1: REUSE FUNCTIONS FROM preprocess.py
# =========================================================================

def discover_files(cleaned_dir):
    """Same as preprocess.py — find all EDF files."""
    files = []
    subject_id_map = {}
    next_id = 0

    for subject_folder in sorted(os.listdir(cleaned_dir)):
        subject_path = os.path.join(cleaned_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        if subject_folder not in subject_id_map:
            subject_id_map[subject_folder] = next_id
            next_id += 1

        group = 'H' if subject_folder.startswith('H_') else 'MDD'
        label = 0 if group == 'H' else 1

        for fname in sorted(os.listdir(subject_path)):
            if not fname.endswith('.edf'):
                continue
            condition = fname.replace('.edf', '').split('_')[-1]

            files.append({
                'filepath':    os.path.join(subject_path, fname),
                'subject_key': subject_folder,
                'group':       group,
                'condition':   condition,
                'label':       label,
                'subject_id':  subject_id_map[subject_folder]
            })

    return files, subject_id_map


def bandpass_filter(data, sfreq, l_freq=FILTER_LOW, h_freq=FILTER_HIGH):
    """Bandpass filter using MNE (zero-phase FIR)."""
    return mne.filter.filter_data(
        data, sfreq,
        l_freq=l_freq, h_freq=h_freq,
        method='fir', fir_window='hamming', verbose=False
    )


def epoch_and_reject(data, sfreq, epoch_dur=EPOCH_DURATION_SEC, threshold=ARTIFACT_THRESHOLD_UV):
    """Epoch + artifact rejection. Returns (n_epochs, n_channels, n_times)."""
    n_channels, n_times = data.shape
    samples_per_epoch = int(epoch_dur * sfreq)
    n_epochs = n_times // samples_per_epoch

    epochs = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end   = start + samples_per_epoch
        epoch = data[:, start:end]

        max_amp = np.max(np.abs(epoch)) * 1e6  # V → µV
        if max_amp > threshold:
            continue

        epochs.append(epoch)

    if len(epochs) == 0:
        return np.array([]).reshape(0, n_channels, samples_per_epoch)

    return np.array(epochs)


def dwt_extract_ca(signal_1d, wavelet=DWT_WAVELET, level=DWT_LEVELS):
    """DWT decomposition → cA coefficients."""
    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)
    return coeffs[0]


def make_scalogram(ca_coeffs, n_scales=CWT_N_SCALES, wavelet=CWT_WAVELET):
    """CWT scalogram from cA coefficients."""
    scales = np.arange(1, n_scales + 1)
    scalogram, _ = pywt.cwt(ca_coeffs, scales, wavelet)
    return np.abs(scalogram)


def scalogram_to_channel(scalogram, img_size=IMG_SIZE):
    """Convert a 2D scalogram to a single (img_size, img_size) channel, values in [0, 1]."""
    smin, smax = scalogram.min(), scalogram.max()
    if smax - smin == 0:
        normalized = np.zeros_like(scalogram, dtype=np.uint8)
    else:
        normalized = ((scalogram - smin) / (smax - smin) * 255).astype(np.uint8)

    img = Image.fromarray(normalized, mode='L')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# =========================================================================
# SECTION 2: SPATIAL RGB PROCESSING
# =========================================================================

def process_file_spatial(filepath, frontal_idx, central_idx, posterior_idx):
    """
    Process one EDF file into spatial RGB images.
    
    For each valid epoch:
        - Average frontal channels → DWT → CWT → Red channel
        - Average central channels → DWT → CWT → Green channel
        - Average posterior channels → DWT → CWT → Blue channel
        - Stack → (224, 224, 3) TRUE RGB
    
    Returns: list of (224, 224, 3) numpy arrays
    """
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # (n_channels, n_times)

    # Bandpass filter
    data = bandpass_filter(data, sfreq)

    # Epoch + artifact rejection → (n_epochs, n_channels, n_times)
    epochs = epoch_and_reject(data, sfreq)
    if len(epochs) == 0:
        return []

    images = []
    for epoch in epochs:
        # Average channels within each spatial group
        frontal_signal  = np.mean(epoch[frontal_idx],  axis=0)  # (n_times,)
        central_signal  = np.mean(epoch[central_idx],  axis=0)
        posterior_signal = np.mean(epoch[posterior_idx], axis=0)

        # Each signal → DWT → CWT → resize to (224, 224)
        r_channel = scalogram_to_channel(make_scalogram(dwt_extract_ca(frontal_signal)))
        g_channel = scalogram_to_channel(make_scalogram(dwt_extract_ca(central_signal)))
        b_channel = scalogram_to_channel(make_scalogram(dwt_extract_ca(posterior_signal)))

        # Stack → (224, 224, 3) TRUE RGB
        img_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        images.append(img_rgb)

    return images


# =========================================================================
# SECTION 3: VISUALIZATION
# =========================================================================

def save_sample_comparison(all_files, output_dir, frontal_idx, central_idx, posterior_idx):
    """Save side-by-side comparison: Healthy vs MDD spatial RGB images."""
    h_file  = next(f for f in all_files if f['group'] == 'H')
    mdd_file = next(f for f in all_files if f['group'] == 'MDD')

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for row, file_info in enumerate([h_file, mdd_file]):
        title = f"{file_info['group']} — {file_info['subject_key']} {file_info['condition']}"

        raw = mne.io.read_raw_edf(file_info['filepath'], preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        data = bandpass_filter(raw.get_data(), sfreq)
        epochs = epoch_and_reject(data, sfreq)

        if len(epochs) == 0:
            for ax in axes[row]:
                ax.set_title(f"{title}\n(no valid epochs)")
            continue

        epoch = epochs[0]

        # Process each region
        signals = {
            'Frontal (R)': np.mean(epoch[frontal_idx], axis=0),
            'Central (G)': np.mean(epoch[central_idx], axis=0),
            'Posterior (B)': np.mean(epoch[posterior_idx], axis=0),
        }

        channels = []
        for i, (region_name, signal) in enumerate(signals.items()):
            ca = dwt_extract_ca(signal)
            scalogram = make_scalogram(ca)
            channel = scalogram_to_channel(scalogram)
            channels.append(channel)

            axes[row, i].imshow(scalogram, aspect='auto', cmap='viridis', origin='lower')
            axes[row, i].set_title(f"{title}\n{region_name}", fontsize=9, fontweight='bold')
            axes[row, i].set_xlabel("Time")
            axes[row, i].set_ylabel("CWT Scale")

        # Combined RGB
        rgb = np.stack(channels, axis=-1)
        axes[row, 3].imshow(rgb)
        axes[row, 3].set_title(f"{title}\nCombined RGB", fontsize=9, fontweight='bold')
        axes[row, 3].axis('off')

    plt.suptitle("Spatial RGB: Frontal(R) + Central(G) + Posterior(B)\nHealthy (top) vs MDD (bottom)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_spatial_rgb.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sample_spatial_rgb.png")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("\n" + "=" * 70)
    print("  SPATIAL RGB PREPROCESSING")
    print("  Reading from: ", CLEANED_DIR)
    print("  Writing to:   ", OUTPUT_DIR)
    print("=" * 70 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover all files
    all_files, subject_id_map = discover_files(CLEANED_DIR)
    print(f"  Found {len(all_files)} files across {len(subject_id_map)} subjects\n")

    # Get channel groups from first file
    first_file = all_files[0]['filepath']
    raw = mne.io.read_raw_edf(first_file, preload=False, verbose=False)
    ch_names = raw.ch_names
    print(f"  Channels ({len(ch_names)}): {ch_names}\n")

    frontal_idx, central_idx, posterior_idx = get_channel_groups(ch_names)
    print(f"  Frontal  ({len(frontal_idx)} ch): indices {frontal_idx}")
    print(f"  Central  ({len(central_idx)} ch): indices {central_idx}")
    print(f"  Posterior ({len(posterior_idx)} ch): indices {posterior_idx}")
    total_mapped = len(frontal_idx) + len(central_idx) + len(posterior_idx)
    print(f"  Total mapped: {total_mapped}/20 channels (1 reference excluded)\n")

    # Save visualization first
    print("  Generating spatial RGB sample visualization...")
    save_sample_comparison(all_files, OUTPUT_DIR, frontal_idx, central_idx, posterior_idx)

    # Process every file
    all_spectrograms = []
    all_labels       = []
    all_subject_ids  = []

    total_epochs_before = 0
    total_epochs_after  = 0
    per_subject_stats = {}

    print("  Processing files...")
    for file_info in tqdm(all_files, desc="  Files", unit="file"):
        subj = file_info['subject_key']

        # Count epochs before rejection
        raw = mne.io.read_raw_edf(file_info['filepath'], preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        n_times = raw.get_data().shape[1]
        samples_per_epoch = int(EPOCH_DURATION_SEC * sfreq)
        n_possible_epochs = n_times // samples_per_epoch
        total_epochs_before += n_possible_epochs

        # Spatial RGB processing
        images = process_file_spatial(
            file_info['filepath'], frontal_idx, central_idx, posterior_idx
        )
        total_epochs_after += len(images)

        if subj not in per_subject_stats:
            per_subject_stats[subj] = {'before': 0, 'after': 0}
        per_subject_stats[subj]['before'] += n_possible_epochs
        per_subject_stats[subj]['after']  += len(images)

        for img in images:
            all_spectrograms.append(img)
            all_labels.append(file_info['label'])
            all_subject_ids.append(file_info['subject_id'])

    # Convert to numpy arrays
    spectrograms = np.array(all_spectrograms, dtype=np.float32)  # (N, 224, 224, 3)
    labels       = np.array(all_labels,       dtype=np.int32)
    subject_ids  = np.array(all_subject_ids,  dtype=np.int32)

    # Save
    np.save(os.path.join(OUTPUT_DIR, "spectrograms.npy"), spectrograms)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"),       labels)
    np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"),  subject_ids)

    # Print summary
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"\n  Total spatial RGB images saved: {len(spectrograms)}")
    print(f"    Shape: {spectrograms.shape}")
    print(f"    Healthy (label=0): {np.sum(labels == 0)}")
    print(f"    MDD     (label=1): {np.sum(labels == 1)}")
    print(f"    Unique subjects:   {len(np.unique(subject_ids))}")
    print(f"\n  Artifact rejection:")
    print(f"    Epochs before: {total_epochs_before}")
    print(f"    Epochs after:  {total_epochs_after}")
    print(f"    Rejection rate: {(1 - total_epochs_after/total_epochs_before)*100:.1f}%")

    print(f"\n  Channel mapping:")
    print(f"    R (Red)   = Frontal   ({len(frontal_idx)} ch)")
    print(f"    G (Green) = Central   ({len(central_idx)} ch)")
    print(f"    B (Blue)  = Posterior ({len(posterior_idx)} ch)")

    print(f"\n  Per-subject epoch counts:")
    print(f"    {'Subject':<12} {'Epochs':<8} {'Label'}")
    print(f"    {'─' * 35}")
    for subj in sorted(per_subject_stats.keys()):
        label = 'H' if subj.startswith('H') else 'MDD'
        print(f"    {subj:<12} {per_subject_stats[subj]['after']:<8} {label}")

    print(f"\n  Files saved to: {OUTPUT_DIR}")
    print(f"    spectrograms.npy  — spatial RGB images")
    print(f"    labels.npy        — 0=Healthy, 1=MDD")
    print(f"    subject_ids.npy   — subject IDs")
    print(f"\n  Next step: python scripts/train_exp_final.py\n")


if __name__ == '__main__':
    main()
