"""
preprocess.py
─────────────
Step 2 of the EEG Depression Detection pipeline.

WHAT IT DOES:
    Reads 120 clean .edf files from data/cleaned/.
    For each file:
        1. Bandpass filter       (0.5–40 Hz)
        2. Artifact rejection    (drop epochs where any channel > 100 µV)
        3. Epoch                 (chop into 700ms non-overlapping segments)
        4. Wavelet decomposition (DWT, db4, 5 levels → keep cA only)
        5. Scalogram             (CWT with Morlet on cA → 2D image)
        6. Resize + normalize    (224×224×3, values in [0, 1])
    Saves three .npy files into data/spectrograms/:
        spectrograms.npy   → shape (N, 224, 224, 3)   the images
        labels.npy         → shape (N,)               0=Healthy, 1=MDD
        subject_ids.npy    → shape (N,)               integer ID per subject
    Also saves a sample scalogram image for visual inspection.

HOW TO RUN:
    conda activate edf_reader
    python scripts/preprocess.py

    No arguments. Finds its own paths relative to project root.
"""

import os
import numpy as np
import mne
import pywt
from PIL import Image
import matplotlib
matplotlib.use('Agg')                          # no display needed — we save to file
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =========================================================================
# SECTION 0: CONFIG
# =========================================================================
# ALL tunable parameters live here. Change a value here and it propagates
# through the entire script. Nothing else has opinions about numbers.
# =========================================================================

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)
CLEANED_DIR     = os.path.join(PROJECT_ROOT, "data", "cleaned")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "data", "spectrograms")

# --- Bandpass filter ---
FILTER_LOW   = 0.5    # Hz — below this is electrical drift
FILTER_HIGH  = 40.0   # Hz — above this is muscle/powerline noise

# --- Artifact rejection ---
ARTIFACT_THRESHOLD_UV = 100.0   # µV — any epoch with a channel above this is discarded

# --- Epoching ---
EPOCH_DURATION_SEC = 0.7        # 700 ms — enough for ~7 alpha cycles

# --- DWT (Discrete Wavelet Transform) ---
DWT_WAVELET = 'db4'             # Daubechies-4 — shape matches EEG oscillations
DWT_LEVELS  = 5                 # decompose to 5 levels, keep only cA (approximation)

# --- CWT Scalogram ---
# Scales control which frequencies the CWT probes.
# We want to cover 0.5–40 Hz. pywt uses scale, not frequency directly.
# For Morlet: frequency ≈ central_freq / (scale * dt)
# We use 64 scales (rows in the scalogram). 64 is enough resolution
# for InceptionV3 to learn from, and keeps memory reasonable.
CWT_WAVELET  = 'morl'           # Morlet — sine wave in a Gaussian envelope
CWT_N_SCALES = 64               # number of frequency rows in the output image

# --- Final image ---
IMG_SIZE = 224                  # InceptionV3 expects 224×224
IMG_CHANNELS = 3                # RGB — InceptionV3 expects 3 channels


# =========================================================================
# SECTION 1: LOAD
# =========================================================================
# Walk through data/cleaned/. Each subfolder is one subject.
# Each subfolder has exactly 3 files: EC, EO, TASK.
# We load all of them. Each file becomes many epochs after epoching.
# All epochs from the same subject share the same subject_id and label.
# =========================================================================

def discover_files(cleaned_dir):
    """
    Returns a list of dicts, one per .edf file found in cleaned_dir.
    Each dict: { filepath, subject_key, group, condition, label, subject_id }

    label:      0 = Healthy, 1 = MDD
    subject_id: a unique integer per subject (used by LOSO later)
    """
    files = []
    subject_id_map = {}   # subject_key → integer ID
    next_id = 0

    for subject_folder in sorted(os.listdir(cleaned_dir)):
        subject_path = os.path.join(cleaned_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        # Assign a unique integer ID to this subject
        if subject_folder not in subject_id_map:
            subject_id_map[subject_folder] = next_id
            next_id += 1

        # Determine group and label from folder name
        # H_4 → group='H', label=0
        # MDD_1 → group='MDD', label=1
        group = 'H' if subject_folder.startswith('H_') else 'MDD'
        label = 0 if group == 'H' else 1

        for fname in sorted(os.listdir(subject_path)):
            if not fname.endswith('.edf'):
                continue
            # Extract condition from filename: H_4_EC.edf → EC
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


# =========================================================================
# SECTION 2: BANDPASS FILTER
# =========================================================================
# mne's filter_data() applies a zero-phase FIR filter.
# Zero-phase means it does not shift the signal in time — important
# for EEG where timing of oscillations matters.
# We pass the raw numpy array (channels × time), not an mne Raw object,
# because we are about to do manual epoching and artifact rejection.
# =========================================================================

def bandpass_filter(data, sfreq, l_freq=FILTER_LOW, h_freq=FILTER_HIGH):
    """
    data:   numpy array, shape (n_channels, n_times)
    sfreq:  sampling frequency in Hz
    Returns: filtered array, same shape
    """
    return mne.filter.filter_data(
        data,
        sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_window='hamming',
        verbose=False
    )


# =========================================================================
# SECTION 3: EPOCH + ARTIFACT REJECTION
# =========================================================================
# We chop the continuous signal into fixed-size segments (epochs).
# Then we check each epoch: if ANY channel in that epoch has an
# amplitude above the threshold, we throw the entire epoch away.
#
# WHY throw the whole epoch, not just that channel?
#   Because a large artifact (like an eye blink) creates electrical
#   field spread — it contaminates nearby channels too. Keeping
#   partial epochs would give the model subtly corrupted data.
#
# WHY non-overlapping epochs?
#   Overlapping epochs from the same recording are not independent.
#   If we overlap and then split by subject (LOSO), it is fine —
#   but it inflates the apparent sample count without adding real
#   information. Non-overlapping is cleaner and more defensible
#   to a reviewer.
# =========================================================================

def epoch_and_reject(data, sfreq, epoch_dur=EPOCH_DURATION_SEC, threshold=ARTIFACT_THRESHOLD_UV):
    """
    data:       numpy array, shape (n_channels, n_times)
    sfreq:      sampling frequency
    epoch_dur:  duration of each epoch in seconds
    threshold:  amplitude threshold in µV

    Returns:    numpy array, shape (n_epochs, n_channels, n_times_per_epoch)
                Only epochs that passed the artifact check are included.
    """
    n_channels, n_times = data.shape
    samples_per_epoch = int(epoch_dur * sfreq)

    # How many complete epochs fit in the recording?
    n_epochs = n_times // samples_per_epoch

    epochs = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end   = start + samples_per_epoch
        epoch = data[:, start:end]                          # shape: (n_channels, samples_per_epoch)

        # Artifact check: max absolute amplitude across ALL channels in this epoch
        # EDF stores in µV already (mne converts on load), so threshold is in µV
        max_amp = np.max(np.abs(epoch)) * 1e6              # convert V → µV for threshold comparison
        if max_amp > threshold:
            continue                                        # reject this epoch

        epochs.append(epoch)

    if len(epochs) == 0:
        return np.array([]).reshape(0, n_channels, samples_per_epoch)

    return np.array(epochs)                                 # shape: (n_good_epochs, n_channels, samples_per_epoch)


# =========================================================================
# SECTION 4: WAVELET DECOMPOSITION (DWT)
# =========================================================================
# DWT decomposes the signal into multiple levels.
# At each level, it splits into:
#   - Approximation (cA) = low frequencies (smoothed version)
#   - Detail (cD)        = high frequencies (the difference)
#
# After 5 levels of decomposition on a 256 Hz signal:
#   Level 1: cA1 captures 0–64 Hz,   cD1 captures 64–128 Hz
#   Level 2: cA2 captures 0–32 Hz,   cD2 captures 32–64 Hz
#   Level 3: cA3 captures 0–16 Hz,   cD3 captures 16–32 Hz
#   Level 4: cA4 captures 0–8 Hz,    cD4 captures 8–16 Hz
#   Level 5: cA5 captures 0–4 Hz,    cD5 captures 4–8 Hz
#
# We already bandpassed to 0.5–40 Hz, so cD1 and cD2 are empty.
# cA5 (0–4 Hz) captures Delta. cD5 (4–8 Hz) captures Theta.
# cD4 (8–16 Hz) captures Alpha + low Beta.
#
# We keep cA (the final approximation after 5 levels = cA5).
# This emphasises the LOW frequencies where depression signatures live.
# The CWT in the next step will then extract the time-frequency
# structure FROM these low-frequency coefficients.
# =========================================================================

def dwt_extract_ca(signal_1d, wavelet=DWT_WAVELET, level=DWT_LEVELS):
    """
    signal_1d:  1D numpy array (one channel, one epoch)
    Returns:    cA coefficients (1D array, shorter than input)

    pywt.wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    We want cA_n which is index 0.
    """
    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)
    return coeffs[0]   # cA at the deepest level


# =========================================================================
# SECTION 5: CWT SCALOGRAM
# =========================================================================
# Now we take the cA coefficients (a 1D signal) and turn them into
# a 2D image using the Continuous Wavelet Transform.
#
# pywt.cwt() takes:
#   - the 1D signal
#   - an array of scales (each scale probes a different frequency)
#   - the wavelet name
# It returns a 2D array: (n_scales, n_time_points)
#
# Each row = one frequency. Each column = one time point.
# Value = how much of that frequency is present at that time.
# That is literally what a spectrogram/scalogram IS.
#
# We take the absolute value (magnitude) because we only care about
# power, not phase (the sign of the oscillation).
# =========================================================================

def make_scalogram(ca_coeffs, n_scales=CWT_N_SCALES, wavelet=CWT_WAVELET):
    """
    ca_coeffs:  1D numpy array (the approximation coefficients)
    Returns:    2D numpy array, shape (n_scales, len(ca_coeffs))

    The scales array: we use 1 to n_scales. These are arbitrary units
    in pywt — what matters is the RANGE covers the frequencies we want.
    For our cA signal (already low-frequency), scales 1–64 give us
    good resolution across the relevant bands.
    """
    scales = np.arange(1, n_scales + 1)
    scalogram, _ = pywt.cwt(ca_coeffs, scales, wavelet)
    return np.abs(scalogram)                               # magnitude only


# =========================================================================
# SECTION 6: RESIZE + NORMALIZE → 224×224×3
# =========================================================================
# InceptionV3 expects a 224×224×3 (height × width × RGB channels) image
# with pixel values in [0, 1].
#
# Our scalogram is (64, T) where T = length of cA coefficients.
# We need to:
#   1. Normalize to [0, 255] range (so PIL can handle it)
#   2. Convert to a PIL Image
#   3. Resize to 224×224 using bilinear interpolation
#   4. Convert to RGB (3 channels — just stack the same grayscale 3×)
#   5. Convert back to numpy array
#   6. Scale to [0, 1] (what the model expects)
#
# WHY replicate grayscale to RGB instead of using a colormap?
#   A colormap (like 'viridis') maps values to colours. It looks nice
#   in papers but it ADDS information that is not in the data — the
#   colour mapping is arbitrary. Replicating grayscale to 3 channels
#   keeps the image honest. The model learns from the actual signal
#   values, not from a colour scale we invented.
# =========================================================================

def scalogram_to_image(scalogram, img_size=IMG_SIZE):
    """
    scalogram:  2D numpy array, shape (n_scales, n_times)
    Returns:    3D numpy array, shape (img_size, img_size, 3), values in [0, 1]
    """
    # Normalize to [0, 255] for PIL
    smin, smax = scalogram.min(), scalogram.max()
    if smax - smin == 0:
        # Flat signal — entire image is zeros. Rare but handle it.
        normalized = np.zeros_like(scalogram, dtype=np.uint8)
    else:
        normalized = ((scalogram - smin) / (smax - smin) * 255).astype(np.uint8)

    # PIL Image → resize → back to numpy
    img = Image.fromarray(normalized, mode='L')            # 'L' = grayscale
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0    # back to [0, 1]

    # Stack to 3 channels: (H, W) → (H, W, 3)
    img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

    return img_rgb


# =========================================================================
# SECTION 7: PROCESS ONE FILE → LIST OF IMAGES
# =========================================================================
# This is the function that chains everything together for a single .edf file.
# It returns a list of 224×224×3 images — one per valid epoch.
#
# The channel averaging step:
#   We have 20 channels per epoch. Each channel is an independent
#   electrical recording from a different spot on the scalp.
#   We could make 20 scalograms per epoch (one per channel) — but
#   that would be 20× more data and 20× slower, and InceptionV3
#   would see each channel in isolation (no spatial relationships).
#
#   Instead: average across channels BEFORE the wavelet steps.
#   This gives us ONE signal per epoch that represents the overall
#   brain activity pattern. It is the standard approach in EEG
#   classification when using a single-image model like InceptionV3.
#   The spatial information across channels is captured implicitly
#   by the averaging — regions with strong coherent activity will
#   dominate the average.
# =========================================================================

def process_file(filepath):
    """
    filepath:  path to a single .edf file in data/cleaned/
    Returns:   list of numpy arrays, each shape (224, 224, 3)
               One image per valid epoch extracted from this file.
               Empty list if no epochs survived artifact rejection.
    """
    # Load the EDF file. preload=True because we need the signal data.
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    sfreq = raw.info['sfreq']                              # sampling rate (256 Hz)
    data  = raw.get_data()                                 # shape: (20, n_times), in Volts

    # Step 1: Bandpass filter
    data = bandpass_filter(data, sfreq)

    # Step 2 + 3: Epoch + artifact rejection
    epochs = epoch_and_reject(data, sfreq)                 # shape: (n_epochs, 20, samples_per_epoch)
    if len(epochs) == 0:
        return []

    # Process each epoch into a scalogram image
    images = []
    for epoch in epochs:
        # Average across 20 channels → 1D signal per epoch
        # shape: (20, samples_per_epoch) → (samples_per_epoch,)
        avg_signal = np.mean(epoch, axis=0)

        # Step 4: DWT → extract cA
        ca = dwt_extract_ca(avg_signal)

        # Step 5: CWT → scalogram
        scalogram = make_scalogram(ca)

        # Step 6: Resize + normalize → 224×224×3
        img = scalogram_to_image(scalogram)

        images.append(img)

    return images


# =========================================================================
# SECTION 8: VISUALIZATION
# =========================================================================
# Save a sample scalogram to disk so you can visually inspect what
# the model will actually see. One Healthy, one MDD, side by side.
# This is also useful for the paper — reviewers like to see this.
# =========================================================================

def save_sample_scalograms(all_files, output_dir):
    """
    Finds the first H and first MDD file, processes them,
    saves a side-by-side comparison image.
    """
    h_file  = next(f for f in all_files if f['group'] == 'H')
    mdd_file = next(f for f in all_files if f['group'] == 'MDD')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_style("whitegrid")

    for ax, file_info in [(axes[0], h_file), (axes[1], mdd_file)]:
        title = f"{file_info['group']} — {file_info['subject_key']} {file_info['condition']}"
        # Load + filter + epoch (reuse the same pipeline)
        raw = mne.io.read_raw_edf(file_info['filepath'], preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        data  = bandpass_filter(raw.get_data(), sfreq)
        epochs = epoch_and_reject(data, sfreq)

        if len(epochs) == 0:
            ax.set_title(f"{title}\n(no valid epochs)")
            continue

        # Take the first valid epoch, average channels, DWT, CWT
        avg_signal = np.mean(epochs[0], axis=0)
        ca         = dwt_extract_ca(avg_signal)
        scalogram  = make_scalogram(ca)

        # Plot with a proper colormap for the VISUALIZATION only
        # (the model still gets grayscale-replicated RGB — see Section 6)
        ax.imshow(scalogram, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("CWT Scale")

    plt.suptitle("Sample Scalograms: Healthy vs MDD\n(what InceptionV3 will see)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_scalograms.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sample_scalograms.png")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("\n" + "=" * 70)
    print("  PREPROCESS")
    print("  Reading from: ", CLEANED_DIR)
    print("  Writing to:   ", OUTPUT_DIR)
    print("=" * 70 + "\n")

    # Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover all 120 files
    all_files, subject_id_map = discover_files(CLEANED_DIR)
    print(f"  Found {len(all_files)} files across {len(subject_id_map)} subjects\n")

    # --- Save sample scalograms first (fast, one epoch each) ---
    print("  Generating sample scalogram visualization...")
    save_sample_scalograms(all_files, OUTPUT_DIR)

    # --- Process every file ---
    all_spectrograms = []    # will hold all 224×224×3 images
    all_labels       = []    # 0 or 1 per image
    all_subject_ids  = []    # integer subject ID per image

    # Running stats for the progress report
    total_epochs_before_rejection = 0
    total_epochs_after_rejection  = 0
    per_subject_stats = {}   # subject_key → {before, after, images}

    print("  Processing files...")
    for file_info in tqdm(all_files, desc="  Files", unit="file"):
        subj = file_info['subject_key']

        # Count epochs before rejection (just for reporting)
        raw = mne.io.read_raw_edf(file_info['filepath'], preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        n_times = raw.get_data().shape[1]
        samples_per_epoch = int(EPOCH_DURATION_SEC * sfreq)
        n_possible_epochs = n_times // samples_per_epoch
        total_epochs_before_rejection += n_possible_epochs

        # The actual processing
        images = process_file(file_info['filepath'])
        total_epochs_after_rejection += len(images)

        # Track per-subject stats
        if subj not in per_subject_stats:
            per_subject_stats[subj] = {'before': 0, 'after': 0}
        per_subject_stats[subj]['before'] += n_possible_epochs
        per_subject_stats[subj]['after']  += len(images)

        # Append to the master lists
        for img in images:
            all_spectrograms.append(img)
            all_labels.append(file_info['label'])
            all_subject_ids.append(file_info['subject_id'])

    # --- Convert to numpy arrays ---
    spectrograms = np.array(all_spectrograms, dtype=np.float32)   # (N, 224, 224, 3)
    labels       = np.array(all_labels,       dtype=np.int32)     # (N,)
    subject_ids  = np.array(all_subject_ids,  dtype=np.int32)     # (N,)

    # --- Save to disk ---
    np.save(os.path.join(OUTPUT_DIR, "spectrograms.npy"), spectrograms)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"),       labels)
    np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"),  subject_ids)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"\n  Total spectrograms saved:  {len(spectrograms)}")
    print(f"    Shape:                   {spectrograms.shape}")
    print(f"    Healthy (label=0):       {np.sum(labels == 0)}")
    print(f"    MDD     (label=1):       {np.sum(labels == 1)}")
    print(f"    Unique subjects:         {len(np.unique(subject_ids))}")
    print(f"\n  Artifact rejection:")
    print(f"    Epochs before rejection: {total_epochs_before_rejection}")
    print(f"    Epochs after rejection:  {total_epochs_after_rejection}")
    print(f"    Rejection rate:          {(1 - total_epochs_after_rejection/total_epochs_before_rejection)*100:.1f}%")

    print(f"\n  Per-subject epoch counts (after rejection):")
    print(f"    {'Subject':<12} {'Epochs':<8} {'Label'}")
    print(f"    {'─' * 35}")
    for subj in sorted(per_subject_stats.keys()):
        label = 'H' if subj.startswith('H') else 'MDD'
        print(f"    {subj:<12} {per_subject_stats[subj]['after']:<8} {label}")

    print(f"\n  Files saved to: {OUTPUT_DIR}")
    print(f"    spectrograms.npy   — the images")
    print(f"    labels.npy         — 0=Healthy, 1=MDD")
    print(f"    subject_ids.npy    — which subject each image came from")
    print(f"    sample_scalograms.png — visual check\n")
    print(f"  Next step:  set up eeg_train env, then run train.py\n")


if __name__ == '__main__':
    main()
