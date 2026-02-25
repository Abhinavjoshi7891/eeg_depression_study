"""
clean_dataset.py
────────────────
Step 1 of the EEG Depression Detection pipeline.

WHAT IT DOES:
    Reads 181 raw .edf files from data/raw/.
    Applies every corruption decision from the dataset audit.
    Drops the two auxiliary channels (23A-23R, 24A-24R) from 22ch TASK files.
    Copies 120 verified files into data/cleaned/, one folder per subject.
    Writes three manifest CSVs into data/manifests/.

HOW TO RUN:
    conda activate edf_reader
    python scripts/clean_dataset.py

    The script finds its own paths relative to the project root.
    No arguments needed.
"""

import os
import re
import shutil
import random
import mne
import pandas as pd
from collections import defaultdict

# =========================================================================
# SECTION 0: CONFIG
# =========================================================================
# ALL decisions from the audit live here. If anything ever changes,
# change it in this one place. Nothing else in the script has opinions
# about what is clean or corrupt.
# =========================================================================

# The project root. Everything is relative to this.
# Assumes this script lives at: eeg_depression/scripts/clean_dataset.py
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
CLEANED_DIR  = os.path.join(PROJECT_ROOT, "data", "cleaned")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")

# --- Subjects to drop entirely (every file from these subjects is removed) ---
# MDD_25: EO file belongs to roslina_w0, EC+TASK to Rossita_W0. Cannot trust.
# H_30:   Identical clone of H_27 (same person, same timestamps). Keep H_27.
# MDD_34: Identical clone of MDD_33 (same person, same timestamps). Keep MDD_33.
DROP_SUBJECTS_ENTIRELY = {"MDD_25", "H_30", "MDD_34"}

# --- Individual TASK files that are copies of someone else's data ---
# These 9 subjects have TASK files whose header says "AzrieMurin" but
# their EC/EO files say a completely different person. The TASK files
# are literal copies of H_4's TASK (H_4 IS AzrieMurin — H_4 is clean).
# H_13's TASK says "Lailah" but EC/EO say "JailineLian" — same pattern.
DROP_TASK_FILES = {
    "H_1", "H_2", "H_3", "H_7", "H_10", "H_11", "H_12", "H_17", "H_29",
    "H_13"   # TASK = Lailah, not JailineLian
}

# --- The two auxiliary channels to remove from 22-channel TASK files ---
# These are non-standard channels added only during TASK recording.
# The 20 channels in EC/EO are a pure subset of the 22 in TASK.
# Dropping these makes every file 20 channels with zero EEG data loss.
CHANNELS_TO_DROP = ["EEG 23A-23R", "EEG 24A-24R"]

# --- Seeds for Approach 2 (balanced 28-subject dataset) ---
# For each seed we randomly select 14 MDD subjects from the 26 clean ones.
# 5 seeds → 5 different balanced datasets → train.py reports mean ± std.
BALANCED_SEEDS = [42, 123, 7, 256, 999]
N_MDD_TO_SELECT = 14   # match the 14 Healthy subjects we have


# =========================================================================
# SECTION 1: PARSE FILENAMES
# =========================================================================
# Same regex that the audit script uses. Proven against every filename
# pattern in this dataset.
#
# Breakdown:
#   ^(\d+_)?          → optional numeric prefix like "6921143_" (flagged as duplicate)
#   (H|MDD)           → group label
#   \s+               → one or more spaces (handles inconsistent whitespace)
#   S(\d+)            → literal "S" + subject number
#   \s+               → one or more spaces
#   (EC|EO|TASK)      → condition
#   \.edf$            → must end with .edf
# =========================================================================

FILENAME_PATTERN = re.compile(
    r'^(\d+_)?(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)\.edf$',
    re.IGNORECASE
)

def parse_filename(filename):
    """
    "H S4 EC.edf"            → {'group':'H',   'num':4,  'condition':'EC',  'has_prefix':False}
    "MDD S1  EO.edf"         → {'group':'MDD', 'num':1,  'condition':'EO',  'has_prefix':False}
    "6921143_H S15 EO.edf"   → {'group':'H',   'num':15, 'condition':'EO',  'has_prefix':True}
    Returns None if it does not match.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {
        'group':      match.group(2).upper(),
        'num':        int(match.group(3)),
        'condition':  match.group(4).upper(),
        'has_prefix': match.group(1) is not None
    }


# =========================================================================
# SECTION 2: READ HEADERS
# =========================================================================
# We only need two things from the header here:
#   - channel count (to know if we need to drop the two extras)
#   - channel names (to do the actual dropping)
# Identity verification was done in the audit. We trust those results.
# =========================================================================

def read_channel_info(filepath):
    """
    Returns the list of channel names in the file.
    Does NOT load any signal data — preload=False means only the header
    is read from disk. Fast.
    """
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    return raw.info.ch_names


# =========================================================================
# SECTION 3: FLAG EVERY FILE
# =========================================================================
# Walk through all 181 files. For each one, decide: KEEP or DROP.
# If DROP, record the reason. This becomes dropped_subjects.csv.
# =========================================================================

def flag_all_files(raw_dir):
    """
    Returns two lists:
        kept  → list of dicts, each is one file that passed all checks
        dropped → list of dicts, each is one file that was removed + why

    Each dict has: filename, filepath, group, subject_key, condition, reason
    """
    kept    = []
    dropped = []

    filenames = sorted([f for f in os.listdir(raw_dir) if f.endswith('.edf')])

    # --- First pass: parse every filename, drop the unparseable and prefixed ---
    parsed_files = []
    for filename in filenames:
        filepath = os.path.join(raw_dir, filename)
        info = parse_filename(filename)

        if info is None:
            # Should not happen — audit proved every file parses.
            # But be safe.
            dropped.append({
                'filename':    filename,
                'subject_key': 'UNKNOWN',
                'condition':   'UNKNOWN',
                'reason':      'Filename does not match expected pattern'
            })
            continue

        subject_key = f"{info['group']}_{info['num']}"

        if info['has_prefix']:
            # The two "6921143_H S15 EO.edf" style files.
            # H_15 already has a valid EO. These are duplicates.
            dropped.append({
                'filename':    filename,
                'subject_key': subject_key,
                'condition':   info['condition'],
                'reason':      'Numeric prefix duplicate — H_15 already has valid EO'
            })
            continue

        # Survived first pass. Keep it for now, flag later if needed.
        parsed_files.append({
            'filename':    filename,
            'filepath':    filepath,
            'group':       info['group'],
            'subject_key': subject_key,
            'condition':   info['condition']
        })

    # --- Second pass: apply the corruption rules ---
    for f in parsed_files:
        subj = f['subject_key']
        cond = f['condition']

        # Rule 1: entire subject dropped
        if subj in DROP_SUBJECTS_ENTIRELY:
            reason_map = {
                'MDD_25': 'Identity mismatch — EO belongs to roslina_w0, EC+TASK to Rossita_W0',
                'H_30':   'Duplicate subject — identical clone of H_27 (Wajid_W0, same timestamps)',
                'MDD_34': 'Duplicate subject — identical clone of MDD_33 (Zakiah_Zainul_W0, same timestamps)'
            }
            dropped.append({
                'filename':    f['filename'],
                'subject_key': subj,
                'condition':   cond,
                'reason':      reason_map[subj]
            })
            continue

        # Rule 2: specific TASK file is a copy of someone else's data
        if subj in DROP_TASK_FILES and cond == 'TASK':
            if subj == 'H_13':
                reason = 'TASK file belongs to Lailah, not JailineLian (EC/EO subject)'
            else:
                reason = 'TASK file is a copy of H_4 (AzrieMurin) — wrong subject'
            dropped.append({
                'filename':    f['filename'],
                'subject_key': subj,
                'condition':   cond,
                'reason':      reason
            })
            continue

        # Passed all rules. Keep it.
        kept.append(f)

    # --- Third pass: check trio completeness on what remains ---
    # Group by subject, see who has all three conditions.
    # Anyone missing a condition after the above removals gets dropped.
    subject_conditions = defaultdict(set)
    for f in kept:
        subject_conditions[f['subject_key']].add(f['condition'])

    incomplete_subjects = set()
    for subj, conditions in subject_conditions.items():
        if conditions != {'EC', 'EO', 'TASK'}:
            missing = {'EC', 'EO', 'TASK'} - conditions
            incomplete_subjects.add(subj)

    # Now split kept into final_kept and newly_dropped
    final_kept = []
    for f in kept:
        if f['subject_key'] in incomplete_subjects:
            missing = {'EC', 'EO', 'TASK'} - subject_conditions[f['subject_key']]
            dropped.append({
                'filename':    f['filename'],
                'subject_key': f['subject_key'],
                'condition':   f['condition'],
                'reason':      f"Incomplete trio — subject missing: {', '.join(sorted(missing))}"
            })
        else:
            final_kept.append(f)

    return final_kept, dropped


# =========================================================================
# SECTION 4: COPY FILES INTO cleaned/
# =========================================================================
# For each kept file:
#   - If it is a TASK file with 22 channels → load, drop the two
#     auxiliary channels, save to cleaned/.
#   - Otherwise → just copy it straight.
#
# WHY we use mne to re-save (not just shutil.copy) for the 22ch files:
#   mne.io.Raw.drop_channels() modifies the in-memory signal, then
#   raw.export() writes a new EDF with only the channels we kept.
#   The signal data is untouched — only the two unused channels are removed.
# =========================================================================

def copy_and_fix(kept_files, raw_dir, cleaned_dir):
    """
    Copies verified files into cleaned_dir.
    Returns a list of dicts describing what was written (for clean_subjects.csv).
    """
    written = []

    for f in kept_files:
        subj = f['subject_key']
        cond = f['condition']

        # Create the subject folder if it does not exist
        subj_folder = os.path.join(cleaned_dir, subj)
        os.makedirs(subj_folder, exist_ok=True)

        # Destination filename: consistent format, no spaces, no ambiguity
        dest_filename = f"{subj}_{cond}.edf"
        dest_path     = os.path.join(subj_folder, dest_filename)

        # Read channel info to decide: copy or fix
        ch_names = read_channel_info(f['filepath'])
        needs_channel_fix = any(ch in ch_names for ch in CHANNELS_TO_DROP)

        if needs_channel_fix:
            # Load the full signal, drop the two channels, re-save
            raw = mne.io.read_raw_edf(f['filepath'], preload=True, verbose=False)
            # Only drop channels that actually exist in this file
            to_drop = [ch for ch in CHANNELS_TO_DROP if ch in raw.ch_names]
            raw.drop_channels(to_drop)
            raw.export(dest_path, overwrite=True, verbose=False)
            final_channels = len(raw.ch_names)
            action = f"copied + dropped {len(to_drop)} channels"
        else:
            # Already 20 channels. Just copy.
            shutil.copy2(f['filepath'], dest_path)
            final_channels = len(ch_names)
            action = "copied as-is"

        written.append({
            'subject_key':    subj,
            'group':          f['group'],
            'condition':      cond,
            'source_file':    f['filename'],
            'dest_file':      dest_filename,
            'n_channels':     final_channels,
            'action':         action
        })

        print(f"    {dest_filename:<25} {action}")

    return written


# =========================================================================
# SECTION 5: WRITE MANIFESTS
# =========================================================================
# Three CSV files. Each one is a self-contained record of a decision.
# A reviewer (or future you) can open any of these and understand
# exactly what happened and why, without reading the script.
# =========================================================================

def write_manifests(written, dropped, manifest_dir, clean_mdd_subjects):
    """
    written  → list of dicts from copy_and_fix (what we kept)
    dropped  → list of dicts from flag_all_files (what we removed + why)
    clean_mdd_subjects → the list of 26 clean MDD subject keys
    """

    # --- clean_subjects.csv ---
    # One row per file that made it into cleaned/.
    # Columns: subject_key, group, condition, source_file, dest_file, n_channels, action
    clean_df = pd.DataFrame(written)
    clean_df.to_csv(
        os.path.join(manifest_dir, "clean_subjects.csv"),
        index=False
    )
    print(f"\n  Wrote clean_subjects.csv  ({len(clean_df)} rows)")

    # --- dropped_subjects.csv ---
    # One row per file that was removed.
    # Columns: filename, subject_key, condition, reason
    dropped_df = pd.DataFrame(dropped)
    dropped_df.to_csv(
        os.path.join(manifest_dir, "dropped_subjects.csv"),
        index=False
    )
    print(f"  Wrote dropped_subjects.csv  ({len(dropped_df)} rows)")

    # --- balanced_seeds.csv ---
    # One row per seed. Each row records which 14 MDD subjects that seed selected.
    # This is what train.py will read to know which subjects go into each
    # Approach 2 run. Fixed and reproducible — same seed always picks the same 14.
    #
    # Why we store it here and not generate it in train.py:
    #   Reproducibility. The selection is made ONCE, saved to disk, and
    #   train.py just reads it. If you re-run train.py ten times, it uses
    #   the same selections every time. No surprises.
    seed_records = []
    for seed in BALANCED_SEEDS:
        rng = random.Random(seed)                          # isolated RNG — does not affect anything else
        selected = rng.sample(clean_mdd_subjects, N_MDD_TO_SELECT)
        seed_records.append({
            'seed':                seed,
            'mdd_subjects_selected': ';'.join(sorted(selected))   # semicolon-separated list
        })

    seeds_df = pd.DataFrame(seed_records)
    seeds_df.to_csv(
        os.path.join(manifest_dir, "balanced_seeds.csv"),
        index=False
    )
    print(f"  Wrote balanced_seeds.csv  ({len(seeds_df)} rows)\n")

    # Print the selections so you can see them
    print("  Balanced selections (Approach 2):")
    print(f"  {'Seed':<8} {'14 MDD subjects selected'}")
    print(f"  {'─' * 70}")
    for rec in seed_records:
        print(f"  {rec['seed']:<8} {rec['mdd_subjects_selected']}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("\n" + "=" * 70)
    print("  CLEAN DATASET")
    print("  Reading from: ", RAW_DIR)
    print("  Writing to:   ", CLEANED_DIR)
    print("=" * 70 + "\n")

    # Verify directories exist
    for d in [RAW_DIR, CLEANED_DIR, MANIFEST_DIR]:
        if not os.path.isdir(d):
            print(f"\n  ERROR: Directory does not exist: {d}")
            print("  Run the mkdir setup commands first.\n")
            return

    # --- Step 1+2+3: Parse, read headers, flag ---
    print("  Flagging files...")
    kept, dropped = flag_all_files(RAW_DIR)

    print(f"\n  Result: {len(kept)} files kept, {len(dropped)} files dropped\n")

    # --- Sanity check: should be exactly 120 kept (40 subjects × 3 conditions) ---
    if len(kept) != 120:
        print(f"  WARNING: Expected 120 kept files, got {len(kept)}.")
        print("  Check the CONFIG section at the top of this script.\n")

    # --- Step 4: Copy and fix channels ---
    print("  Copying to cleaned/ and fixing channels where needed...")
    written = copy_and_fix(kept, RAW_DIR, CLEANED_DIR)

    # --- Extract the clean MDD subject list for balanced_seeds ---
    # This is every MDD subject that made it through cleaning.
    clean_mdd_subjects = sorted(set(
        f['subject_key'] for f in kept if f['group'] == 'MDD'
    ))
    clean_h_subjects = sorted(set(
        f['subject_key'] for f in kept if f['group'] == 'H'
    ))

    # --- Step 5: Write manifests ---
    print("\n  Writing manifests...")
    write_manifests(written, dropped, MANIFEST_DIR, clean_mdd_subjects)

    # --- Final summary ---
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"\n  Clean subjects:  {len(clean_h_subjects)} H + {len(clean_mdd_subjects)} MDD = {len(clean_h_subjects)+len(clean_mdd_subjects)} total")
    print(f"  Clean files:     {len(written)} (should be {(len(clean_h_subjects)+len(clean_mdd_subjects))} × 3 = {(len(clean_h_subjects)+len(clean_mdd_subjects))*3})")
    print(f"  Dropped files:   {len(dropped)}")
    print(f"  All files are now 20 channels.\n")
    print(f"  Next step:  conda activate edf_reader")
    print(f"              python scripts/preprocess.py\n")


if __name__ == '__main__':
    main()
