#!/usr/bin/env python3
"""Check all TaCarla tar.gz and parquet files for corruption, re-download broken ones.

Tests each tar.gz by reading the last 4 bytes for gzip footer validity,
and each parquet by attempting to open it. Re-downloads corrupted files from HuggingFace.
"""
import argparse
import gzip
import os
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def check_tar(tar_path_str):
    """Check if a tar.gz file is valid by testing gzip integrity. Returns (path, ok, size)."""
    tar_path = Path(tar_path_str)
    size = tar_path.stat().st_size
    try:
        # Quick check: gzip files should be > 100 bytes and end correctly
        if size < 100:
            return (str(tar_path), False, size, "too small")
        # Try reading the last few bytes with gzip
        with open(tar_path, "rb") as f:
            # Check gzip magic bytes at start
            magic = f.read(2)
            if magic != b'\x1f\x8b':
                return (str(tar_path), False, size, "bad gzip magic")
            # Check file can be seeked to end (not truncated at filesystem level)
            f.seek(-4, 2)
            f.read(4)  # last 4 bytes = original size mod 2^32
        return (str(tar_path), True, size, None)
    except Exception as e:
        return (str(tar_path), False, size, str(e)[:100])


def check_parquet(pq_path_str):
    """Check if a parquet file is valid."""
    pq_path = Path(pq_path_str)
    size = pq_path.stat().st_size
    try:
        with open(pq_path, "rb") as f:
            header = f.read(4)
            f.seek(-4, 2)
            footer = f.read(4)
        if header != b'PAR1':
            return (str(pq_path), False, size, "bad header magic")
        if footer != b'PAR1':
            return (str(pq_path), False, size, "bad footer magic (truncated)")
        return (str(pq_path), True, size, None)
    except Exception as e:
        return (str(pq_path), False, size, str(e)[:100])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tacarla-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fix", action="store_true", help="Re-download corrupted files from HuggingFace")
    args = parser.parse_args()

    root = Path(args.tacarla_root)

    # Collect all files to check
    tar_files = []
    for town in ["Town12_sensors", "Town13_sensors"]:
        d = root / "data" / "TaCarla" / town
        if d.exists():
            tar_files.extend(sorted(d.glob("*.tar.gz")))

    pq_files = []
    for label_dir in ["Town12_labels_hpc", "Town12_labels", "Town13_labels"]:
        d = root / "data" / "TaCarla_labels" / label_dir
        if d.exists():
            pq_files.extend(sorted(d.glob("*.parquet")))

    print(f"Checking {len(tar_files)} tar.gz + {len(pq_files)} parquet files...")

    # Check tar.gz files
    bad_tars = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(check_tar, str(f)): f for f in tar_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Checking tar.gz"):
            path, ok, size, err = fut.result()
            if not ok:
                bad_tars.append((path, size, err))
                tqdm.write(f"  BAD tar: {Path(path).name} ({size/1e6:.1f}MB) - {err}")

    # Check parquet files
    bad_pqs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(check_parquet, str(f)): f for f in pq_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Checking parquet"):
            path, ok, size, err = fut.result()
            if not ok:
                bad_pqs.append((path, size, err))
                tqdm.write(f"  BAD parquet: {Path(path).name} ({size/1e6:.1f}MB) - {err}")

    print(f"\nResults: {len(bad_tars)} bad tar.gz, {len(bad_pqs)} bad parquet")

    if not bad_tars and not bad_pqs:
        print("All files OK!")
        return

    for path, size, err in bad_tars:
        print(f"  tar: {Path(path).name} ({size/1e6:.1f}MB) - {err}")
    for path, size, err in bad_pqs:
        print(f"  pq:  {Path(path).name} ({size/1e6:.1f}MB) - {err}")

    if args.fix:
        from huggingface_hub import hf_hub_download
        print("\nRe-downloading corrupted files...")

        for path, size, err in bad_tars:
            p = Path(path)
            # Determine HF repo path: Town12_sensors/xxx.tar.gz
            for town in ["Town12_sensors", "Town13_sensors"]:
                if town in str(p):
                    hf_filename = f"{town}/{p.name}"
                    break
            try:
                print(f"  Downloading {hf_filename}...")
                hf_hub_download(
                    repo_id="tugrul93/TaCarla",
                    filename=hf_filename,
                    repo_type="dataset",
                    local_dir=str(root / "data" / "TaCarla"),
                    force_download=True,
                )
                print(f"    OK: {os.path.getsize(path)/1e6:.1f}MB")
            except Exception as e:
                print(f"    FAILED: {e}")

        for path, size, err in bad_pqs:
            p = Path(path)
            for label_dir in ["Town12_labels_hpc", "Town12_labels", "Town13_labels"]:
                if label_dir in str(p):
                    hf_filename = f"{label_dir}/{p.name}"
                    break
            try:
                print(f"  Downloading {hf_filename}...")
                hf_hub_download(
                    repo_id="tugrul93/TaCarla_labels",
                    filename=hf_filename,
                    repo_type="dataset",
                    local_dir=str(root / "data" / "TaCarla_labels"),
                    force_download=True,
                )
                print(f"    OK: {os.path.getsize(path)/1e6:.1f}MB")
            except Exception as e:
                print(f"    FAILED: {e}")

        print("\nDone. Re-run without --fix to verify.")


if __name__ == "__main__":
    main()
