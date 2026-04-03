#!/usr/bin/env python3
"""Fast extraction of rgb_camera images from TaCarla tar.gz archives.

Reads tar.gz into memory, filters rgb_camera jpgs, writes out in batch.
Avoids NFS write-lock contention that stalls parallel tar processes.

Usage:
    python scripts/tacarla/fast_extract.py \
        --tar-list /tmp/tars_to_extract_full.txt \
        --output-dir /path/to/extracted_sensors \
        --workers 8
"""
import argparse
import io
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def extract_one(args):
    tar_path_str, output_dir = args
    tar_path = Path(tar_path_str.strip())
    episode = tar_path.name.replace(".tar.gz", "")
    dest = Path(output_dir) / episode
    marker = dest / ".rgb_extracted"

    if marker.exists():
        return (episode, 0, None)

    try:
        # Read entire tar.gz into memory (avoids NFS read contention during extraction)
        with open(tar_path, "rb") as f:
            data = f.read()

        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if "/rgb_camera/" in member.name and member.name.endswith(".jpg") and member.isfile():
                    # Write directly to destination
                    out_path = dest / member.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    count += 1

        marker.touch()
        return (episode, count, None)
    except Exception as e:
        return (episode, 0, str(e)[:200])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.tar_list) as f:
        all_tars = [line.strip() for line in f if line.strip()]

    # Filter already done
    output_dir = Path(args.output_dir)
    to_do = []
    done = 0
    for t in all_tars:
        ep = Path(t).name.replace(".tar.gz", "")
        if (output_dir / ep / ".rgb_extracted").exists():
            done += 1
        else:
            to_do.append((t, str(output_dir)))

    print(f"Total: {len(all_tars)}, done: {done}, to extract: {len(to_do)}, workers: {args.workers}")

    if not to_do:
        print("All done!")
        return

    total_files = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_one, t): t[0] for t in to_do}
        with tqdm(total=len(to_do), desc="Extracting", unit="ep") as pbar:
            for fut in as_completed(futs):
                ep, count, err = fut.result()
                total_files += count
                if err:
                    errors += 1
                    tqdm.write(f"  ERR {ep}: {err}")
                pbar.update(1)
                pbar.set_postfix(files=total_files, err=errors)

    print(f"\nDone: {total_files} images extracted, {errors} errors")


if __name__ == "__main__":
    main()
