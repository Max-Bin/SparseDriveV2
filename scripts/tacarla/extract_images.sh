#!/bin/bash
# Extract TaCarla sensor archives (tar.gz) for training.
# Only extracts rgb_camera images to save space.
# Usage: bash scripts/tacarla/extract_images.sh <tacarla_root> <output_dir>

set -e

TACARLA_ROOT="${1:?Usage: $0 <tacarla_root> <output_dir>}"
OUTPUT_DIR="${2:?Usage: $0 <tacarla_root> <output_dir>}"

mkdir -p "$OUTPUT_DIR"

for town in Town12 Town13; do
    SENSORS_DIR="$TACARLA_ROOT/data/TaCarla/${town}_sensors"
    if [ ! -d "$SENSORS_DIR" ]; then
        echo "Skipping $town: $SENSORS_DIR not found"
        continue
    fi

    count=$(ls "$SENSORS_DIR"/*.tar.gz 2>/dev/null | wc -l)
    echo "Extracting $count archives from $town..."

    i=0
    for tar_file in "$SENSORS_DIR"/*.tar.gz; do
        episode_name=$(basename "$tar_file" .tar.gz)
        dest="$OUTPUT_DIR/$episode_name"

        if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
            i=$((i+1))
            continue
        fi

        mkdir -p "$dest"
        tar -xzf "$tar_file" -C "$dest" --wildcards "*/rgb_camera/*.jpg" 2>/dev/null || true

        i=$((i+1))
        if [ $((i % 100)) -eq 0 ]; then
            echo "  [$town] Extracted $i / $count"
        fi
    done
    echo "  [$town] Done: $i archives"
done

echo "Extraction complete. Output: $OUTPUT_DIR"
