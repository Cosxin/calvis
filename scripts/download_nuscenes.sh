#!/bin/bash
# Download nuScenes mini dataset with resume support
# Usage: bash scripts/download_nuscenes.sh [data_dir]
set -e

DATA_DIR="${1:-/Users/enron/Documents/calvis/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
FILE="v1.0-mini.tgz"

echo "=== nuScenes Mini Download Script ==="
echo "Target: $DATA_DIR/$FILE"

# Check if already extracted
if [ -d "$DATA_DIR/v1.0-mini" ] && [ -d "$DATA_DIR/samples" ]; then
    echo "✓ Already extracted. Skipping."
    exit 0
fi

# Resume-capable download
if [ -f "$FILE" ]; then
    CURRENT_SIZE=$(stat -f%z "$FILE" 2>/dev/null || echo 0)
    echo "Resuming from $(echo "scale=1; $CURRENT_SIZE/1048576" | bc)MB..."
else
    echo "Starting fresh download..."
fi

MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
    echo "--- Attempt $i/$MAX_RETRIES ---"
    if curl -C - -L -# --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 0 -o "$FILE" "$URL"; then
        echo "Download complete!"
        break
    else
        EC=$?
        if [ $EC -eq 33 ]; then
            echo "Server doesn't support resume, restarting..."
            rm -f "$FILE"
        elif [ $i -eq $MAX_RETRIES ]; then
            echo "Failed after $MAX_RETRIES attempts."
            exit 1
        else
            echo "Interrupted. Retrying in 10s..."
            sleep 10
        fi
    fi
done

FINAL_SIZE=$(stat -f%z "$FILE" 2>/dev/null || echo 0)
echo "File size: $(echo "scale=1; $FINAL_SIZE/1048576" | bc)MB"

echo "=== Extracting ==="
tar xzf "$FILE" -C "$DATA_DIR"

if [ -d "$DATA_DIR/v1.0-mini" ]; then
    echo "✓ Done! Data at: $DATA_DIR"
else
    echo "✗ Extraction failed"
    exit 1
fi
