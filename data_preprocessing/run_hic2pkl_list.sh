#!/bin/bash

# Usage: bash run_hic2pkl_list.sh /path/to/hic/files /path/to/output/pkls

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_hic_dir> <output_pkl_dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

RESOLUTION=1000
NORM=0   # 0: None
MODE=2   # 2: Scipy sparse (cis-only)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/hic2array.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Could not find hic2array.py at $PYTHON_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting HiC Extraction"
echo "Input Dir:  $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Resolution: $RESOLUTION"
echo "Date:       $(date)"
echo "=========================================="

start=$SECONDS

# Check if directory contains files to avoid loop errors
shopt -s nullglob
files=("$INPUT_DIR"/*.hic)

if [ ${#files[@]} -eq 0 ]; then
    echo "No .hic files found in $INPUT_DIR"
    exit 0
fi

for input_hic in "${files[@]}"; do
    filename=$(basename -- "$input_hic")
    output_pkl="${OUTPUT_DIR}/${filename%.hic}.pkl"

    # Check if output already exists
    if [[ -f "$output_pkl" ]]; then
        echo "[SKIP] ${filename}: Output already exists."
        continue
    fi

    echo "[PROCESSING] ${filename}..."
    
    python "$PYTHON_SCRIPT" "$input_hic" "$output_pkl" "$RESOLUTION" "$NORM" "$MODE"
    
    # Check return code
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to process $filename"
    fi
done

full_duration=$(( SECONDS - start ))
echo "=========================================="
echo "All jobs completed in ${full_duration} seconds"
echo "Date: $(date)"
echo "=========================================="
