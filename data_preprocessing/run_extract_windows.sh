#!/bin/bash

# Usage: bash run_extract_windows.sh <hic_pkl_dir> <bedpe_file> <output_dir> <accession_list>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <hic_pkl_dir> <bedpe_file> <output_dir> <accession_list>"
    exit 1
fi

INPUT_PKL_DIR="$1"
BEDPE_FILE="$2"
OUTPUT_DIR="$3"
ACCESSION_LIST="$4"

RESOLUTION=1000
WINDOW_HEIGHT=192
WINDOW_WIDTH=960

# Here we assume the python script is in the same folder as this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/extract_hic_windows.py"

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "Extracting Windows"
echo "  Input Dir:      $INPUT_PKL_DIR"
echo "  Bedpe File:     $BEDPE_FILE"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Accession List: $ACCESSION_LIST"
echo "Date:       $(date)"
echo "=========================================================="

start=$SECONDS

mapfile -t ACCESSIONS < <(grep -v '^[[:space:]]*$' "$ACCESSION_LIST" | grep -v '^[[:space:]]*#')
TOTAL_ACC=${#ACCESSIONS[@]}

if (( TOTAL_ACC == 0 )); then
    echo "ERROR: No accessions found in $ACCESSION_LIST"
    exit 1
fi

i=0
for acc in "${ACCESSIONS[@]}"; do
    ((i++))
    acc=$(echo "$acc" | xargs)
    
    INPUT_PKL="${INPUT_PKL_DIR}/${acc}.pkl"
    OUTPUT_PKL="${OUTPUT_DIR}/${acc}.pkl"

    if [ ! -f "$INPUT_PKL" ]; then
        echo "Warning: Input pickle not found: $INPUT_PKL"
        continue
    fi

    if [ -f "$OUTPUT_PKL" ]; then
        echo "  [$i/$TOTAL_ACC] Skipping ${acc} (Output exists)"
        continue
    fi

    echo "  [$i/$TOTAL_ACC] Processing ${acc}..."

    python "${PYTHON_SCRIPT}" \
        --hic_pkl_path "${INPUT_PKL}" \
        --bedpe_path "${BEDPE_FILE}" \
        --resolution "${RESOLUTION}" \
        --output_pkl_path "${OUTPUT_PKL}" \
        --window_height "${WINDOW_HEIGHT}" \
        --window_width "${WINDOW_WIDTH}"

    if [ $? -ne 0 ]; then
        echo "Error processing $acc"
    fi

done

full_duration=$(( SECONDS - start ))
echo "=========================================================="
echo "Extraction completed in ${full_duration} seconds."
echo "Date: $(date)"
echo "=========================================================="