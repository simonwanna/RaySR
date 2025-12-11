#!/bin/bash

# Script to generate radio map data with different antenna patterns
# This helps investigate how antenna patterns affect the data

# Configuration
SCENE="etoile"  # Change to: san_francisco, munich, florence, or etoile
N_SAMPLES=10    # Number of samples per pattern
OUTPUT_BASE="src/poc/pattern_analysis"

# Antenna patterns to test
PATTERNS=("iso" "dipole" "hw_dipole" "tr38901")

echo "Starting antenna pattern data generation..."
echo "Scene: $SCENE"
echo "Samples per pattern: $N_SAMPLES"
echo "Patterns: ${PATTERNS[@]}"
echo ""

for pattern in "${PATTERNS[@]}"; do
    echo "================================================"
    echo "Generating data for pattern: $pattern"
    echo "================================================"
    
    OUTPUT_DIR="${OUTPUT_BASE}/pattern_${pattern}"
    
    uv run generate \
        scene_name="$SCENE" \
        transmitter.tx_array_pattern="$pattern" \
        generator.n_samples=$N_SAMPLES \
        generator.dataset_path="$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated data for $pattern"
    else
        echo "✗ Failed to generate data for $pattern"
    fi
    echo ""
done

echo "================================================"
echo "Data generation complete!"
echo "Results saved in: $OUTPUT_BASE"
echo ""
echo "To analyze the results, run:"
echo "uv run python src/poc/analyze_antenna_patterns.py --data_dir $OUTPUT_BASE"
echo "================================================"
