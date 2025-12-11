# 📊 Data Science Investigation Guide

## Overview

This guide helps you investigate two key aspects of radio map generation:

1. **Antenna Pattern Analysis**: How different antenna patterns affect radio map data (sparsity, values, tensor characteristics)
2. **Data Density Investigation**: How to get denser HR data with fewer missing points

---

## 🎯 Part 1: Antenna Pattern Analysis

### Goal
Understand how different antenna patterns (`iso`, `dipole`, `hw_dipole`, `tr38901`) affect the generated radio maps.

### Steps

#### 1. Generate data for each antenna pattern

```bash
# Run the generation script
./generate_pattern_data.sh
```

This script will:
- Generate 10 samples for each of the 4 antenna patterns
- Save data to `src/poc/pattern_analysis/pattern_<name>/`
- Use the Etoile scene by default (you can edit the script to change)

**Customize the script** (optional):
Edit `generate_pattern_data.sh` to change:
- `SCENE`: Choose from `etoile`, `san_francisco`, `munich`, `florence`
- `N_SAMPLES`: Number of samples per pattern (10 is good for quick analysis)
- `PATTERNS`: Add or remove antenna patterns

#### 2. Analyze the results

```bash
uv run python src/poc/analyze_antenna_patterns.py --data_dir src/poc/pattern_analysis
```

This will:
- Calculate sparsity metrics for each pattern
- Compare signal strength distributions
- Count valid pixels vs. missing values
- Generate comparison plots and tables
- Save visualization: `src/poc/pattern_analysis/antenna_pattern_comparison.png`

#### 3. Visualize specific samples

```bash
# View sample 0 (first sample) for all patterns side-by-side
uv run python src/poc/analyze_antenna_patterns.py \
    --data_dir src/poc/pattern_analysis \
    --visualize_sample 0
```

This shows the actual radio maps for visual comparison.

### What to Look For

**Sparsity Metrics:**
- **Sparsity %**: Percentage of pixels without valid signal data
- **Valid pixels**: How many pixels actually have signal values
- Lower sparsity = better coverage

**Signal Distribution:**
- **Mean, Std, Min, Max**: Statistical properties of signal values (in dB)
- **Percentiles (25th, 50th, 75th, 95th)**: Distribution shape
- Different patterns may have different signal concentration

**Key Questions:**
1. Which pattern gives the densest data (lowest sparsity)?
2. Do patterns affect signal value ranges significantly?
3. Are there visual differences in the radio maps?
4. Which pattern is best for your application?

---

## 🎯 Part 2: Data Density Investigation

### Goal
Figure out how to reduce sparsity and get more complete radio maps.

### Steps

#### 1. Run density experiments

```bash
uv run python src/poc/analyze_density.py
```

This will test three main factors:

**Test 1: samples_per_tx (number of rays)**
- Tests: 100K, 500K, 1M, 5M, 10M rays
- More rays = more coverage but slower generation
- Current default: 1M

**Test 2: max_depth (ray bounces)**
- Tests: 3, 5, 7, 10 bounces
- More bounces = rays can reach more areas through reflections
- Current default: 5

**Test 3: Number of transmitters**
- Tests: 1, 2, 4, 9 transmitters
- More transmitters = better coverage from multiple sources
- Current default: 1

#### 2. Review the results

The script will:
- Print sparsity statistics for each configuration
- Generate plots showing the relationship between each parameter and sparsity
- Save plots to `src/poc/density_analysis/`

**Generated plots:**
- `samples_per_tx_comparison.png`: Effect of ray count
- `max_depth_comparison.png`: Effect of ray bounces
- `n_tx_comparison.png`: Effect of multiple transmitters

### What to Look For

**Trade-offs:**
- **Generation time** vs. **Data quality**: More samples/depth = slower but denser
- **Diminishing returns**: At what point does increasing parameters not help much?

**Key Questions:**
1. What's the minimum `samples_per_tx` needed for acceptable coverage?
2. Does increasing `max_depth` beyond 5 help significantly?
3. How much does adding more transmitters improve coverage?
4. What's the best configuration for your use case?

### Recommendations Based on Results

After running the experiments, you can update your config:

**Edit `src/poc/configs/data/transmitter.yaml`:**
```yaml
n_tx: 4  # If multiple transmitters help
```

**Edit `src/poc/data_modules/generator.py`:**
In the `_generate_sample` method, change:
```python
rm_hr = self.rm_solver(
    self.scene,
    max_depth=7,           # If higher depth helps
    samples_per_tx=5*10**6,  # If more samples help
    cell_size=config.hr_cell_size,
    ...
)
```

---

## 📝 Advanced Customization

### Modify Antenna Pattern Generation

Edit `generate_pattern_data.sh`:
```bash
# Add custom parameters
uv run generate \
    scene_name="$SCENE" \
    transmitter.tx_array_pattern="$pattern" \
    transmitter.n_tx=2 \                    # Multiple transmitters
    transmitter.coverage_size=200.0 \       # Larger coverage area
    generator.n_samples=$N_SAMPLES \
    generator.dataset_path="$OUTPUT_DIR"
```

### Test Different Metrics

Edit `src/poc/analyze_density.py` to change the metric:
```python
generator = RadioMapDataGenerator(
    metric_type="rss",  # Try: "path_gain", "rss", or "sinr"
    ...
)
```

### Analyze Existing Data

If you already have generated data:
```bash
# Analyze existing dataset
uv run python src/poc/analyze_antenna_patterns.py \
    --data_dir src/poc/processed_data/train_val \
    --patterns iso
```

---

## 🔧 Troubleshooting

### Issue: "No samples found"
- Make sure you ran `generate_pattern_data.sh` first
- Check that the data directory exists and contains `.pt` files

### Issue: Script takes too long
- Reduce `N_SAMPLES` in `generate_pattern_data.sh`
- Use a simpler scene (e.g., `etoile` is faster than `san_francisco`)
- For density tests, reduce the range of parameters tested

### Issue: Out of memory
- Reduce `samples_per_tx` in density tests
- Reduce `hr_grid_size` in `transmitter.yaml` (lower resolution)
- Process fewer samples at a time

---

## 📊 Expected Output

### Antenna Pattern Analysis

**Console output:**
```
ANTENNA PATTERN COMPARISON - HIGH RESOLUTION MAPS
================================================================================
Metric                    iso            dipole      hw_dipole        tr38901
--------------------------------------------------------------------------------
Sparsity %               45.23            42.15           40.89           38.45
Valid pixels          143521           151234          154890          161023
Mean (dB)             -85.34           -83.21          -84.56          -82.78
...
```

**Plot:** Side-by-side comparison of sparsity, signal distribution, and coverage

### Density Analysis

**Console output:**
```
Testing samples_per_tx = 100,000
  Sparsity: 52.34%
  Valid pixels: 124,567 / 262,144

Testing samples_per_tx = 1,000,000
  Sparsity: 45.23%
  Valid pixels: 143,521 / 262,144
```

**Plots:** Line graphs showing how each parameter affects sparsity

---

## 🎓 Understanding the Results

### Sparsity
- **Low sparsity (< 30%)**: Good coverage, most pixels have values
- **Medium sparsity (30-50%)**: Acceptable, some gaps
- **High sparsity (> 50%)**: Many missing points, may need improvement

### When to Use Each Pattern

- **`iso` (isotropic)**: Uniform radiation in all directions, good baseline
- **`dipole`**: More realistic, directional characteristics
- **`hw_dipole` (half-wave dipole)**: Standard antenna model
- **`tr38901`**: 3GPP standard pattern for 5G simulations

### Optimization Strategy

1. **Start with default settings** and measure baseline sparsity
2. **Test antenna patterns** to see if changing pattern helps
3. **Increase samples_per_tx** if you have computational budget
4. **Add more transmitters** if coverage is the priority
5. **Increase max_depth** if scene has many obstacles

---

## 💡 Tips for Your Report

For your Data Science course, document:

1. **Hypothesis**: What did you expect to happen?
2. **Methodology**: What parameters did you test and why?
3. **Results**: Show the plots and statistics
4. **Analysis**: 
   - Which antenna pattern is best for your use case?
   - What's the optimal balance between quality and computation time?
   - How do the parameters interact?
5. **Conclusion**: Recommendations for future data generation

Good luck! 🚀
