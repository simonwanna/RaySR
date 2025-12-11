# 🚀 Quick Start - Data Science Investigation

## TL;DR - What to Run

### Part 1: Antenna Pattern Analysis

```bash
# 1. Generate data for different antenna patterns (~5-10 min)
./generate_pattern_data.sh

# 2. Analyze the results
uv run python src/poc/analyze_antenna_patterns.py --data_dir src/poc/pattern_analysis

# 3. Visualize specific samples
uv run python src/poc/analyze_antenna_patterns.py --data_dir src/poc/pattern_analysis --visualize_sample 0
```

**Output:** 
- Comparison table of sparsity and signal statistics
- Plots in `src/poc/pattern_analysis/antenna_pattern_comparison.png`

---

### Part 2: Data Density Investigation

```bash
# Run all density experiments (~10-15 min)
uv run python src/poc/analyze_density.py
```

**Output:**
- Sparsity statistics for different configurations
- Plots in `src/poc/density_analysis/`:
  - `samples_per_tx_comparison.png`
  - `max_depth_comparison.png`
  - `n_tx_comparison.png`

---

## Key Files Created

| File | Purpose |
|------|---------|
| `generate_pattern_data.sh` | Generate data with different antenna patterns |
| `src/poc/analyze_antenna_patterns.py` | Analyze antenna pattern effects |
| `src/poc/analyze_density.py` | Test configurations for denser data |
| `DATA_SCIENCE_INVESTIGATION_GUIDE.md` | Full documentation |

---

## Customization

### Change the Scene

Edit `generate_pattern_data.sh`:
```bash
SCENE="san_francisco"  # or: munich, florence, etoile
```

### Change Number of Samples

Edit `generate_pattern_data.sh`:
```bash
N_SAMPLES=20  # More samples = better statistics but slower
```

### Test Different Parameters

Edit `src/poc/analyze_density.py` to change the tested ranges:
```python
# Line ~52: Change sample counts
sample_counts = [10**5, 10**6, 10**7]  # Customize this

# Line ~130: Change depths
depths = [3, 5, 7, 10]  # Customize this

# Line ~204: Change transmitter counts
n_tx_values = [1, 2, 4, 9]  # Customize this
```

---

## What You'll Learn

### Antenna Patterns
- Which pattern gives densest coverage?
- How do patterns affect signal distribution?
- Visual differences between patterns

### Data Density
- Optimal number of rays (`samples_per_tx`)
- Effect of ray bounces (`max_depth`)
- Impact of multiple transmitters (`n_tx`)

---

## Expected Runtime

| Task | Time |
|------|------|
| Generate pattern data (4 patterns × 10 samples) | ~5-10 min |
| Analyze antenna patterns | ~30 sec |
| Density experiments (all 3 tests) | ~10-15 min |

**Total: ~15-25 minutes**

---

## Key Metrics to Report

1. **Sparsity %**: Lower is better (more coverage)
2. **Valid pixels**: Higher is better (more data points)
3. **Signal mean/std**: Understand data distribution
4. **Generation time**: Trade-off with quality

---

## Need Help?

See `DATA_SCIENCE_INVESTIGATION_GUIDE.md` for:
- Detailed explanations
- Troubleshooting tips
- Advanced customization
- Report writing guidance
