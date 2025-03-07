"""
Configuration constants for the ablation module.
This file contains magic numbers and default parameters used across the module.
"""

# Default scale factor for ablation (0.0 indicates complete removal)
DEFAULT_SCALE = 0.0

# Default threshold for sparsification: values below this are zeroed out
DEFAULT_SPARSITY_THRESHOLD = 0.1

# Small constant to avoid division by zero in KL divergence calculations
EPSILON = 1e-10

# Progress callback constants for find_best_ablation_combo function:
# INITIAL_PROGRESS is used for candidate evaluation progress,
# FINAL_PROGRESS is added during greedy iterative expansion.
INITIAL_PROGRESS = 40
FINAL_PROGRESS = 60

# Fraction of candidates to preselect (top 20%)
PRESELECTION_FRACTION = 0.2

# Default maximum number of head-layer pairs to ablate if not specified
DEFAULT_MAX_HEAD_LAYER_PAIRS = 10
