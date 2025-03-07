"""
Global configuration constants for the project.

This file contains magic numbers and default parameters used across modules,
including ablation hooks and circuit finder functionality.
"""

# === Ablation Module Constants ===
DEFAULT_SCALE = 0.0  # Scale factor for ablation (0.0 indicates complete removal)
DEFAULT_SPARSITY_THRESHOLD = 0.1  # Threshold below which activations are zeroed
EPSILON = 1e-10  # Small constant to avoid division by zero in computations

INITIAL_PROGRESS = 40  # Initial progress value for candidate evaluation
FINAL_PROGRESS = 60    # Progress range for greedy iterative expansion
PRESELECTION_FRACTION = 0.2  # Fraction of candidates to preselect
DEFAULT_MAX_HEAD_LAYER_PAIRS = 10  # Default maximum number of head-layer pairs to ablate

# === Circuit Finder Constants ===
MAX_PATH_PATCHING_TIME = 60  # Maximum time (in seconds) for path patching experiments

# Head classification thresholds (as fractions of total layers)
HEAD_CLASSIFICATION_LATE_LAYER_THRESHOLD = 0.7  # Layers >= 70% considered "late"
HEAD_CLASSIFICATION_MID_LAYER_THRESHOLD = 0.4   # Layers < 40% considered "early"

# Probability thresholds for head role classification
NAME_MOVER_PROB_THRESHOLD = 0.3  # Minimum probability to consider a head as "Name Mover"
INDUCTION_PROB_LOW = 0.2         # Lower bound for induction probability threshold
INDUCTION_PROB_HIGH = 0.5        # Upper bound for induction probability threshold

# Progress stages for build_circuit_from_ablation (values for progress callback updates)
PROGRESS_STAGE_1 = 10
PROGRESS_STAGE_2 = 20
PROGRESS_STAGE_3 = 35
PROGRESS_STAGE_4 = 70
PROGRESS_STAGE_5 = 90

# Dummy values for utility methods (if real implementation is not provided)
ATTENTION_TO_TOKEN_DUMMY = 0.2
HEAD_INFLUENCE_DUMMY = 0.3
PATH_PATCHING_DUMMY_EFFECT = 0.1

# Progress update for run_path_patching: start and range values
PATH_PATCHING_PROGRESS_START = 40
PATH_PATCHING_PROGRESS_RANGE = 30

# Main app constants
NUM_COMBOS = 4                      # Number of layer-head combos per page in the heatmap
SUBPLOT_SIZE = 650                  # Base size for each subplot in the heatmap grid
HEATMAP_MARGIN = {"l": 80, "r": 40, "t": 60, "b": 150}  # Margins for the overall heatmap figure
TOP_K = 5                           # Number of top tokens to display in predictions/dropdowns
SLIDER_MIN = 0.0
SLIDER_MAX = 1.0
SLIDER_STEP = 0.01
SLIDER_MARKS = {i/10: f"{i/10}" for i in range(0, 11)}
