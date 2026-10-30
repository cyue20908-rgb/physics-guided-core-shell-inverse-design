"""Canonical paths used by the organized reproducibility package."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPOSITORY_ROOT / "data"
CONFIG_DIR = REPOSITORY_ROOT / "configs"
CHECKPOINT_DIR = REPOSITORY_ROOT / "checkpoints"
RESULT_DIR = REPOSITORY_ROOT / "results"
REPRODUCIBILITY_DIR = REPOSITORY_ROOT / "reproducibility"

DATASET_PATH = DATA_DIR / "dataset" / "multilayer_mie_dataset.npz"
MATERIAL_DATA_DIR = DATA_DIR / "inputs" / "material_database"
REFERENCE_DATA_DIR = DATA_DIR / "inputs" / "reference_data"
SPLIT_DIR = DATA_DIR / "splits"
INVERSE_TRAINING_DATA = (
    DATA_DIR / "inverse_design_training" / "inverse_design_training_data.npz"
)

DIRECT_MODEL_DIR = CHECKPOINT_DIR / "forward" / "direct_spectrum_network"
WAVENUMBER_MODEL_DIR = (
    CHECKPOINT_DIR / "forward" / "wavenumber_conditioned_network"
)
PHYSICS_GUIDED_MODEL_DIR = (
    CHECKPOINT_DIR / "forward" / "physics_guided_fourier_network"
)
INVERSE_MODEL_DIR = (
    CHECKPOINT_DIR / "inverse" / "spectrum_conditioned_inverse_network"
)

FROZEN_ROBUST_RESULTS_DIR = (
    RESULT_DIR / "robust_design" / "full_factorial_robustness_evaluation"
)

REPRODUCED_RESULT_DIR = RESULT_DIR / "reproduced"
REPRODUCED_DATASET_BUILD_DIR = REPRODUCED_RESULT_DIR / "dataset_build"
REPRODUCED_INVERSE_RESULTS_DIR = (
    REPRODUCED_RESULT_DIR / "spectrum_inverse_design"
)
REPRODUCED_ROBUST_RESULTS_DIR = REPRODUCED_RESULT_DIR / "robust_design"
REPRODUCED_POSTHOC_RESULTS_DIR = (
    REPRODUCED_RESULT_DIR / "posthoc_optical_analysis"
)
