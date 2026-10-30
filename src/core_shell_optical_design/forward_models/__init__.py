"""Forward surrogate models."""

from .physics_guided_fourier_network import (
    CONDITIONING_MODES,
    COORDINATE_MODES,
    PGFFNet,
    PHYSICS_FEATURE_NAMES,
)
from .controlled_ablation import (
    AblationVariant,
    build_ablation_model,
    input_informed_parameter_count,
    parameter_count,
    variants_from_protocol,
)

__all__ = [
    "CONDITIONING_MODES",
    "COORDINATE_MODES",
    "PGFFNet",
    "PHYSICS_FEATURE_NAMES",
    "AblationVariant",
    "build_ablation_model",
    "input_informed_parameter_count",
    "parameter_count",
    "variants_from_protocol",
]
