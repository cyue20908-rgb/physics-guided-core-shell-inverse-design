from __future__ import annotations

import json
from pathlib import Path

import torch

from core_shell_optical_design.forward_models.controlled_ablation import (
    build_ablation_model,
    input_informed_parameter_count,
    parameter_count,
    variants_from_protocol,
)
from core_shell_optical_design.forward_models.physics_guided_fourier_network import (
    PGFFNet,
    PHYSICS_FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "forward" / "controlled_ablation" / "protocol.json"
)


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8-sig"))


def dummy_inputs() -> dict[str, torch.Tensor]:
    wavelengths = torch.linspace(400.0, 1000.0, 7)
    material_n = torch.linspace(1.0, 2.5, 6).view(-1, 1).expand(-1, 7).clone()
    material_k = torch.zeros_like(material_n)
    return {
        "geometry_mean": torch.tensor([170.0, 50.0, 50.0, 50.0]),
        "geometry_std": torch.tensor([29.0, 17.0, 17.0, 17.0]),
        "wavelengths_nm": wavelengths,
        "material_n": material_n,
        "material_k": material_k,
        "physics_mean": torch.zeros(len(PHYSICS_FEATURE_NAMES)),
        "physics_std": torch.ones(len(PHYSICS_FEATURE_NAMES)),
    }


def small_model_config() -> dict:
    return {
        "hidden_dim": 16,
        "fourier_frequencies": 4,
        "fourier_scale": 3.0,
        "dropout": 0.0,
        "n_structure_blocks": 1,
        "n_wavelength_blocks": 1,
        "n_physics_blocks": 1,
        "n_fusion_blocks": 1,
    }


def test_protocol_primary_variants_form_a_component_ladder() -> None:
    protocol = load_protocol()
    variants = variants_from_protocol(protocol)
    assert protocol["primary_variants"] == [
        "raw_k0",
        "fourier_only",
        "physics_only",
        "fourier_physics_additive",
        "full_pgff",
    ]
    assert variants["raw_k0"].coordinate_mode == "raw_k0"
    assert variants["fourier_only"].coordinate_mode == "fourier"
    assert variants["physics_only"].use_physics_features
    assert variants["fourier_physics_additive"].conditioning_mode == "additive"
    assert variants["full_pgff"].conditioning_mode == "film"


def test_all_variants_have_identical_registered_parameter_count() -> None:
    protocol = load_protocol()
    variants = variants_from_protocol(protocol)
    counts = []
    informed_counts = []
    for name in protocol["variants"]:
        torch.manual_seed(123)
        model = build_ablation_model(
            variant=variants[name],
            model_config=protocol["model"],
            fourier_seed=2026,
            **dummy_inputs(),
        )
        counts.append(parameter_count(model))
        informed_counts.append(input_informed_parameter_count(model, variants[name]))
    assert len(set(counts)) == 1
    assert min(informed_counts) / counts[0] >= 0.98


def test_default_pgff_is_the_explicit_full_variant() -> None:
    inputs = dummy_inputs()
    config = small_model_config()
    common = {
        **inputs,
        "hidden_dim": config["hidden_dim"],
        "n_fourier_frequencies": config["fourier_frequencies"],
        "fourier_scale": config["fourier_scale"],
        "dropout": config["dropout"],
        "n_structure_blocks": config["n_structure_blocks"],
        "n_wavelength_blocks": config["n_wavelength_blocks"],
        "n_physics_blocks": config["n_physics_blocks"],
        "n_fusion_blocks": config["n_fusion_blocks"],
    }
    torch.manual_seed(123)
    default_model = PGFFNet(**common)
    torch.manual_seed(123)
    explicit_model = PGFFNet(
        **common,
        coordinate_mode="fourier",
        use_physics_features=True,
        conditioning_mode="film",
    )
    geometry = torch.tensor([[150.0, 40.0, 50.0, 60.0]])
    material = torch.tensor([[1, 2, 3, 4]])
    default_output = default_model(geometry, material)
    explicit_output = explicit_model(geometry, material)
    for key in ("q_sca", "q_abs", "q_ext"):
        assert torch.equal(default_output[key], explicit_output[key])


def test_raw_k0_variant_does_not_use_the_random_fourier_basis() -> None:
    protocol = load_protocol()
    variant = variants_from_protocol(protocol)["raw_k0"]
    torch.manual_seed(123)
    model_a = build_ablation_model(
        variant=variant,
        model_config=small_model_config(),
        fourier_seed=1,
        **dummy_inputs(),
    )
    torch.manual_seed(123)
    model_b = build_ablation_model(
        variant=variant,
        model_config=small_model_config(),
        fourier_seed=999,
        **dummy_inputs(),
    )
    geometry = torch.tensor([[150.0, 40.0, 50.0, 60.0]])
    material = torch.tensor([[1, 2, 3, 4]])
    output_a = model_a(geometry, material)["q_sca"]
    output_b = model_b(geometry, material)["q_sca"]
    assert torch.equal(output_a, output_b)


def test_no_physics_variant_is_invariant_to_material_optical_tables() -> None:
    protocol = load_protocol()
    variant = variants_from_protocol(protocol)["fourier_only"]
    inputs_a = dummy_inputs()
    inputs_b = dummy_inputs()
    inputs_b["material_n"] = inputs_b["material_n"] + 0.4
    inputs_b["material_k"] = inputs_b["material_k"] + 0.02
    torch.manual_seed(123)
    model_a = build_ablation_model(
        variant=variant,
        model_config=small_model_config(),
        fourier_seed=2026,
        **inputs_a,
    )
    torch.manual_seed(123)
    model_b = build_ablation_model(
        variant=variant,
        model_config=small_model_config(),
        fourier_seed=2026,
        **inputs_b,
    )
    geometry = torch.tensor([[150.0, 40.0, 50.0, 60.0]])
    material = torch.tensor([[1, 2, 3, 4]])
    output_a = model_a(geometry, material)["q_sca"]
    output_b = model_b(geometry, material)["q_sca"]
    assert torch.equal(output_a, output_b)
