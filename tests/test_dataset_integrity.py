from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset" / "multilayer_mie_dataset.npz"
SPLIT_DIR = ROOT / "data" / "splits"


def test_dataset_structure_geometry_and_material_registry() -> None:
    with np.load(DATASET, allow_pickle=True) as archive:
        required = {
            "sample_id", "radii", "material_indices", "num_layers",
            "wavelengths", "Q_ext", "Q_sca", "Q_abs", "material_names",
        }
        assert required.issubset(archive.files)
        sample_id = np.asarray(archive["sample_id"])
        radii = np.asarray(archive["radii"], dtype=np.float64)
        materials = np.asarray(archive["material_indices"], dtype=np.int64)
        layers = np.asarray(archive["num_layers"], dtype=np.int64)
        wavelengths = np.asarray(archive["wavelengths"], dtype=np.float64)
        names = np.asarray(archive["material_names"]).astype(str)

    assert sample_id.shape == (30000,)
    assert np.array_equal(sample_id, np.arange(30000))
    assert radii.shape == (30000, 4)
    assert materials.shape == (30000, 4)
    assert np.all(layers == 4)
    assert wavelengths.shape == (101,)
    assert np.isclose(wavelengths[0], 400.0)
    assert np.isclose(wavelengths[-1], 1000.0)
    assert np.all(np.diff(wavelengths) > 0.0)

    geometry = np.column_stack((radii[:, 0], np.diff(radii, axis=1)))
    assert np.all((geometry[:, 0] >= 120.0) & (geometry[:, 0] <= 220.0))
    assert np.all((geometry[:, 1:] >= 20.0) & (geometry[:, 1:] <= 80.0))
    assert np.all(radii[:, -1] <= 300.0)
    assert np.all(np.diff(radii, axis=1) > 0.0)

    assert len(names) == 6
    air_indices = np.flatnonzero(np.char.lower(names) == "air")
    assert air_indices.size == 1
    air = int(air_indices[0])
    assert np.all(materials >= 0)
    assert np.all(materials < len(names))
    assert np.all(materials[:, 1:] != air)
    assert np.all(materials[:, 1:] != materials[:, :-1])
    unique_sequences, counts = np.unique(materials, axis=0, return_counts=True)
    assert unique_sequences.shape == (400, 4)
    assert np.all(counts == 75)


def test_fixed_splits_are_disjoint_and_exhaustive() -> None:
    train = np.load(SPLIT_DIR / "idx_train.npy", allow_pickle=False)
    validation = np.load(SPLIT_DIR / "idx_val.npy", allow_pickle=False)
    test = np.load(SPLIT_DIR / "idx_test.npy", allow_pickle=False)
    assert train.shape == (21000,)
    assert validation.shape == (4500,)
    assert test.shape == (4500,)
    assert np.intersect1d(train, validation).size == 0
    assert np.intersect1d(train, test).size == 0
    assert np.intersect1d(validation, test).size == 0
    combined = np.concatenate((train, validation, test))
    assert np.array_equal(np.sort(combined), np.arange(30000))


def test_mie_energy_closure() -> None:
    with np.load(DATASET, allow_pickle=False) as archive:
        qext = np.asarray(archive["Q_ext"], dtype=np.float64)
        qsca = np.asarray(archive["Q_sca"], dtype=np.float64)
        qabs = np.asarray(archive["Q_abs"], dtype=np.float64)
    assert np.isfinite(qext).all()
    assert np.isfinite(qsca).all()
    assert np.isfinite(qabs).all()
    assert min(float(qext.min()), float(qsca.min()), float(qabs.min())) >= -1.0e-8
    residual = np.abs(qext - qsca - qabs)
    assert float(residual.max()) <= 1.0e-6
