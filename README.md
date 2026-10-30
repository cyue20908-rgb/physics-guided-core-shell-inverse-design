# Physics-Guided Hybrid Inverse Design of Multilayer Core-Shell Nanoparticles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23805270.svg)](https://doi.org/10.5281/zenodo.23805270)

This repository is the reproducibility package for the study
*Physics-Guided Hybrid Inverse Design of Multilayer Core-Shell Nanoparticles for
Spectral Reconstruction and Robust Scattering*.

It contains the source code, fixed optical inputs, principal trained models,
compact numerical results, and independent-solver validation data used in the
study. Manuscript files, temporary training outputs, runtime logs, and large
proprietary COMSOL `.mph` models are not included.

## Authors

Xinyue Chang, Huanhuan An, Jing Zhang, Haiyu Guo, Yuchun Gou, and Junfei Fang  
School of Mechanical Engineering, Shaanxi University of Technology, Hanzhong,
China

Correspondence: Yuchun Gou (`ycgou@snut.edu.cn`) and Junfei Fang
(`jffang@snut.edu.cn`).

## Repository contents

See [`RELEASE.md`](RELEASE.md) for a brief description of the repository
folders.

The principal frozen assets are:

- `data/`: material optical constants, datasets, targets, and fixed splits;
- `checkpoints/`: trained forward and inverse model weights;
- `results/`: compact numerical evidence supporting the paper;
- `validation/`: COMSOL and independent-solver comparison data.

## Environment and installation

Python 3.11 is recommended. The recorded results used PyTorch 2.6.0 and
PyMieDiff 0.12.

```bash
python -m pip install -r reproducibility/environment/requirements.txt
python -m pip install -e . --no-deps
python -m pip install pytest==9.0.3
```

Use a PyTorch build appropriate for the local CPU or CUDA environment.

## Verification

From the repository root, run:

```bash
python -B -m pytest -p no:cacheprovider
python scripts/verification/run_environment_preflight.py --device auto --pymiediff-device cpu
python scripts/verification/compare_comsol_pymiediff.py
```

New verification outputs are written below `results/reproduced/`, which is
excluded from version control.

Some frozen JSON records retain historical absolute Windows paths as provenance
fields. These entries are not runtime settings; the packaged workflows resolve
their operational paths relative to the repository root.

Package integrity can be checked against
`reproducibility/manifests/package_checksums.csv`. A downloaded release archive
can be verified with:

```bash
python scripts/verification/verify_reproducibility_archive.py <archive.zip>
```

## Main workflow entry points

The principal scripts are organized as follows:

- `scripts/dataset/`: material preparation and learning-dataset generation;
- `scripts/forward_model_training/`: forward-model training and fair ablation;
- `scripts/spectrum_inverse_design/`: inverse model and spectrum reconstruction;
- `scripts/robust_design/`: robust broadband-scattering design;
- `scripts/posthoc_optical_analysis/`: field, multipole, and spectral analyses;
- `scripts/verification/`: package and independent-solver checks.

Full retraining and inverse-design searches are computationally expensive. The
repository includes the principal frozen checkpoints and compact result tables
so that the reported outputs can be inspected without retraining every model.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub and
Zenodo can read this file directly. The repository is available at
[github.com/cyue20908-rgb/physics-guided-core-shell-inverse-design](https://github.com/cyue20908-rgb/physics-guided-core-shell-inverse-design).
The fixed v1.0.2 reproducibility snapshot is archived at
[https://doi.org/10.5281/zenodo.23805270](https://doi.org/10.5281/zenodo.23805270).
Archived releases are grouped under the version-independent concept DOI
[https://doi.org/10.5281/zenodo.23803881](https://doi.org/10.5281/zenodo.23803881).

Suggested citation:

> Chang, X., An, H., Zhang, J., Guo, H., Gou, Y., & Fang, J. (2026).
> *Reproducibility Package for Physics-Guided Hybrid Inverse Design of
> Multilayer Core-Shell Nanoparticles* (v1.0.2). Zenodo.
> https://doi.org/10.5281/zenodo.23805270

## License

Software source code is licensed under the
[MIT License](LICENSE-CODE). Original research data, model weights, numerical
results, validation files, reproducibility records, and repository
documentation are licensed under
[CC BY 4.0](LICENSE-DATA). See [`LICENSE`](LICENSE) for scope and third-party
material exceptions.
