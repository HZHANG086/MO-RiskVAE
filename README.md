# MO-RiskVAE

Experimental research code for multi-omics survival risk modeling in multiple myeloma, curated from the author's research archive.

Related paper: [MO-RiskVAE: A Multi-Omics Variational Autoencoder for Survival Risk Modeling in Multiple Myeloma](https://arxiv.org/abs/2604.06267), arXiv:2604.06267 (2026), by Zixuan Chen, Heng Zhang, YuPeng Qin, WenPeng Xing, Qiang Wang, Da Wang, Changting Lin and Meng Han.

The archive contains experiments on latent regularization and continuous–discrete representations consistent with the subject of this paper. This release preserves available source; it has **not** been validated as an exact, complete reproduction of all paper results. No benchmark scores are claimed here.

## Contents

| Directory | Available material |
| --- | --- |
| `gumbel-softmax-vae/myevae/` | Continuous–discrete VAE research implementation and inherited package metadata |
| `HSIC-myeVAE/myevae/` | MMD/HSIC research implementation; standalone training entry file is missing from the supplied snapshot |
| `myevae/scripts/` | Experiment orchestration, survival evaluation, bootstrap intervals, paired tests and seed aggregation |
| `myevae/configs/` | Historical experiment configurations |
| `analysis/` | Latent-space, ablation, Kaplan–Meier and figure scripts |
| `snapshots/archived-src/` | Separate source snapshot recovered from the nested `src.zip`; retained without choosing it as the canonical model |

## Setup and use

Use Python 3.9 or later in an isolated environment. The original metadata lists PyTorch, scikit-survival, scikit-learn, matplotlib, dask and a specialized SHAP fork. Experiment utilities also require NumPy, pandas, PyYAML and SciPy; plotting scripts may additionally require lifelines and seaborn. Exact dependency resolution has not been tested.

For the Gumbel implementation, install only that variant in an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./gumbel-softmax-vae/myevae
python -m pip install -r myevae/requirements-experiments.txt
python -m myevae --help
```

Both variants expose the Python package `myevae`; use separate environments or explicit source paths to avoid importing the wrong implementation. The retained specialized SHAP dependency may need to be obtained from its original maintainer. Do not substitute an arbitrary SHAP version and assume equivalent behavior.

With authorized, locally prepared input data, the supplied Gumbel entry point accepts this command shape:

```bash
python -m myevae train os -n experiment -i /path/to/inputs -o /path/to/output
```

This is an interface example, not a validated reproduction recipe. Review `param_grid.py`, `params.py`, `type_prefixes.py` and the selected estimator before training. Preprocessing expects modality-prefixed feature columns, matched sample indices and endpoint-specific label fields. The supplied datasets use `PUBLIC_ID`, `oscdy`, `censos`, `pfscdy` and `censpfs`; confirm censoring/event conventions for each cohort before evaluating.

## Reproduction status and known limitations

- The `myevae/` experiment-management directory does not include its expected baseline `src/` model. A separate standalone MMD model tree is also absent.
- Batch launchers request `--seed`, `--beta-start`, `--beta-end`, `--beta-step` or Gumbel options not exposed by the supplied variant CLI. They are historical scripts and cannot be presented as a working end-to-end pipeline with this snapshot.
- The HSIC snapshot lacks `src/myevae/__init__.py` and `train.py`; its `__main__.py` therefore does not provide a complete standalone training CLI.
- Analysis scripts retain original `/work/zh/...` server paths. Change these to your local, authorized inputs and checkpoint locations before use. Configuration placeholders such as `feature_overlap: TBD` are not measured results.
- Some source files use `eval` to reconstruct serialized parameters and use PyTorch checkpoint loading. Only use trusted local parameter files and checkpoints.
- The package metadata references `data/ensembl_ref.csv`. This annotation file is omitted from this source-only release pending provenance and redistribution review. Obtain a suitable reference and verify compatibility before gene annotation/SHAP workflows.
- Checkpoints, datasets, per-sample predictions, latent vectors and generated figures are excluded. No training, inference, runtime import or numerical reproduction has been verified in this release. Static checks are documented in `RELEASE_AUDIT.md`.

## Data and privacy

No patient-level data is distributed. The original archive included linked clinical survival labels, expression/CNA features, patient identifiers and latent representations. These were excluded even when called “example” or associated with public cohorts. Obtain data from its authorized source under the applicable access and redistribution conditions. Do not commit local clinical datasets or generated patient-level predictions.

## Attribution and licensing

Heng Zhang states that he authored the experimental research code and extensions. The underlying MyeVAE framework is credited separately; its original copyright and MIT notice are preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license scope. This release makes no new license grant for experimental additions.

## Citation

```bibtex
@misc{chen2026moriskvae,
  title = {MO-RiskVAE: A Multi-Omics Variational Autoencoder for Survival Risk Modeling in Multiple Myeloma},
  author = {Zixuan Chen and Heng Zhang and YuPeng Qin and WenPeng Xing and Qiang Wang and Da Wang and Changting Lin and Meng Han},
  year = {2026},
  eprint = {2604.06267},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2604.06267}
}
```
