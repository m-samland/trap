# TRAP – Temporal Reference Analysis for Planets

[![Python](https://img.shields.io/badge/Python-3.11%2C%203.12%2C%203.13-brightgreen.svg)](https://github.com/m-samland/trap)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/m-samland/trap/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F201937308-blue)](https://doi.org/10.1051/0004-6361/201937308)

**TRAP** is a novel algorithm for the detection of exoplanets in high-contrast direct imaging data. Unlike traditional image-based approaches, TRAP models and removes stellar contamination using causal **temporal regression**, offering improved sensitivity to exoplanet signals at small angular separations from the host star.

> 📄 For a detailed methodology and scientific background, see:  
> [Samland et al. 2021, A&A, 646, A24](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..24S/abstract)

---

## Key Features

- **Temporal Systematics Modeling**: Models time-dependent stellar residuals using causal regression for improved contrast at small inner working angles.
- **Automatic Detection and Characterization**: Automatically detect point sources, measure their astrometry per wavelength channel, and extract their spectra.
- **Spectral Template Matching**: Works with external spectral templates (using [`species`](https://github.com/tomasstolker/species)) for improved detection.
- **Integral-Field and Dual-Band Data**: Ready-made configurations for IFS and IRDIS-style data, including optional enrichment of the temporal regressor pool with other wavelength slices.
- **Edge-of-Field Reduction**: An optional detector footprint mask lets the search run out to the edge of irregularly shaped fields of view instead of stopping at a fixed radius.
- **Dataclass Configuration**: Reductions are configured with frozen, introspectable dataclasses (`TrapConfig` / `TrapReductionConfig`) — no hidden global state.
- **Parallel Processing**: Efficient parallelism via [joblib](https://joblib.readthedocs.io/)/loky with a memmap-backed shared-array store, scalable from laptops to clusters.
- **Automated Visualization**: Generates diagnostic plots, spectra, and contrast curves for each reduction pipeline run.

---

## Installation

TRAP requires **Python 3.11, 3.12, or 3.13**. It can be installed directly from GitHub:

```bash
pip install git+https://github.com/m-samland/trap
```

> ℹ️ TRAP parallelizes with `joblib`/loky and a memmap-backed shared-array store, so there is no cluster runtime to install separately. Large input arrays are written once to a scratch directory (`/dev/shm` when available, otherwise the system temp directory) and memory-mapped read-only by the worker processes; set `TrapReductionConfig.scratch_dir` if your machine needs a different location. To scale beyond one node, submit scheduler job arrays over wavelengths or epochs.

---

## Quick Start

A [Jupyter notebook](examples/tutorial.ipynb) and [example dataset](examples/test_data/) based on **VLT/SPHERE** observations are provided. They demonstrate the full workflow: loading data, performing temporal regression, generating detection maps, and extracting companion spectra.

Reductions are configured with dataclasses rather than a CLI:

```python
from trap.parameters import trap_config_for_ifs

config = trap_config_for_ifs()
reduction_config = config.reduction.merge(result_folder="./results")
```

`TrapReductionConfig` is frozen — derive variants with `merge(**kwargs)` instead of assigning to attributes.

---

### 🔗 Related Project: `spherical`

Looking for a full data reduction pipeline for **VLT/SPHERE-IFS** data?

Check out [**spherical**](https://github.com/m-samland/spherical) — a companion package to TRAP that provides an **end-to-end workflow** from data discovery and calibration to post-processing.

- 🔍 Automates retrieval of SPHERE-IFS raw and calibration data from the ESO archive
- ⚙️ Uses a Python-based, open-source pipeline based on the **CHARIS data reduction pipeline**  
  ([Samland et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract)) — a high-performance alternative to the official SPHERE-DRH
- 📊 Seamlessly integrates with **TRAP** for temporal analysis and exoplanet detection
- 🌈 Automatically applies TRAP’s **template matching** to collapse detection maps using spectral models (e.g., L- and T-type planets)

> **TRAP is fully supported as the post-processing backend in `spherical`**, making it easy to go from raw SPHERE data to scientifically validated planet detections.

👉 Learn more at: [https://github.com/m-samland/spherical](https://github.com/m-samland/spherical)

---

## Contributing

We warmly welcome contributions to **TRAP**! Here's how to get started:

### Setup for Developers

Clone the repository and install it locally for development with tests:

```bash
git clone https://github.com/m-samland/trap.git
cd trap
pip install -e ".[test]"
```

Alternatively, a [pixi](https://pixi.sh) environment is provided:

```bash
pixi install -e dev
pixi shell -e dev
```

### Contributing Guidelines

- Please use **feature branches or forks** for developing new features or bug fixes.
- **Issue and Pull Request templates** are provided—please use them.
- Always run the code linting tool (`ruff`) before submitting a Pull Request:

```bash
ruff check .
```

If you’re unsure where to start, check out the [good first issues](https://github.com/m-samland/trap/labels/good%20first%20issue) or open a discussion.

---

## 📖 Citing TRAP

If you use **TRAP** in your research, please cite:

> **Matthias Samland et al. (2021)** – *A temporal systematics model for improved direct detection of exoplanets at small angular separations*  
> [A&A, Vol. 646, A24](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..24S)  
> DOI: [10.1051/0004-6361/201937308](https://doi.org/10.1051/0004-6361/201937308)

```bibtex
@ARTICLE{2021A&A...646A..24S,
  author       = {{Samland}, M. and {Bouwman}, J. and {Hogg}, D.~W. and {Brandner}, W. and {Henning}, T. and {Janson}, M.},
  title        = "{TRAP: a temporal systematics model for improved direct detection of exoplanets at small angular separations}",
  journal      = {\aap},
  year         = 2021,
  month        = feb,
  volume       = {646},
  pages        = {A24},
  doi          = {10.1051/0004-6361/201937308},
  archivePrefix = {arXiv},
  eprint       = {2011.12311},
  primaryClass = {astro-ph.EP},
  adsurl       = {https://ui.adsabs.harvard.edu/abs/2021A&A...646A..24S},
  adsnote      = {Provided by the SAO/NASA Astrophysics Data System}
}
```

For other citation formats, visit the [ADS entry](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..24S/exportcitation).

---

## Versioning

The peer-reviewed publication describes release version **v1.0.0** of TRAP.  
Subsequent changes and feature additions are documented in the [CHANGELOG](CHANGELOG.md).

**v2.0.0 contains breaking changes.** The legacy `Reduction_parameters` object and its bridge
methods were removed, `include_noise` was renamed to `estimate_noise_from_data`, and Ray was
replaced by `joblib`/loky. See the [CHANGELOG](CHANGELOG.md) for the migration details.

---

## License

This project is licensed under the [MIT License](https://github.com/m-samland/trap/blob/main/LICENSE).

---

## Author

**Matthias Samland**  
GitHub: [@m-samland](https://github.com/m-samland)
