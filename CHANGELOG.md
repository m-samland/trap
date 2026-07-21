# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Edge-of-FoV reduction** – `run_complete_reduction` accepts a
  `valid_pixel_mask` (2D `H×W` or 3D `n_wave×H×W`) describing which detector
  pixels carry real data. Positions outside the footprint are excluded from
  scheduling, per-position `signal_mask` / `reduction_mask` / single- and
  multi-wavelength regressor pools are intersected with the footprint, and
  positions with fewer than `reduction_mask_min_pixels` surviving reduction
  pixels return `NaN` in the detection map instead of raising. Two new knobs
  on `TrapReductionConfig`: `search_region_outer_bound: Optional[int] = 85`
  (`None` triggers footprint-derived auto-derivation) and
  `reduction_mask_min_pixels: int = 30`. Callers that don't pass a footprint
  see identical behavior.
- **Multi-wavelength regressors for IFS (WP2)** – the temporal regressor pool
  can be enriched with time series from other wavelength slices. The speckle
  field zooms radially by `s = λ_j/λ_ref` about the star center while
  astrophysical sources stay static, so per-slice masks are built by scaling
  the reference-pool geometry and excluding the static source position (sized
  with the slice FWHM plus margin), slice bad pixels, and known companions.
  Configured on `TrapReductionConfig` via `multiwavelength_regressors`
  (`None` | `"pool"` = full scaled annulus | `"occluded"` = scaled
  reference-signal-mask footprint, a subset of `"pool"` | `"sdi"` = the
  scaled footprint with the static-signal *and* known-companion exclusions
  dropped, so the speckle pool at the tested position is admitted; classic
  SDI trick for a "dark" donor channel — e.g. IRDIS K2 for methane-rich
  companions — will self-subtract where the source is bright, bad-pixel
  exclusion still applies), `regressor_wavelength_indices`, and
  `max_regressor_pool_size` (total pool budget in units of the
  single-wavelength pool; occluded pixels are always kept, annulus
  enrichment is subsampled per slice). The preprocessed cube is
  stored once as `(λ, y, x, t)` in the shared-array store for efficient
  scattered time-series reads; the per-wavelength `(t, y, x)` working slice
  is transposed out per iteration and removed afterwards. Single-wavelength
  reductions and the default `multiwavelength_regressors=None` path are
  bit-identical to before. Only mask construction and training-matrix
  concatenation changed; solvers are untouched.
- Optional coronagraph throughput correction: pass a `(separation_mas, throughput)`
  table via `TrapReductionConfig.coronagraph_transmission` to attenuate the
  forward model by the separation-dependent coronagraph transmission, correcting
  underestimated contrasts at small separations (#31).
- **Shared-array store** (`trap.shared_arrays`) – large input arrays are dumped
  once as `.npy` files to a scratch directory and worker processes memmap them
  read-only, so the OS page cache provides a single shared in-RAM copy per node.
  The scratch directory is configurable via `TrapReductionConfig.scratch_dir` /
  `TrapResources.scratch_dir` and defaults to `/dev/shm` when present with
  sufficient headroom (cluster nodes), otherwise the system temp directory
  (see `trap.parameters.resolve_scratch_dir`).
- Serial-vs-parallel equivalence test on a synthetic cube
  (`tests/test_parallel_equivalence.py`) plus unit tests for the shared-array
  store (`tests/test_shared_arrays.py`).

### Fixed
- `inject_signal` no longer raises when the injection stamp overlaps the
  array boundary — the destination slice and stamp are clipped per frame
  and frames entirely outside the array are skipped.
- `build_runtime_state` clamps the auto-computed `data_crop_size` at the
  input FoV with an INFO log instead of raising when the requested crop
  would exceed it.
- Detection-map peak extraction (`detection.py:4036`) is NaN-safe:
  positions marked `NaN` by the reduction no longer poison the `argmax`
  result of the candidate cluster.
- **`trap_config_for_irdis()` now sets `instrument_type="photometry"`** (was
  `"imaging"`). The old value matched neither branch of
  `SpectralTemplate.__init__` (which checks for `'ifu'` and `'photometry'`),
  so `contrast_modelbox` was never assigned and template-matching detection on
  IRDIS DBI crashed with `AttributeError: 'SpectralTemplate' object has no
  attribute 'contrast_modelbox'`. `"photometry"` selects the existing branch
  that integrates model spectra through per-channel filter bandpasses via
  `species.SyntheticPhotometry`, which is the correct treatment for DBI. No
  other TRAP-side changes are needed — callers just have to populate
  `Instrument.filters` with species-registered filter names (spherical's
  `run_trap` handles this via a SPHERE-specific obs-mode → SVO filter-name
  mapping).

### Changed
- **`include_noise` → `estimate_noise_from_data`; ivar cube is always used when passed.**
  The old gate on `TrapReductionConfig.include_noise` silently discarded any
  `inverse_variance_full` handed to `run_complete_reduction` when the flag was
  False, which was a footgun: passing an ivar cube looked like a request to
  use it, but the flag had to be flipped separately. The gate is now:
  explicit ivar always wins; the (renamed) `estimate_noise_from_data` flag
  only controls the fallback path when NO ivar is supplied (True estimates
  Poisson + read-noise from the data itself; False leaves the fit
  unweighted). Callers migrating: rename `include_noise=…` to
  `estimate_noise_from_data=…`; behavior is unchanged unless you were
  passing an ivar cube with `include_noise=False`, in which case it now
  actually gets used. Renamed everywhere (`parameters.py`, `regression.py`,
  the tutorial `.ipynb` / `.py` variants, the IRDIS debris tutorial, the
  CV validation scripts); dead `1./y` fallback in `regression.py`'s four
  inner gates removed (the top-level gate already covers it).
- **Ray removed; multiprocessing now uses joblib/loky** – `run_trap_search` and
  `multi_position_cross_validation` dispatch position chunks through
  `joblib.Parallel` (loky backend) instead of Ray remote functions. Results are
  identical to the serial path; single-wavelength reductions are unchanged.
  Startup time, memory reservation and dependency footprint shrink, worker
  logs/exceptions now surface on the driver, and one code path serves laptop
  and cluster (multi-node scaling via scheduler job arrays over
  wavelengths/epochs). BLAS threads in workers are capped to 1, matching Ray's
  previous implicit behavior.
- `run_complete_reduction` dumps the preprocessed per-wavelength data (and
  inverse variance) to the shared-array store once, before the
  component/wavelength loops, instead of re-transferring the cube to workers
  for every component fraction. Position chunking raised from `2 × ncpus` to
  `8 × ncpus` to reduce idle tails.
- Progress reporting is now a driver-side `tqdm` bar ticking per completed
  chunk; the Ray-based `ProgressBarActor`/`ProgressBar` in `trap.utils` were
  removed, along with the no-op `==` statements that belonged to them
  (roadmap item 5 / improvement note 7).

### Removed
- Dependency on `ray[default]`; `joblib` added instead.

### Fixed
- **Out-of-grid stellar parameters no longer abort template matching** – `add_default_templates` built the stellar template from the solar-only `bt-nextgen` grid but passed the requested `stellar_parameters` straight to `species`' `get_model`, so a sub-solar `[Fe/H]` (or an out-of-range Teff/log g) raised `ValueError: … smaller than the lower boundary of the model grid`. Values are now clamped to the grid boundaries (via `ReadModel.get_bounds()`) before `get_model`, snapping to the nearest edge with a `warnings.warn`, so any caller's stellar parameters degrade gracefully instead of crashing.

### Changed
- **Library logging instead of `print`** – Replaced the library's `print()` calls with standard-library `logging` (per-module `logging.getLogger(__name__)`, a single `NullHandler` at the package root). The library sets no levels or handlers of its own; callers control verbosity via `logging.getLogger("trap").setLevel(...)`, so a driver such as `spherical` can quiet routine output down to warnings/errors and keep its progress bar intact. Per-position diagnostics on the Ray worker path are `debug` only. `likelihood_tools.py` and `embed_shell.py` are unchanged.

## [1.3.0] - 2026-07-03

### Added
- **Dataclass-based configuration system** – New parameter classes (`TrapConfig`, `TrapReductionConfig`, `DetectionParameters`, `InstrumentConfig`, `StellarParameters`, `TrapResources`, `ProcessingParameters`) replace the legacy `Reduction_parameters` object as the primary way to configure reductions and detection. `detection.py` and the reduction wrapper now consume these classes directly.
- **Development environment** – Added `pixi.toml` for a reproducible Pixi-managed development environment.

### Changed
- **Internal cleanup** – Deduplicated detection-image population (`fill_detection_image`) and output-path construction (`OutputPaths`) in the reduction wrapper, and consolidated the `crop_box_*` helpers. No change to reduction results.
- **Correlation output naming** – The `correlation_matrix_binned` output now carries a `_corr` infix, consistent with the other residual-correlation outputs (affects only runs with residual correlation enabled).

### Deprecated
- **Legacy parameter objects** – `Reduction_parameters` and the `TrapReductionConfig.to_reduction_parameters()` / `TrapConfig.get_reduction_parameters()` bridge methods now emit a `DeprecationWarning` and will be removed in a future release. Use `TrapReductionConfig` / `TrapConfig` directly.

### Fixed
- **Known-companion regressor exclusion** – Removed a stray assignment that discarded the computed known-companion mask, so `yx_known_companion_position` again excludes known companions from the regressor pool.
- **Cross-validation robustness** – `temporal_pca_cross_validation` now fills failed solver fits with NaN instead of dropping into a debug shell.
- **Latent `NameError`** – `run_trap_with_model_wavelength` now accepts the `runtime` argument it referenced.
- **NumPy 2.0 compatibility** – Replaced the removed `np.histogram(normed=...)` argument with `density=...`.
- **Docstring typo** – Corrected `constrast_curve_sigma` to `contrast_curve_sigma`.

### Removed
- **Dead code** – Removed concluded experiments and unreachable/commented-out blocks (eigendecomposition and timing benchmarks in `pca_regression`, a post-`return` block and hardcoded plot limits in `regression`).
- **Unreleased Gaia coupling** – Removed the never-released `use_gaia_stellar_parameters` field from `DetectionParameters`; the stellar-parameter handover now lives entirely in the `spherical` wrapper.

## [1.2.1] - 2025-08-10

### Added
- **Signal-based Weighting** – Introduced `use_signal_weighting` parameter in reduction pipeline for improved signal-to-noise ratio in contrast estimation by weighting pixels based on expected signal strength.
- **Progress Bar Control** – Added `use_progress_bar` parameter to enable/disable progress feedback during long-running reductions, improving user experience and transparency.

### Changed
- **Reduction Mask Default** – Changed default `reduction_mask_size_in_lambda_over_d` from 2.0 to 1.0 pixels for better performance in typical science cases.
- **Search Region Expansion** – Increased default `search_region_outer_bound` from 55 to 85 pixels to improve detection performance.
- **Import Organization** – Switched from relative to absolute imports in regression module for better maintainability.

### Fixed
- **Candidate Validation** – Improved robustness in template matching and detection pipeline when no candidates survive second iteration, preventing downstream errors.

---

## [1.2.0] - 2025-07-07

### Added
- **Config Parameter System** – Introduced a new parameter configuration system based on dataclasses, following the spherical pipeline approach, improving flexibility and clarity in setting up defaults. Legacy objects are still used under the hood for now.

### Changed
- **Search Radius Consistency** – Standardized default `iterative_search_exclusion_radius` across detection pipeline and spectral extraction to 15 pixels, adjustable by users.
- **Annulus Width Consistency** – Removed hardcoded annulus width in spectral extraction to match pipeline-level configuration.
- **Improved Server Handling** – Ensured Ray server properly shuts down even in case of unexpected crashes, increasing pipeline robustness.
- **Outer Search Region Bound** – Cast `outer_search_region_bound` explicitly to integer for improved type safety.
- **Notebook Progressbar** – Notebook compatible progressbar is used by all modules. 

### Fixed
- **Critical FWHM Bug** – Corrected calculation of PSF size (`lambda/D` conversion) compatible with astropy >= 6.1. This critical bug kept the pipeline from working properly with up-to-date astropy.
- **Pickling Consistency** – Standardized object serialization using dill in both reduction and detection modules, fixing parameter persistence issues.
- **Latex Syntax Warning** – Fixed incorrect LaTeX escape sequences in detection curve plotting to eliminate syntax warnings.

---

## [1.1.0] - 2025-04-13

### Added
- **Forced Photometry Mode** – Introduced an option to perform *forced photometry* for known companion positions, allowing direct flux measurement at specified coordinates.  
- **Pickle I/O Utilities** – Added utility functions in `trap.utils` for saving and loading TRAP objects (e.g. results or models) via pickle, simplifying persistence of analysis results.  
- **Example Data & Tutorial** – Provided a Jupyter tutorial notebook and sample dataset in the `examples/` directory to help users get started with TRAP’s workflow.  
- **Documentation & CI** – Established a Sphinx documentation framework (`docs/` directory) and added continuous integration workflows (GitHub Actions for testing and docs).  
- **GitHub Templates** – Added issue templates for bug reports and feature requests, and a pull request template.

### Changed
- **Package Layout** – Restructured the project to a modern *“src”* layout under `src/trap/`, using PEP 621-based `pyproject.toml`.  
- **Python & Dependency Support** – Now supports Python 3.11+ (including Python 3.12). Dropped support for Python 3.9/3.10.  
- **Detection Defaults** – Improved defaults in the detection pipeline for better performance and usability.  
- **Detection Map Normalization** – Detection maps are now automatically **empirically normalized** to correspond to the detection significance (in σ) of a point source, improving interpretability.  
- **Logging Verbosity** – Reduced Ray's logging and multiprocessing noise for a cleaner CLI experience.  
- **Cross-Validation** – Adjusted the regression cross-validation strategy for better model selection.

### Fixed
- **Species Template Matching** – Fixed bugs in spectral template matching with `species`.  
- **NaN and Zero Handling** – Improved robustness to missing data and zero placeholders.  
- **Result Saving** – Fixed issues with saving contrast curves and spectral extraction overwriting detection maps.  
- **Parameter Bugs** – Fixed argument handling in wrappers and detection masking logic.  
- **Miscellaneous Fixes** – Code cleanup, better NaN handling, and bug fixes across modules.

### Removed
- **Legacy Code** – Removed unused code paths, imports, and debug routines.  
- **Python 3.9/3.10 Support** – Dropped support for older Python versions due to updated dependencies.

---

## [1.0.0] - 2024-03-28

### Added
- Initial release.

### Changed
- Initial implementation of core functionality.

### Fixed
- No known issues.

[Unreleased]: https://github.com/m-samland/trap/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/m-samland/trap/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/m-samland/trap/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/m-samland/trap/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/m-samland/trap/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/m-samland/trap/releases/tag/v1.0.0