# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **A single unfittable candidate no longer aborts the whole target.** `fit_2d_gaussian`
  used `LevMarLSQFitter`, which enforces bounds by clipping inside the objective: a
  parameter driven past a bound lands in a flat region whose numerical Jacobian column is
  zero, MINPACK returns NaN parameters and astropy raises `NonFiniteValueError`. Fit A
  leaves both widths and the orientation free, so this fired on speckle structure — 4 of
  72,464 above-threshold positions across four SPHERE/IRDIS targets, each one fatal. The
  fitter is now `TRFLSQFitter` (genuinely bounded, and what astropy recommends for bounded
  models), with a fixed-shape retry behind it and a NaN row carrying `fit_ok=False` as the
  last resort. An unfittable cutout — including the all-NaN case that previously raised
  `RuntimeError` — is reported, not raised. Cutouts trimmed at the frame edge no longer
  misalign the model grid.
- **A candidate at the inner working angle no longer destroys the reduction.** With
  `search_region_inner_bound=1` the contrast table is finite from 1 px, so the
  `smallest_separation_in_pixel` guard admitted the coronagraph centre residual as a
  candidate; rebuilding the noise profile with that candidate masked at
  `companion_mask_radius=11` blanked every annulus from 1 to 8 px, leaving its own fit
  cutout entirely NaN. Three independent guards now apply: new
  `DetectionParameters.minimum_candidate_separation` (5 px) drops candidates inside the
  stellar PSF core, `make_radial_profile` falls back to the un-masked annulus statistic
  below `minimum_annulus_pixels` (10) instead of writing NaN, and a companion's exclusion
  radius is capped at `separation - 1` so it cannot swallow the annuli it needs.
- **A failing template or per-channel astrometry no longer costs the combined tables.**
  `match_all_templates` iterates templates in dict order and `measure_per_channel_astrometry`
  runs *before* the overall tables are written, so one failure lost every later template and
  both `overall_*.csv` files even when the per-template tables were already complete. Each
  template, the contrast plotting, and the per-channel astrometry are now individually
  log-and-continue; the combined tables are built from whatever succeeded.

### Added
- **SNR-scaled candidate exclusion radius.** The radius blanked around an accepted peak
  scales as `sqrt(snr / candidate_threshold)`, capped at `2.5 ×` the base radius. A fixed
  radius is tuned for a marginal detection, but a bright binary's contamination is a swarm
  of detached above-threshold blobs that re-enter the search as spurious candidates: on
  HD_140408 (125σ binary at 35 px) this cut the candidate list from 18/14 to 9/6 per channel
  while leaving three clean targets — including one with a real 30σ source — unchanged.
  Connected above-threshold regions are masked alongside the disk. Controlled by
  `exclusion_radius_snr_scaling` / `max_exclusion_radius_factor`.
- **`DetectionParameters.candidate_exclusion_radius`** decouples the iterative search
  exclusion radius from `search_radius`, which also serves as the cross-template and
  cross-channel association radius — widening one used to silently widen the other. `None`
  keeps the previous shared behaviour.
- **`DetectionParameters.max_candidates`** (50) bounds the previously unlimited candidate
  loop. Every candidate costs a full contrast-table renormalization, so a saturated map was
  a runtime hazard as much as a scientific one; truncation is logged.

## [2.0.0] - 2026-07-30

First release with validated astrometry. Contains breaking changes — see the entries marked
**Breaking** under Changed and Removed.

### Added
- **Per-channel astrometry, reported as the primary position.** The spectral collapse used
  for template detection maximises SNR but is astrometrically biased — it folds in channels
  carrying no signal at the source. The companion is now fitted in each wavelength channel
  and combined in the source-aligned `(r, t)` frame, with two guards: the override needs at
  least `per_channel_min_channel_fraction` (0.5) of channels to contribute, and the combined
  σ is floored at the best contributing channel's σ unless
  `per_channel_independent_channels=True` — neighbouring channels are speckle-correlated, so
  the formal `1/Σ(1/σ²)` shrinkage claimed a √n gain that does not exist. New
  `astrometry_source` column (`per_channel` | `collapse`) and `per_channel_astrometry.csv`;
  detection significance, `best_template` and the spectrum still come from the collapse. On
  51 Eri IFS OBS_H the reported separation moved 447.66 → 454.30 mas against a GRAVITY truth
  of 455.364 ± 0.653 mas. (#35, benchmark in
  `spherical/tests/data/51eri_astrometry_benchmark.md`)
- **Edge-of-FoV reduction.** `run_complete_reduction` accepts a `valid_pixel_mask` (2D `H×W`
  or 3D `n_wave×H×W`) describing which detector pixels carry real data. Out-of-footprint
  positions are excluded from scheduling, per-position masks and regressor pools are
  intersected with the footprint, and positions with fewer than `reduction_mask_min_pixels`
  (30) surviving pixels return `NaN` instead of raising. `search_region_outer_bound` is now
  `Optional[int] = 85`, where `None` derives the bound from the footprint. Passing no mask
  leaves behaviour identical, but `trap_config_for_ifs` / `trap_config_for_irdis` set
  `auto_footprint=True`, which infers the mask from all-NaN pixels connected to the array
  border; the `TrapReductionConfig` default remains `False`. (#35)
- **Multi-wavelength regressors for IFS (WP2) — experimental, off by default.** The temporal
  regressor pool can be enriched
  with time series from other wavelength slices: the speckle field zooms radially by
  `s = λ_j/λ_ref` about the star centre while astrophysical sources stay static, so per-slice
  masks scale the reference-pool geometry and exclude the static source position, known
  companions and slice bad pixels. Configured via
  `TrapReductionConfig.multiwavelength_regressors` (`None` | `"pool"` = full scaled annulus |
  `"occluded"` = scaled reference-signal-mask footprint | `"sdi"` = that footprint with the
  static-source and known-companion exclusions dropped, the classic SDI trick for a "dark"
  donor channel), `regressor_wavelength_indices` and `max_regressor_pool_size`. Not yet
  validated across a range of datasets, and the modes are not expected to be equally useful:
  treat results as exploratory and compare against a `None` run. The single-wavelength and
  default `None` paths are bit-identical to before — only mask construction and training-matrix
  concatenation changed, solvers are untouched. (#35)
- **Shared-array store** (`trap.shared_arrays`) — large input arrays are dumped once as
  `.npy` files to a scratch directory and memmapped read-only by worker processes, so the OS
  page cache holds a single in-RAM copy per node. Configurable via
  `TrapReductionConfig.scratch_dir` / `TrapResources.scratch_dir`; defaults to `/dev/shm`
  when present with sufficient headroom, otherwise the system temp directory.

### Changed
- **Breaking: `include_noise` renamed to `estimate_noise_from_data`, and an explicit ivar
  cube is now always used.** The old flag silently discarded an `inverse_variance_full`
  handed to `run_complete_reduction` when False, which was a footgun. The gate is now:
  explicit ivar always wins, and the renamed flag only controls the fallback when no ivar is
  supplied (True estimates Poisson + read noise from the data, False leaves the fit
  unweighted). Callers must rename the keyword — there is no compatibility alias.
- **Breaking: Ray removed; multiprocessing now uses joblib/loky.** `run_trap_search` and
  `multi_position_cross_validation` dispatch position chunks through `joblib.Parallel`
  instead of Ray remote functions. Results are identical to the serial path and
  single-wavelength reductions are unchanged. Startup time, memory reservation and
  dependency footprint shrink, worker logs and exceptions now surface on the driver, and one
  code path serves laptop and cluster. Worker BLAS threads are capped to 1, matching Ray's
  previous implicit behaviour.
- **SPHERE anamorphism is now corrected by default** — `trap_config_for_ifs` sets
  `yx_anamorphism=[1.0059, 1.0011]` and `trap_config_for_irdis` `[1.0062, 1.0]`. The
  common-path cylindrical mirrors distort all three science subsystems, not just IRDIS (Maire
  et al. 2016); the IFS values match the correction Vigan's `sphere` package applies in
  `sph_ifs_combine_data`. TRAP compensates in the forward model rather than interpolating the
  data, so input cubes must stay uncorrected. Omitting it understated the separation of a
  source near the detector y axis by up to ~0.6% — 2.2 mas for 51 Eri b. (#35)
- Preprocessed per-wavelength data and inverse variance are dumped to the shared-array store
  once, before the component and wavelength loops, instead of being re-transferred to workers
  for every component fraction; position chunking raised from `2 × ncpus` to `8 × ncpus` to
  shorten idle tails. Progress reporting is now a driver-side `tqdm` bar ticking per
  completed chunk.
- **Library logging instead of `print`.** Per-module `logging.getLogger(__name__)` with a
  single `NullHandler` at the package root. The library sets no levels or handlers of its
  own, so callers control verbosity via `logging.getLogger("trap").setLevel(...)` — a driver
  such as `spherical` can quiet routine output to warnings and keep its progress bar intact.
  `likelihood_tools.py` and `embed_shell.py` are unchanged.

### Fixed
- **Unconstrained WLS parameters now return infinite variance instead of zero.**
  `solve_linear_equation_simple` caught a singular normal matrix — a reduction pixel whose
  inverse variance is zero in every signal-carrying frame — and returned
  `diag(pinv(AtWA)) = 0` for the unconstrained directions, which
  `compute_contrast_weighted_average` then inverts into an infinite weight. It takes a pixel
  with no usable data at all, so in practice this affected edge-of-FoV and heavily masked
  reductions; where it did occur the pixel could dominate the inverse-variance average or turn
  it into `NaN`. Only the singular except-path changed; well-conditioned fits are untouched.
  (`1e0e709`)
- **`_crop_box` pads instead of silently returning an empty array.** Raw numpy slicing on the
  last two axes turned a crop with `center=(126, 129)` and `boxsize=261` on a `(262, 262)`
  input into slice `[-4:257]` = `[258:257]`, i.e. shape `(0, 0)` rather than `(261, 261)`.
  All four public crop helpers now route through a padded implementation that always returns
  the requested shape, filling out-of-bounds regions by dtype (`NaN` / `False` / `0`), and
  `build_runtime_state`'s footprint crop additionally goes through `Cutout2D(mode='partial')`
  to survive fractional-pixel centres. Took down a bet Pic OBS_H run with a broadcast error.
  (`8f48961`, `4e7d9a8`)
- **A re-run can no longer leave another run's companion tables in `template_matching/`.**
  `companion_table_*`, `validated_companion_table*`, `companion_spectra_*.pdf` and
  `contrast_plot_*` were written on the success path of `run_template_matching` alone, so a
  template that found no candidate this run left the previous run's copies beside freshly
  written detection maps, indistinguishable from current results. The same held one level up
  for `overall_*.csv`, which *is* read as the run's result. Both now remove their products up
  front, so a missing file is the unambiguous signal for "this run found nothing"; products
  written before the candidate search are untouched. Fixes a latent `NameError` in passing —
  `template_name` and the output directory were bound only inside the `file_paths is None`
  branch but used unconditionally. (`7395902`)
- **Cross-template combination no longer ingests stale per-template CSVs.**
  `combine_template_matched_companion_tables` rebuilt the overall table by re-reading
  `{prefix}companion_table_{template}.csv` from disk, so a template that found nothing in the
  current run contributed its file from a previous run — inflating
  `n_templates_above_threshold`, fabricating `*_sigma_template_scatter` and tripping
  `astrometry_template_disagreement`. It now combines the in-memory tables populated by
  `match_all_templates` this run. (#35)
- **The stacked detection cube written by `DetectionAnalysis.read_output` no longer goes
  stale.** It was written under `if not os.path.exists(...)`, so
  `detection_ncomp???_frac?.??_temporal.fits` stayed frozen at whatever the first run
  produced while every other detection product refreshed. Analysis results were unaffected
  (they use the in-memory cube), but anyone opening the file directly, or keying on its
  modification time to check whether a reduction had rerun, saw the first run. (`404335c`)
- **`trap_config_for_irdis()` now sets `instrument_type="photometry"`** (was `"imaging"`).
  The old value matched neither branch of `SpectralTemplate.__init__`, so
  `contrast_modelbox` was never assigned and IRDIS DBI template matching crashed with
  `AttributeError`. `"photometry"` integrates model spectra through per-channel filter
  bandpasses via `species.SyntheticPhotometry`, the correct treatment for DBI; callers must
  populate `Instrument.filters` with species-registered filter names.
- Edge-of-FoV reduction no longer crashes with `numpy.linalg.LinAlgError: SVD did not
  converge` when pool pixels carry per-frame or per-wavelength `NaN` entries that pass the
  border-connected footprint gate. `_run_reduction_loops` ORs per-wavelength non-finite
  pixels into `bad_pixel_mask`, and `_assemble_training_matrix` drops any remaining
  non-finite columns before the SVD; downstream only uses the temporal basis, so dropping
  training-matrix columns is safe.
- Detection-map peak extraction is `NaN`-safe: positions marked `NaN` by the reduction no
  longer poison the `argmax` of the candidate cluster.
- `inject_signal` no longer raises when the injection stamp overlaps the array boundary — the
  destination slice and stamp are clipped per frame, and frames entirely outside the array
  are skipped.
- `build_runtime_state` clamps the auto-computed `data_crop_size` at the input FoV with an
  INFO log instead of raising when the requested crop would exceed it.
- The joblib/loky "A worker stopped while some jobs were given to the executor" warning
  during reduction: both `parallel_config` blocks in `reduction_wrapper` now pass
  `idle_worker_timeout=3600` (joblib's default is 300 s). One loky pool is reused across
  every wavelength channel and chunks are equalised by size rather than runtime, so a worker
  that ran out of positions early could idle past the timeout and be reaped. The warning was
  always harmless — it is emitted only on a graceful worker exit and loky respawns a
  replacement — but alarming in a log.

### Removed
- **Breaking: the legacy `Reduction_parameters` object and its bridge methods are gone.**
  Deprecated in 1.3.0, they are removed here: the `Reduction_parameters` class,
  `TrapReductionConfig.to_reduction_parameters()` and `TrapConfig.get_reduction_parameters()`.
  `TrapReductionConfig` / `TrapConfig` are now the only accepted configuration surface —
  `run_complete_reduction` and the detection analyses raise `TypeError` on anything else, and
  no code path emits a `DeprecationWarning` any more. Callers replace
  `config.get_reduction_parameters()` plus attribute assignment with
  `config.reduction.merge(result_folder=...)`, which returns a new frozen config instead of
  mutating one. Two consequences for stored results: `run_complete_reduction` no longer writes
  the duplicate `reduction_parameters.obj` beside `reduction_config.obj`, and
  `DetectionAnalysis.read_output(read_parameters=True)` can no longer read result folders
  produced before `reduction_config.obj` existed — it raises `FileNotFoundError` naming the
  cause rather than failing to unpickle the removed class. Such folders must be re-reduced, or
  read with `read_parameters=False` and an explicitly supplied config.
- Dependency on `ray[default]`; `joblib` added instead.
- `ProgressBarActor` / `ProgressBar` from `trap.utils`, along with the no-op `==` statements
  that belonged to them.

## [1.3.1] - 2026-07-08

### Added
- Optional coronagraph throughput correction: pass a `(separation_mas, throughput)` table via
  `TrapReductionConfig.coronagraph_transmission` to attenuate the forward model by the
  separation-dependent transmission, correcting underestimated contrasts at small
  separations. Applied at injection, so point-source contrasts and contrast curves are both
  corrected, and the table persists through `reduction_config.obj` so the detection module's
  candidate re-reduction uses the identical throughput. (#31)

### Fixed
- **Out-of-grid stellar parameters no longer abort template matching.**
  `add_default_templates` built the stellar template from the solar-only `bt-nextgen` grid
  but passed the requested `stellar_parameters` straight to `species`' `get_model`, so a
  sub-solar `[Fe/H]` or an out-of-range Teff/log g raised `ValueError`. Values are now
  clamped to the grid boundaries via `ReadModel.get_bounds()`, snapping to the nearest edge
  with a `warnings.warn`.
- **All-NaN cutouts now raise a clear error** (#30). A cutout containing only `NaN` raises an
  explicit `RuntimeError` instead of failing obscurely downstream, and `radial_bounds` is now
  consistently a tuple.

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

[Unreleased]: https://github.com/m-samland/trap/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/m-samland/trap/compare/v1.3.1...v2.0.0
[1.3.1]: https://github.com/m-samland/trap/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/m-samland/trap/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/m-samland/trap/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/m-samland/trap/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/m-samland/trap/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/m-samland/trap/releases/tag/v1.0.0