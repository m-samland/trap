# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/m-samland/spherical/compare/v1.1.0...HEAD 
[1.1.0]: https://github.com/m-samland/trap/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/m-samland/trap/releases/tag/v1.0.0