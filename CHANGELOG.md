# Changelog

## 1.0.2 - 2026-09-01

### Added

- Support for reproducible algorithm runs using NumPy generators or integer seeds.
- Tests for random-number handling and linear-approximation feature functions.

### Changed

- Pass state-feature functions explicitly to linear-approximation, eligibility-trace,
  and policy-gradient algorithms.
- Standardize argument ordering across policy and Monte Carlo helpers.
- Make Pygame an optional `render` dependency and add upper bounds to dependencies.

### Fixed

- Update plotting code for current Matplotlib behavior.
- Correct documentation citations and examples.
