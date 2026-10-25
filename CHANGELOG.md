# Changelog

## Unreleased

## 1.0.4 - 2026-10-25

### Changed

- Accept policy arrays directly in semi-gradient TD(0) prediction for
  multidiscrete gym-classics environments.
- Accept sampled episodes with coordinate state labels in gridworld images.

## 1.0.3 - 2026-10-22

### Added

- Build and validate source and wheel distributions in CI, test the installed
  wheel, and publish GitHub releases through PyPI Trusted Publishing.
- Maintainer instructions for checking and publishing releases.

### Changed

- Use SPDX license metadata and include documentation and tests in source releases.
- Require Python 3.10 or newer; drop Python 3.9 support.
- Add Python 3.15 to the CI test matrix.

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
