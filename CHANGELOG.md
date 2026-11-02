# Changelog

## Unreleased

### Added

- Add `EmptyMazeGridworld`, registered as `EmptyMaze-v0`: a deterministic open
  grid with start and goal in opposite corners and configurable dimensions
  defaulting to `dims=(4, 4)`.

### Changed

- Return a list of raw states from `env.states()` when `tabular=False`, in
  state-ID order. Tabular environments continue to return a range of IDs.
- Support `tabular=False` in `greedy_policy` and `greedy_policy_Q`, keeping
  value tables and returned policy arrays indexed by state ID.
- Rename the `epsilon_greedy_action` parameter `policy` to `Q` to reflect that
  it contains action values. Keyword callers must now use `Q=`.
- Raise `FloatingPointError` on NumPy overflow or invalid operations in
  function-approximation algorithms and helpers, including eligibility traces,
  policy gradients, and Fourier features. Restore callers' NumPy error settings
  after each call.
- Document policy helpers' compatibility with tabular, raw-state, and other
  Gymnasium environments.

### Fixed

- Preserve gridworld plotting, animation, and policy-array conversion when
  `env.states()` returns raw states.

## 1.0.4 - 2026-10-27

### Changed

- Automatically register environments with Gymnasium on import. Explicit
  `register()` calls remain supported and repeated calls are silent. Allow
  legacy Gym registration in addition to Gymnasium registration.
- Renamed `semi_gradient_TD0_estimation` to `semi_gradient_TD0` in
  `gym_classics2.algorithms.linear_approximation`.
- Renamed `semi_gradient_Sarsa_0` to `semi_gradient_Sarsa0` in
  `gym_classics2.algorithms.linear_approximation`.
- Remove explicit registration calls from example notebooks and documentation;
  explain that importing the package registers its environments.
- Open the example notebook links in Google Colab.
- Document the abstract and concrete environments and available algorithm families.
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
