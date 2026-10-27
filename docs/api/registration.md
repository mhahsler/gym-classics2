# Registration API

Importing `gym_classics2` automatically registers all bundled environments with
Gymnasium. Existing `register()` calls are harmless; repeated calls for the same
backend do nothing. Use `register(backend="gym")` to additionally register with
legacy Gym when it is installed.

::: gym_classics2.register
