# Publishing a release

Run these commands from the standalone `gym-classics2` repository.

## Check the distributions locally

Use a clean checkout and a temporary virtual environment:

```bash
python -m venv /tmp/gym-classics2-release
source /tmp/gym-classics2-release/bin/activate
python -m pip install "build>=1.2,<2" "twine>=6,<7"
python -m build --outdir /tmp/gym-classics2-dist
python -m twine check --strict /tmp/gym-classics2-dist/*
python -m pip install /tmp/gym-classics2-dist/*.whl "pytest>=7,<10"
python -m pip check
python -m pytest
```

Use an empty output directory for each build. `python -m build` builds the
wheel from the source distribution, checking that the source archive contains
the files needed to build it. Also test the installed wheel outside the source
checkout (as the Package workflow does):

```bash
cd /tmp
python -m pytest --import-mode=importlib /absolute/path/to/gym-classics2/tests
```

## Configure publishing once

1. Create a GitHub environment named `pypi` in `mhahsler/gym-classics2`.
2. In PyPI, configure a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
   for project `gym-classics2`, owner `mhahsler`, repository `gym-classics2`,
   workflow `release.yml`, and environment `pypi`. For the first upload, use
   PyPI's pending publisher form; for an existing project, use its Publishing settings.

The workflow uses short-lived credentials from GitHub; no API token is needed.

## Release

1. Choose an unused PyPI version in `pyproject.toml`. Move the relevant changelog
   entries from Unreleased into a dated version section.
2. Commit and push the release changes. Check that the Documentation and Package
   workflows pass.
3. Create and publish a GitHub release with tag `v<version>` pointing to that
   commit (for example, `v1.0.3`). The Package workflow verifies the tag, builds
   both distributions, validates their metadata, and tests the installed wheel
   on the supported Python versions before uploading to PyPI.
4. Verify the project page and install the release in a fresh environment:

   ```bash
   python -m pip install "gym-classics2==1.0.3"
   python -m pip check
   ```

PyPI does not allow replacing an uploaded distribution. Fix release problems
with a new version. Pushes, pull requests, and manual workflow runs validate
packages without publishing them.
