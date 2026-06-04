#!/usr/bin/env bash
# Local mirror of .github/workflows/build_notebooks.yaml so contributors can
# regenerate the *.ipynb files from the *.py without pushing to GitHub.
#
# Requires `jupytext` and `jupyter nbconvert` to be available on PATH (both
# are pulled in by the `tracking` env if you ran setup.sh; otherwise install
# them with `pip install jupytext nbconvert` or `pipx install jupytext`).

set -euo pipefail

for dir in 01-Transformers 02-Tracking; do
  pushd "$dir" > /dev/null

  jupytext --to ipynb \
    --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' \
    solution.py

  jupyter nbconvert solution.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags solution \
    --to notebook --output exercise.ipynb

  jupyter nbconvert solution.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags task \
    --to notebook --output solution.ipynb

  popd > /dev/null
done
