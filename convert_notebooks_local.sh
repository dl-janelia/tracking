#!/bin/zsh

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Hey, you should source this script, not execute it!"
    echo "Try: source setup.sh"
    echo "Exiting..."
    exit 1
fi

PYJ=/Users/adomi/.local/pipx/venvs/jupytext/bin/python

for dir in 01-Transformers 02-Tracking; do
  cd $dir
  jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' solution.py

  "$PYJ" -m nbconvert solution.ipynb\
    --TagRemovePreprocessor.enabled=True\
    --TagRemovePreprocessor.remove_cell_tags solution --to notebook\
    --output exercise.ipynb

  "$PYJ" -m nbconvert solution.ipynb\
    --TagRemovePreprocessor.enabled=True\
    --TagRemovePreprocessor.remove_cell_tags task --to notebook --output\
    solution.ipynb
  cd ..
done