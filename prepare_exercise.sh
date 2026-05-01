# Run black on .py files
black solution$1.py


jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' solution$1.py
jupyter nbconvert solution$1.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags solution --to notebook --output exercise$1.ipynb
jupyter nbconvert solution$1.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags task --to notebook --output solution$1.ipynb
