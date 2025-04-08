#!/bin/bash

# uv pip install currently leads to compatibility issues with ipywidgets.
# As a result, we are using the standard pip installation method.
# uv pip install --system --editable . 
pip install --root-user-action=ignore --editable . 

git config --global --add safe.directory /workspaces/juart
git config core.fileMode false