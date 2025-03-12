#!/bin/bash

# Set git to not change file permissions
git config core.fileMode false

# Make editable install of JAIL package
pip install -e /workspaces/jail[dev]