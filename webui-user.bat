@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--enable-insecure-extension-access --api --xformers --opt-sdp-attention --no-half-vae --listen --skip-torch-cuda-test --skip-version-check --skip-python-version-check --disable-nan-check

call webui.bat
