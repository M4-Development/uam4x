# UAM4X

## Install

In a new Python environment, install from this directory:

```bash
python -m pip install -e .
```

This installs the UAM4X framework and the public `uam4x_general`,
`uam4x_ndarc`, `uam4x_openvsp`, and `uam4x_ixgen` plugin packages together.
Do not also install the duplicate public packages from the older
`uam4x_plugins` directory in the same environment. CAMRAD II and M4SS remain
separate private plugin installations.

## Run examples

The examples are scripts in the source checkout. Run them yourself from this
project directory after installing the package:

```bash
python examples/aleatory_cantilever/branching_script.py
python examples/aleatory_cantilever/run_analysis.py
```

Use the corresponding `branching_script.py` and analysis script for another
example. Some examples have different analysis filenames, such as
`uam4x_quad/quad_rotor_analysis.py` and
`multi_objective/run_analysis_no_vis.py`. Scripts use paths relative to their
own files, so they can also be run from an IDE.

NDARC and IXGEN examples require `rcotools` and their respective licensed
executables. OpenVSP examples need the OpenVSP Python bindings and Pillow.
The executable paths in the relevant example scripts must point to local
installations.

## Documentation

The framework, public plugin API, and all in-repository examples share one
Sphinx source at `documentation/`. Build it from this directory:

```bash
python -m pip install -r documentation/requirements.txt
sphinx-build -M html documentation documentation/_build
```
