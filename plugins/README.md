# Public UAM4X plugins

This directory contains the general, NDARC, OpenVSP, and IXGEN plugin source
packages. They are installed together with the main `uam4x` distribution by
running `python -m pip install -e .` from the project root. The plugin packages
keep their existing import names (`uam4x_general`, `uam4x_ndarc`,
`uam4x_openvsp`, and `uam4x_ixgen`).

CAMRAD II and M4SS remain separate private plugin packages. See the unified [plugin documentation](../documentation/plugins/index.rst)
and the main [README](../README.md) for example commands and external dependencies.
