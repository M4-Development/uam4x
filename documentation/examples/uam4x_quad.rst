Quad rotor
==========

This example sweeps rotor tip speed using NDARC and OpenVSP. The ``basic``
solution links the NDARC rotor radius output to an OpenVSP geometry group
and tracks OpenVSP wetted area and volume. ``branching_script.py`` creates
four variants: ``turbo_electric``, ``electric``, ``turbo_electric_at_sea``,
and ``electric_at_sea``. Each has a ``vtip_trade`` parameter sweep.

Run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/uam4x_quad/branching_script.py
   python examples/uam4x_quad/quad_rotor_analysis.py

``quad_rotor_analysis.py`` runs ``vtip_trade`` for all four variants and saves
results in each variant directory. The NDARC executable path and per-variant
input files are configured in ``examples/uam4x_quad/conf.py``. Install NDARC,
``rcotools``, the OpenVSP Python bindings, and Pillow before running.

Generator setup
---------------

``setup_solution`` uses ``NdarcPluginGenerator`` to declare the tip-speed
inputs and tracked radius output. It uses ``VSPGroupGenerator`` to configure
OpenVSP's update and analysis steps, then links the NDARC radius to the
OpenVSP parameter. This example does not use
``ExpressionPluginGenerator``:

.. literalinclude:: ../../examples/uam4x_quad/branching_script.py
   :language: python
   :pyobject: setup_solution
