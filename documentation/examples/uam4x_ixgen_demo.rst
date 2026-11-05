IXGEN demo
==========

This example couples NDARC and IXGEN to update rotor blade geometry and mass.
``branching_script.py`` builds the ``quad`` variant with a ``basic`` solution,
real ``NdarcPlugin`` and ``IxgenPlugin`` steps, blade conversion and mass
calculations, and a ``vel_trade`` sweep over rotor tip speed with nested
optimization of ``Tech1``.

Run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/uam4x_ixgen_demo/branching_script.py
   python examples/uam4x_ixgen_demo/run_analysis.py

``run_analysis.py`` runs ``vel_trade`` and saves results under
``variants/quad/``. This path invokes the external NDARC and IXGEN
executables. Install ``rcotools`` and configure ``NDARC_EXE_PATH`` and
``IXGEN_EXE_PATH`` near the top of ``branching_script.py``. The NDARC and
IXGEN input files are under ``input_files/``.

``custom.py`` also defines ``FakeFastIxgen``, but the configured solution and
regression test use the real ``IxgenPlugin`` step.
