Borehole example
================

This example evaluates borehole flow rate with the ``BoreholeAnalysis`` plugin
defined in ``custom.py``. ``branching_script.py`` creates a ``borehole`` variant,
a ``basic`` solution tracking ``borehole_calc.flow_rate``, and one
``borehole_problem`` trade study. The trade uses ``RegressionPCE`` with Latin
hypercube sampling across eight uncertain inputs.

Run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/borehole_example/branching_script.py
   python examples/borehole_example/run_analysis.py
   python examples/borehole_example/run_post.py

``run_analysis.py`` saves the trade results under ``variants/borehole/``.
``run_post.py`` fits a fifth-order surrogate, compares its moments and
predictions with the bundled ``full_analysis.csv`` Monte Carlo reference,
and writes ``predicted_vs_mc_stacked.csv`` in the example directory.

``mc_borehole.py`` is a separate reference-data generator. Its output path is
currently hard-coded to an older checkout, so configure that path before
running it. The bundled ``full_analysis.csv`` lets the main workflow run
without regenerating the reference.
