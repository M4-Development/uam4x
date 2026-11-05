Aleatory cantilever
===================

This example estimates uncertainty in a cantilever failure function. The
``BeamProblem`` subsystem in ``custom.py`` holds the beam parameters.
``branching_script.py`` uses ``ExpressionPluginGenerator`` to calculate and
track ``calculate_failure.g_val`` in the ``basic`` solution. It creates the
``nipc_uq`` variant with three trade studies:

* ``lhs_pce_analysis``: regression PCE with Latin hypercube samples.
* ``mc_pce_analysis``: regression PCE with Monte Carlo samples.
* ``quad_pce_analysis``: fifth-order Gaussian quadrature PCE.

The uncertain inputs are bending moment and yield strength. Run from the
``uam4x`` project directory, in this order:

.. code-block:: bash

   python examples/aleatory_cantilever/branching_script.py
   python examples/aleatory_cantilever/run_analysis.py
   python examples/aleatory_cantilever/post_analysis.py

``run_analysis.py`` executes all three trades and saves their results under
``examples/aleatory_cantilever/variants/nipc_uq/``. ``post_analysis.py`` fits
surrogates, compares their moments and prediction errors with the bundled
``m4uq_ref`` data, and writes ``m4uq_surrogate_comparison.xlsx`` in the variant
directory. The report needs the ``xlsxwriter`` package.

Expression plugin setup
-----------------------

The current ``setup_solution`` function declares the failure expression,
tracks ``g_val``, and links the beam parameters to generator inputs:

.. literalinclude:: ../../examples/aleatory_cantilever/branching_script.py
   :language: python
   :pyobject: setup_solution
