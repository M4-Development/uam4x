Multi-objective example
=======================

This wing example trades drag against weight. ``custom.py`` defines a
``WeightingFactor`` subsystem and ``CalcObjective`` plugin.
``branching_script.py`` uses ``ExpressionPluginGenerator`` for wing area and
creates two solutions: ``no_vis`` for calculations and ``vis`` with an
additional ``VSPUpdatePluginGenerator`` step. The ``demo`` variant contains a
``sweep`` of objective weights, each with an optimization, and a separate
``optimization_only`` trade.

For the calculation path, run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/multi_objective/branching_script.py
   python examples/multi_objective/run_analysis_no_vis.py
   python examples/multi_objective/post_process.py

``run_analysis_no_vis.py`` runs both trades and one single solution evaluation.
``post_process.py`` reads the ``no_vis`` results and makes a drag-versus-weight
scatter plot. To execute the OpenVSP solution instead, run
``run_analysis_visual.py`` after the branching script. It uses
``openvsp_config`` and OpenVSP graphics settings.

The branching script imports the OpenVSP plugin while creating both solutions,
so even the ``no_vis`` workflow needs the OpenVSP Python bindings and Pillow
available at import time. Only the ``vis`` solution executes OpenVSP.

Generator setup
---------------

``define_base_calc`` creates the expression-based area step in both
solutions and links its output to the objective plugin:

.. literalinclude:: ../../examples/multi_objective/branching_script.py
   :language: python
   :pyobject: define_base_calc

``define_visual`` adds the OpenVSP update generator only to ``vis``:

.. literalinclude:: ../../examples/multi_objective/branching_script.py
   :language: python
   :pyobject: define_visual
