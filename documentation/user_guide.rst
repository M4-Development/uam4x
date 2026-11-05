User guide
==========

A study holds vehicles, solutions, trade studies, and variants. A solution
links vehicle parameters and plugin inputs and outputs. A trade study controls
parameter sweeps or optimizations. The :doc:`api_reference` describes the
framework classes; :doc:`plugins/index` describes the installed public plugin
APIs.

Work with an example
--------------------

The source examples are in ``examples/``. Each has a ``branching_script.py``
that creates and saves the study variants and an analysis script that runs
them. Most also define example-specific plugins or subsystems in ``custom.py``;
the quadrotor example uses ``conf.py`` for its variant configuration. Some
examples also include post-processing scripts.

From the ``uam4x`` project directory, after :doc:`installation`:

.. code-block:: bash

   python examples/multi_objective/branching_script.py
   python examples/multi_objective/run_analysis_no_vis.py
   python examples/multi_objective/post_process.py

The multi-objective example defines a wing, an expression-based area step,
and an objective calculation. It creates a sweep over objective weights and an
optimization. The ``no_vis`` solution has no OpenVSP execution step; the
``vis`` solution adds ``VSPUpdatePluginGenerator``. The branching script imports
that generator while creating the study, so the OpenVSP Python bindings and
Pillow are needed even for the ``no_vis`` workflow. To execute the visual
solution, use ``run_analysis_visual.py``. See
:doc:`examples/multi_objective` for both paths.

The analysis scripts use paths relative to their own files, so they can also
be launched from an IDE. Run the branching script first to generate the study
and variant directories expected by the analysis scripts. Each example page
lists its own scripts and external software requirements.
