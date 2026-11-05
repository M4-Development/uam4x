Sellar example
==============

This example solves the coupled Sellar optimization problem. ``custom.py``
defines the base subsystem, two discipline plugins, and the objective plugin.
``branching_script.py`` places the disciplines in a cyclic ``Group`` with a
``NonlinearBlockGS`` solver. The ``sellar`` variant has the ``sellar_basic``
solution and a ``sellar_problem`` optimization with design variables ``x``
and ``z``, two constraints, and one objective. The script selects
``pyOptSparseDriver``, so that driver must be available in the environment.

Run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/sellar_example/branching_script.py
   python examples/sellar_example/run_analysis.py
   python examples/sellar_example/run_post.py

``run_analysis.py`` runs the optimization and calls ``save_to_json`` to persist
the study and results under ``variants/sellar/``. ``run_post.py`` prints the
saved design variables, objective, and constraints.
