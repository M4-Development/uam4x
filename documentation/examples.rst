Examples
========

This section documents the maintained examples in ``uam4x_examples``.
Each page follows the same workflow used by the examples themselves:

1. ``custom.py`` defines custom subsystems and plugins.
2. ``branching_script.py`` builds the study, solution graph, trade
   studies, and variants.
3. The analysis script loads a generated variant, runs the configured
   trade studies, and saves results.
4. Optional post-processing scripts read saved results and produce
   derived values, comparisons, or plots.

.. toctree::
   :maxdepth: 1

   examples/sellar_example
   examples/tilt_rotor
   examples/uam4x_quad
   examples/uam4x_ixgen_demo

