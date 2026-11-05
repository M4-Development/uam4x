Examples
========

These eight examples live under ``examples/`` in this repository and are run as
source scripts from the ``uam4x`` project directory. Install the framework
first (:doc:`installation`). For each example, run its ``branching_script.py``
to generate variants, then its analysis script; run any post-processing scripts
after results have been saved. The pages below list the actual trade studies,
variants, dependencies, and commands used by the current scripts.

Some examples require external solvers or Python packages beyond the core
installation. Their pages call out those requirements and where local paths
must be configured.

.. toctree::
   :maxdepth: 1

   examples/aleatory_cantilever
   examples/mixed_cantilever
   examples/ishigami_problem
   examples/borehole_example
   examples/sellar_example
   examples/multi_objective
   examples/uam4x_quad
   examples/uam4x_ixgen_demo

Generator examples
------------------

The :doc:`examples/aleatory_cantilever` and
:doc:`examples/ishigami_problem` pages show their
``ExpressionPluginGenerator`` setup directly from ``branching_script.py``.
:doc:`examples/multi_objective` shows both expression and OpenVSP generator
usage. :doc:`examples/uam4x_quad` shows the NDARC and OpenVSP group
generators. The Borehole, mixed cantilever, Sellar, and IXGEN examples add
plugin classes directly rather than using those generators.
