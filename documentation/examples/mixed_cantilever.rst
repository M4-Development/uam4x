Mixed cantilever
================

This example combines aleatory uncertainty in bending moment and yield
strength with an epistemic interval for beam thickness. ``custom.py`` defines
the ``CantileverBase`` subsystem and ``CantiliverDeflection`` plugin. The
``mixed_cantilever`` variant has a ``basic`` solution tracking
``cantilever_deflection.g`` and three trade studies:

* ``mc_cantilever_problem`` uses ``BasicMixedIVP`` with Latin hypercube and
  Halton samplers.
* ``pce_cantilever_problem`` uses ``MixedRegressionPCE``.
* ``m4_samples`` evaluates the points in ``m4_ref/samples.csv`` with
  ``MixedCSVPoints``.

Run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/mixed_cantilever/branching_script.py
   python examples/mixed_cantilever/run_analysis.py
   python examples/mixed_cantilever/mc_post.py
   python examples/mixed_cantilever/pce_post.py
   python examples/mixed_cantilever/m4_post.py

``run_analysis.py`` runs all three trades. ``mc_post.py`` plots probability
boxes and horsetails for the direct mixed analysis. ``pce_post.py`` and
``m4_post.py`` fit second-order mixed PCE surrogates and plot probability
boxes, horsetails, and Sobol indices. The post scripts compare results with
the bundled ``m4_ref`` files.
