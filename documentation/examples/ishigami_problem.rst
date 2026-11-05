Ishigami problem
================

This example uses the current expression plugin and UQ APIs. ``custom.py``
defines the ``Ishigami`` subsystem and its parameters; it does not implement
the Ishigami calculation. ``branching_script.py`` builds the function as a
tracked ``ishigami_calc.g`` output with ``ExpressionPluginGenerator`` and
creates one ``ishigami`` variant.

The variant has two trade studies over ``X1``, ``X2``, and ``X3``:

* ``mc_ishigami_problem`` uses ``SobolSensitivity`` with a Sobol sampler.
* ``pce_ishigami_problem`` uses ``RegressionPCE`` with a Latin hypercube sampler.

Run from the ``uam4x`` project directory:

.. code-block:: bash

   python examples/ishigami_problem/branching_script.py
   python examples/ishigami_problem/run_analysis.py
   python examples/ishigami_problem/run_basic_post.py
   python examples/ishigami_problem/run_pce_post.py

``run_analysis.py`` runs both trades and saves results under
``variants/ishigami/``. The post-processing scripts compare computed Sobol
indices with analytical values and write Excel reports. They need the
``xlsxwriter`` package. The regression test runs the same trades directly
through ``Engine`` rather than launching ``run_analysis.py``.

Expression plugin setup
-----------------------

The calculation is configured in ``setup_solution``. ``Ishigami`` in
``custom.py`` supplies the parameter values, while the generated expression
step produces and tracks ``g``:

.. literalinclude:: ../../examples/ishigami_problem/branching_script.py
   :language: python
   :pyobject: setup_solution
