Ishigami Problem
================

The ``ishigami_problem`` example documents a standard nonlinear sensitivity
benchmark. It compares direct Sobol sensitivity estimation against a
regression-PCE surrogate on the Ishigami function.

Run
---

.. code-block:: bash

    cd uam4x_ex/ishigami_problem
    python branching_script.py
    python run_analysis.py
    python run_basic_post.py
    python run_pce_post.py

``custom.py``
-------------

The subsystem stores Ishigami parameters and inputs. The solution uses an
expression plugin to compute the scalar response ``g``.

.. code-block:: python

    @register_subsystem
    class Ishigami(Subsystem):
        def setup(self):
            self.add_parameter("a", 7.0, float)
            self.add_parameter("b", 0.1, float)
            self.add_parameter("X1", 0.0, float)
            self.add_parameter("X2", 0.0, float)
            self.add_parameter("X3", 0.0, float)

``branching_script.py``
-----------------------

The model contains a single Ishigami subsystem on one vehicle. The solution
links all five inputs into the expression plugin and tracks
``ishigami_calc.g``.

.. code-block:: python

    ishigami = ExpressionPluginGenerator("ishigami_calc", solution=solution)
    for name in ("a", "b", "X1", "X2", "X3"):
        ishigami.add_float_input(name)
    ishigami.add_float_output(
        "g", expr="sin(X1)+a*sin(X2)**2+b*X3**4*sin(X1)", tracked=True
    )
    ishigami.add_to_study()
    for name in ("a", "b", "X1", "X2", "X3"):
        solution.add_link(
            f"vehicles.ishigami_vehicle.ishigami_base.{name}",
            ishigami.input(name),
        )

Two UQ trade studies are configured:

- ``mc_ishigami_problem`` using ``SobolSensitivity`` with Sobol sampling
- ``pce_ishigami_problem`` using ``RegressionPCE`` with Latin hypercube samples

.. code-block:: python

    mc_uq = SobolSensitivity(5000, SamplingClass=SobolSampler, seed=42)
    mc_uq.add_aleatory_variable(
        "vehicles.ishigami_vehicle.ishigami_base.X1",
        UniformDistribution(low=-np.pi, high=np.pi),
    )

``run_analysis.py``
-------------------

The analysis script loads the generated variant, runs both trade studies, and
saves the results.

.. code-block:: python

    if __name__ == "__main__":
        study_dir = os.path.join(THIS_DIR, "variants", "ishigami")
        study = Study.load_study_from_directory(study_dir)
        engine = Engine(study, os.path.join(study_dir, "work"))
        engine.run_tradestudy("pce_ishigami_problem", solution_name="basic", create_step_work_dirs=False)
        engine.run_tradestudy("mc_ishigami_problem", solution_name="basic", create_step_work_dirs=False)
        study.save_results_to_json(study_dir)

Post-processing
---------------

``run_basic_post.py`` uses ``SobolSensitivityPost`` on the direct sampling
study, plots total-order indices, and compares computed first-order and
total-order Sobol terms against analytical Ishigami reference values.

.. code-block:: python

    solution_results = study.results.get_solution_result("basic")
    ts = solution_results.get_tradestudy("mc_ishigami_problem")
    uq_post = SobolSensitivityPost(ts)
    uq_post.plot_total_order_indices("ishigami_calc.g", save_dir=study_dir)
    first_order_terms = uq_post.get_first_order_indices("ishigami_calc.g")
    total_order_terms = uq_post.get_total_order_indices("ishigami_calc.g")

``run_pce_post.py`` uses ``RegressionPCEPost`` to build a surrogate from the
PCE trade study, evaluate first-order and total-order Sobol indices from the
surrogate, and compare them against the same analytical reference values.

.. code-block:: python

    solution_results = study.results.get_solution_result("basic")
    uq_trade = solution_results.get_tradestudy("pce_ishigami_problem")
    uq_post = RegressionPCEPost(uq_trade)
    surrogate = uq_post.generate_surrogate("ishigami_calc.g", poly_order=7)
    s1 = surrogate.get_first_order_sobol(
        "vehicles.ishigami_vehicle.ishigami_base.X1"
    )
    st1 = surrogate.get_total_order_sobol(
        "vehicles.ishigami_vehicle.ishigami_base.X1"
    )
    surrogate.plot_total_order_sobol(save_dir=study_dir)

Both scripts write tabular comparison outputs to Excel files in the variant
directory.
