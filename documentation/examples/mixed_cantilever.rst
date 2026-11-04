Mixed Cantilever
================

The ``mixed_cantilever`` example demonstrates mixed aleatory and epistemic
uncertainty for the cantilever-beam failure problem. It includes three
workflows:

- interval-valued propagation with nested sampling
- mixed regression PCE
- replay of externally generated sample points from a CSV-style reference set

Run
---

.. code-block:: bash

    cd uam4x_examples/mixed_cantilever
    python branching_script.py
    python run_analysis.py
    python mc_post.py
    python pce_post.py
    python m4_post.py

``custom.py``
-------------

The subsystem stores the cantilever inputs, and the plugin computes the
failure metric ``g``.

.. code-block:: python

    @register_subsystem
    class CantileverBase(Subsystem):
        def setup(self):
            self.add_parameter("M", 1.0, float)
            self.add_parameter("Y", 1.0, float)
            self.add_parameter("T", 1.0, float)
            self.add_parameter("b", 1.3, float)
            self.add_parameter("h", 2.0, float)
            self.add_parameter("P", 500.0, float)

    @register_plugin
    class CantiliverDeflection(Plugin):
        def run_plugin(self, links, work_dir):
            g = 1 - ((4 * links["M"]) / (links["b"] * links["h"] ** 2 * links["Y"])
                     - (links["P"] / (links["b"] * links["h"] * links["Y"])) ** 2) * links["T"]
            return {"g": g}

``branching_script.py``
-----------------------

The model builds one vehicle with a single cantilever subsystem and tracks the
``cantilever_deflection.g`` response.

.. code-block:: python

    from uam4x.solution import Solution

    def setup_solution(study: Study):
        solution = Solution("basic")
        solution.add_step(CantiliverDeflection, "cantilever_deflection")
        solution.add_link("vehicles.base.cantilever_base.M", "cantilever_deflection.M")
        solution.add_link("vehicles.base.cantilever_base.Y", "cantilever_deflection.Y")
        solution.add_link("vehicles.base.cantilever_base.T", "cantilever_deflection.T")
        solution.add_tracked_output("cantilever_deflection.g")
        study.add_solution(solution)

The trade-study setup adds:

- ``mc_cantilever_problem`` using ``BasicMixedIVP``
- ``pce_cantilever_problem`` using ``MixedRegressionPCE``
- ``m4_samples`` using ``MixedCSVPoints``

The first two use aleatory variables for moment and modulus plus an epistemic
interval on thickness factor ``T``. Polynomial order is chosen when fitting the
mixed regression surrogate, so either saved sample set can be fitted at a
supported order.

``run_analysis.py``
-------------------

The analysis script runs the three mixed-UQ trade studies and saves results.

.. code-block:: python

    if __name__ == "__main__":
        study_dir = os.path.join(THIS_DIR, "variants", "mixed_cantilever")
        study = Study.load_study_from_directory(study_dir)
        engine = Engine(study, os.path.join(study_dir, "work"))
        engine.run_tradestudy("mc_cantilever_problem", solution_name="basic", create_step_work_dirs=False)
        engine.run_tradestudy("pce_cantilever_problem", solution_name="basic", create_step_work_dirs=False)
        engine.run_tradestudy("m4_samples", solution_name="basic", create_step_work_dirs=False)
        study.save_results_to_json(study_dir)

Post-processing
---------------

``mc_post.py`` uses ``BasicMixedPost`` to compute p-level bounds and produce
p-box and horsetail plots for the direct mixed-sampling workflow.

.. code-block:: python

    solution_results = study.results.get_solution_result("basic")
    trade = solution_results.get_tradestudy("mc_cantilever_problem")
    post = BasicMixedPost(trade)
    low, high = post.get_plevel_bounds("cantilever_deflection.g", 0.0265)
    post.plot_pbox("cantilever_deflection.g", save_dir=study_dir)
    post.plot_horsetails("cantilever_deflection.g", save_dir=study_dir)

``pce_post.py`` uses ``MixedRegressionPCEPost`` to generate a mixed surrogate,
sample it, plot p-box and horsetail outputs, and compare Sobol indices and
predictions against the ``m4_ref`` reference files.

.. code-block:: python

    solution_results = study.results.get_solution_result("basic")
    trade = solution_results.get_tradestudy("pce_cantilever_problem")
    post = MixedRegressionPCEPost(trade)
    surrogate = post.generate_surrogate("cantilever_deflection.g", poly_order=2)
    surrogate.sample(50, 1000)
    surrogate.plot_pbox(save_dir=study_dir)
    surrogate.plot_horsetails(save_dir=study_dir)
    surrogate.get_total_order_sobol("vehicles.base.cantilever_base.Y")

``m4_post.py`` performs a similar surrogate-based comparison on the
``m4_samples`` trade study that reuses the external sample set.

.. code-block:: python

    solution_results = study.results.get_solution_result("basic")
    trade = solution_results.get_tradestudy("m4_samples")
    post = MixedRegressionPCEPost(trade)
    surrogate = post.generate_surrogate("cantilever_deflection.g", poly_order=2)
    prediction = surrogate.predict({
        "vehicles.base.cantilever_base.M": 2500.0,
        "vehicles.base.cantilever_base.Y": 2400.0,
        "vehicles.base.cantilever_base.T": 1.0,
    })
