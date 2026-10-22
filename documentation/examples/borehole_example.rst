Borehole Example
================

The ``borehole_example`` demonstrates regression polynomial-chaos modeling for
the classical borehole flow-rate benchmark with eight uncertain inputs.

Run
---

.. code-block:: bash

    cd uam4x_examples/borehole_example
    python branching_script.py
    python run_analysis.py
    python run_post.py

``custom.py``
-------------

The subsystem stores the eight borehole input variables, and the plugin
computes the scalar flow-rate response.

.. code-block:: python

    @register_subsystem
    class Borehole(Subsystem):
        def setup(self):
            self.add_parameter("rw", 0.0, float)
            self.add_parameter("r", 0.0, float)
            self.add_parameter("Tu", 0.0, float)
            self.add_parameter("Hu", 0.0, float)
            self.add_parameter("Tl", 0.0, float)
            self.add_parameter("Hl", 0.0, float)
            self.add_parameter("L", 0.0, float)
            self.add_parameter("Kw", 0.0, float)

The plugin applies the standard borehole equation:

.. code-block:: python

    @register_plugin
    class BoreholeAnalysis(Plugin):
        def run_plugin(self, links, work_dir):
            numerator = 2 * np.pi * links["Tu"] * (links["Hu"] - links["Hl"])
            denominator = np.log(links["r"] / links["rw"]) * (
                1
                + (2 * links["L"] * links["Tu"])
                / (np.log(links["r"] / links["rw"]) * links["rw"] ** 2 * links["Kw"])
                + links["Tu"] / links["Tl"]
            )
            return {"flow_rate": numerator / denominator}

``branching_script.py``
-----------------------

The model adds one borehole subsystem and routes all eight variables into a
single analysis step. The tracked output is ``borehole_calc.flow_rate``.

The trade-study setup uses a fifth-order ``RegressionPCE`` with Latin
hypercube sampling and eight aleatory variables, including normal,
lognormal, and uniform distributions.

.. code-block:: python

    uq = RegressionPCE(
        num_samples=num_samples,
        SamplingClass=LatinHypercubeSampler,
        seed=42,
    )
    uq.add_aleatory_variable(
        "vehicles.borehole_vehicle.borehole_base.rw",
        NormalDistribution(mean=0.1, std=0.0161812),
    )

``run_analysis.py``
-------------------

The analysis script loads the saved variant, runs the borehole trade study,
and writes results back to disk.

.. code-block:: python

    if __name__ == "__main__":
        study_dir = os.path.join(THIS_DIR, "variants", "borehole")
        study = Study.load_study_from_directory(study_dir)
        engine = Engine(study, os.path.join(study_dir, "work"))
        engine.run_tradestudy("borehole_problem", solution_name="basic", create_step_work_dirs=False)
        study.save_results_to_json(study_dir)

``run_post.py``
---------------

The post-processing script uses ``RegressionPCEPost`` to generate a surrogate
for ``borehole_calc.flow_rate`` and compares:

- analytical surrogate moments
- moments computed from surrogate-generated samples
- reference Monte Carlo moments from ``full_analysis.csv``
- pointwise surrogate predictions against Monte Carlo outputs
- empirical quantiles from the surrogate and reference data

.. code-block:: python

    solution_results = study.results.get_solution_result("basic")
    ts = solution_results.get_tradestudy("borehole_problem")
    uq_post = RegressionPCEPost(ts)
    surrogate_model = uq_post.generate_surrogate("borehole_calc.flow_rate", poly_order=5)
    analytical_mean = float(surrogate_model.analytical_mean)
    analytical_std = float(surrogate_model.analytical_std)
    new_samples, predicted_outputs = surrogate_model.sample(
        num_samples=5000, method="LHS"
    )
    prediction = surrogate_model.predict({
        "vehicles.borehole_vehicle.borehole_base.rw": 0.1,
        "vehicles.borehole_vehicle.borehole_base.r": 1000.0,
    })

It also writes a stacked comparison CSV for downstream inspection.
