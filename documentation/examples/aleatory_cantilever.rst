Aleatory Cantilever
===================

The ``aleatory_cantilever`` example demonstrates aleatory uncertainty
propagation and polynomial-chaos surrogate construction for a simple
cantilever-beam failure model. It compares three UQ workflows on the same
response:

- regression PCE built from Latin hypercube samples
- regression PCE built from Monte Carlo samples with a holdout split
- Gauss quadrature PCE

Run
---

.. code-block:: bash

    cd uam4x_examples/aleatory_cantilever
    python branching_script.py
    python run_analysis.py
    python post_analysis.py

``custom.py``
-------------

The subsystem stores beam geometry, loading, and material strength.

.. code-block:: python

    @register_subsystem
    class BeamProblem(Subsystem):
        def setup(self):
            self.add_parameter("width", 1.3, float)
            self.add_parameter("height", 2.0, float)
            self.add_parameter("axial_load", 500.0, float)
            self.add_parameter("bending_moment", 1000.0, float)
            self.add_parameter("yield_strength", 1000.0, float)

The plugin computes the beam failure margin ``g_val`` from those inputs.

.. code-block:: python

    @register_plugin
    class CalculateBeamFailure(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("height", float, 1.0)
            self.declare_input("width", float, 1.0)
            self.declare_input("axial_load", float, 1.0)
            self.declare_input("bending_moment", float, 1.0)
            self.declare_input("yield_strength", float, 1.0)
            self.declare_output("g_val", float, 1.0)

        def run_plugin(self, links, work_dir):
            g_val = 1 - 4 * links["bending_moment"] / (
                links["height"] ** 2 * links["width"] * links["yield_strength"]
            ) - (
                links["axial_load"]
                / (links["height"] * links["width"] * links["yield_strength"])
            ) ** 2
            return {"g_val": g_val}

``branching_script.py``
-----------------------

The model creates one vehicle with one beam component and assigns baseline
deterministic values.

.. code-block:: python

    def setup_model(study: Study):
        veh1 = Vehicle("vehicle1")
        beam = BeamProblem("beam1")
        veh1.add_component(beam)
        beam.set_parameter("width", 1.3)
        beam.set_parameter("height", 2.0)
        beam.set_parameter("axial_load", 500.0)
        beam.set_parameter("bending_moment", 1000.0)
        beam.set_parameter("yield_strength", 1000.0)
        study.add_vehicle(veh1)

The solution adds the beam-failure plugin and tracks ``calculate_failure.g_val``.

.. code-block:: python

    def setup_solution(study: Study):
        study.solution.add_step(CalculateBeamFailure, "calculate_failure")
        study.solution.add_link(
            "vehicles.vehicle1.beam1.height", "calculate_failure.height"
        )
        study.solution.add_tracked_output("calculate_failure.g_val")

The trade-study setup defines three aleatory analyses: two regression-PCE
studies and one Gauss quadrature PCE study.

.. code-block:: python

    def setup_tradestudy(study: Study):
        lhs_reg_pce_analysis = RegressionPCE(
            42, LatinHypercubeSampler, poly_order=5, seed=0
        )
        lhs_reg_pce_analysis.add_aleatory_variable(
            "vehicles.vehicle1.beam1.bending_moment",
            NormalDistribution(mean=2500.0, std=200.0),
        )
        lhs_reg_pce_analysis.add_aleatory_variable(
            "vehicles.vehicle1.beam1.yield_strength",
            UniformDistribution(low=2000.0, high=2800.0),
        )
        study.control.add_tradestudy("lhs_pce_analysis", lhs_reg_pce_analysis)

``run_analysis.py``
-------------------

The analysis script loads the generated variant and runs all three UQ trade
studies before saving results back to the variant directory.

.. code-block:: python

    if __name__ == "__main__":
        save_path = os.path.join(THIS_DIR, "variants", "nipc_uq")
        study = Study.load_study_from_directory(save_path)
        engine = Engine(study, os.path.join(save_path, "work"))
        engine.run_tradestudy("lhs_pce_analysis", num_processors=1)
        engine.run_tradestudy("mc_pce_analysis", num_processors=1)
        engine.run_tradestudy("quad_pce_analysis", num_processors=1)
        study.save_results_to_json(save_path)

``post_analysis.py``
--------------------

The post-processing script builds regression and quadrature surrogates,
computes moments, evaluates surrogate quality metrics, and compares UAM4X
results against the reference ``m4uq_ref`` data set.

For example, it constructs post-processing helpers from completed trade
studies, generates surrogates for ``calculate_failure.g_val``, and queries
analytical moments directly from those surrogate objects.

.. code-block:: python

    lhs_nipc_ts = study.results.get_tradestudy("lhs_pce_analysis")
    lhs_nipc_post = RegressionPCEPost(lhs_nipc_ts)
    lhs_nipc_surrogate = lhs_nipc_post.generate_surrogate("calculate_failure.g_val")
    lhs_mean = float(lhs_nipc_surrogate.analytical_mean)
    lhs_std = float(lhs_nipc_surrogate.analytical_std)

    quad_ts = study.results.get_tradestudy("quad_pce_analysis")
    quad_nipc_post = GaussQuadtraturePCEPost(quad_ts)
    quad_nipc_surrogate = quad_nipc_post.generate_surrogate("calculate_failure.g_val")
    quad_mean = float(quad_nipc_surrogate.analytical_mean)
    quad_std = float(quad_nipc_surrogate.analytical_std)

It also demonstrates surrogate validation against an independent test set by
assembling ``X_test`` and ``y_test`` from a second trade study and calling the
surrogate metric helpers.

.. code-block:: python

    mc_nipc_ts = study.results.get_tradestudy("mc_pce_analysis")
    x1 = np.array(mc_nipc_ts.get_variable(
        "vehicles.vehicle1.beam1.bending_moment"
    ))[:, None]
    x2 = np.array(mc_nipc_ts.get_variable(
        "vehicles.vehicle1.beam1.yield_strength"
    ))[:, None]
    X_test = np.hstack((x1, x2))
    y_test = np.array(mc_nipc_ts.get_variable("calculate_failure.g_val"))

    lhs_nipc_surrogate.get_surrogate_mse(X_test, y_test)
    lhs_nipc_surrogate.get_surrogate_r2(X_test, y_test)
    lhs_nipc_surrogate.get_surrogate_rel_mse(X_test, y_test)

The remainder of the script evaluates predictions on reference sample points
and reports confidence-interval and correlation comparisons against the
``m4uq_ref`` data set.
