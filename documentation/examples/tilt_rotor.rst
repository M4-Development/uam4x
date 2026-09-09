Tilt Rotor
==========

First, check out the example repository:

.. code-block:: bash

    git clone git@github.com:M4-Development/uam4x_tiltrotor.git
    cd uam4x_tiltrotor

You can also use HTTPS:

.. code-block:: bash

    git clone https://github.com/M4-Development/uam4x_tiltrotor.git
    cd uam4x_tiltrotor

The ``tilt_rotor`` example demonstrates a coupled sizing and weight
workflow. It builds a multi-tiltrotor vehicle, links passenger payload
and technology factors into NDARC, aggregates component weights, runs an
OpenVSP/M4SS-style structural weight step, and tracks the residual
between the sizing and structural paths.

Run
---

.. code-block:: bash

    cd uam4x_examples/tilt_rotor
    python branching_script.py
    python run_analysis.py

Dependencies
------------

The example is configured for ``uam4x_ndarc``, ``uam4x_m4ss``,
``openvsp``, NDARC, M4SS, and NASTRAN. The checked-in workflow uses
fake-fast helper plugins for parts of the M4SS/OpenVSP path.

``custom.py``
-------------

Technology factors are stored on the vehicle and linked into NDARC.

.. code-block:: python

    @register_subsystem
    class NdarcTechFactors(Subsystem):
        def setup(self):
            self.add_parameter("Tech1", 1.0, float)
            self.add_parameter("Tech2", 1.0, float)
            self.add_parameter("Tech3", 1.0, float)
            self.add_parameter("Tech4", 1.0, float)
            self.add_parameter("Tech5", 1.0, float)
            self.add_parameter("Tech6", 1.0, float)
            self.add_parameter("Tech7", 1.0, float)
            self.add_parameter("Tech8", 1.0, float)
            self.add_parameter("Tech9", 1.0, float)
            self.add_parameter("Tech10", 1.0, float)

Passenger weight and component summation plugins provide simple glue
calculations around the external tools.

.. code-block:: python

    @register_plugin
    class CalcTotalPassWeight(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("npass", int, 0)
            self.declare_input("wpass", float, 0.0)
            self.declare_output("w_pass_tot", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"w_pass_tot": links["npass"] * links["wpass"]}

    @register_plugin
    class SumComponents(Plugin):
        def setup_plugin(self, plugin_options):
            self.ncomponents = plugin_options["ncomponents"]
            for i in range(self.ncomponents):
                self.declare_input(f"W{i + 1}", float, 0.0)
            self.declare_output("total", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"total": sum(links[f"W{i + 1}"] for i in range(self.ncomponents))}

The fake-fast OpenVSP/M4SS plugin keeps the same input/output shape as
the heavier tool chain but allows for faster evaluations.

.. code-block:: python

    @register_plugin
    class FakeFastOpenvspM4ssPlugin(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("openvsp.parm_id.HFZNSRXUBZN", float, 0.0)
            self.declare_input("openvsp.parm_id.WNIEKOFKSMF", float, 0.0)
            self.declare_input("m4ss.MODIFY.FuselageGeom.88", float, 0.0)
            self.declare_output("m4ss.weight.total", float, 0.0)
            self.declare_output("m4ss.weight.structural", float, 0.0)
            self.declare_output("m4ss.weight.nsm", float, 0.0)

        def run_plugin(self, links, work_dir):
            results_openvsp = self.run_openvsp(links_openvsp, work_dir)
            results_m4ss = self.run_fake_fast_m4ss(
                links_m4ss,
                results_openvsp["vsp3_path"],
                work_dir,
            )
            return results

The residual plugin compares the weights from the two paths.

.. code-block:: python

    @register_plugin
    class Residual(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("w1", float, 0.0)
            self.declare_input("w2", float, 0.0)
            self.declare_output("residual", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"residual": links["w1"] - links["w2"]}

``branching_script.py``
-----------------------

The model setup creates the tiltrotor vehicle and flight-mode rotor
parameters.

.. code-block:: python

    def setup_turbo_electric_model(study: Study):
        mtr = Vehicle("mtr")
        wing = Wing("wing")
        wing.set_parameter("span", 35.56)
        wing.set_parameter("area", 142.1)
        aircraft = Aircraft("aircraft")
        aircraft.set_parameter("npass", 6)
        aircraft.set_parameter("wpass", 200.0)
        tech_factors = NdarcTechFactors("tech")

        rotor = Rotor("rotor")
        rotor.set_parameter("nblade", 8)
        rotor.add_mode("vflight").set_parameter("tilt", 90.0)
        rotor.add_mode("cruise").set_parameter("tilt", 0.0)

        mtr.add_component(wing)
        mtr.add_component(rotor)
        mtr.add_component(aircraft)
        mtr.add_component(tech_factors)
        study.add_vehicle(mtr)

The electric variant edits the NDARC step to use an electric NJOB file.

.. code-block:: python

    def modify_to_electric(study: Study):
        study.get_solution("basic").edit_step("ndarc", {"njob_path": NJOB_ELECTRIC_PATH})

The solution setup starts with passenger payload, NDARC, and NDARC
output aggregation.

.. code-block:: python

    from uam4x.solution import Solution

    def setup_solution(study: Study):
        solution = Solution("basic")
        solution.add_step(CalcTotalPassWeight, "totalPassWeight", {})
        solution.add_link("vehicles.mtr.aircraft.npass", "totalPassWeight.npass")
        solution.add_link("vehicles.mtr.aircraft.wpass", "totalPassWeight.wpass")

        solution.add_step(NdarcPlugin, "ndarc", {
            "njob_path": NJOB_PATH,
            "ndarc_exe": NDARC_EXE_PATH,
            "out_files": {
                "OUT_geometry": 0,
                "OUT_design": 1,
                "OUT_aircraft": 0,
                "OUT_perf": 0,
                "OUT_solution": 1,
            },
            "input_dict": {
                "TechFactors.TECH_body": {"type": "float", "default_value": 1.0},
                "TechFactors.TECH_prim(1)": {"type": "float", "default_value": 1.0},
                "Wing(1).span": {"type": "float", "default_value": 1.0},
                "SIZEMISSION(1).Wpay": {"type": "float", "default_value": 1.0},
                "SIZEMISSION(2).Wpay": {"type": "float", "default_value": 1.0},
            },
            "output_request_dict": {
                "design.Aircraft Weight Detail.Rotor 1.WEIGHT EMPTY.total.lb": {
                    "type": "float",
                    "default_value": 0.0,
                },
                "design.Aircraft Weight Detail.Wing 1.WEIGHT EMPTY.total.lb": {
                    "type": "float",
                    "default_value": 0.0,
                },
            },
        })

        solution.add_link("vehicles.mtr.tech.Tech1", "ndarc.TechFactors.TECH_body")
        solution.add_link("vehicles.mtr.tech.Tech2", "ndarc.TechFactors.TECH_prim(1)")
        solution.add_link("vehicles.mtr.wing.span", "ndarc.Wing(1).span")
        solution.add_link("totalPassWeight.w_pass_tot", "ndarc.SIZEMISSION(1).Wpay")
        solution.add_link("totalPassWeight.w_pass_tot", "ndarc.SIZEMISSION(2).Wpay")

The full script then adds ``SumComponents`` steps, the
``FakeFastOpenvspM4ssPlugin`` step, residual calculations, and tracked
outputs.

.. code-block:: python

        solution.add_step(SumComponents, "nsm1", {"ncomponents": 3})
        solution.add_link(
            "ndarc.design.Aircraft_Weight_Detail.Rotor_1.WEIGHT_EMPTY.total.lb",
            "nsm1.W1",
        )

        solution.add_step(FakeFastOpenvspM4ssPlugin, "openvsp_m4ss_vertical", {
            "base_vsp3_path": VSP_FILE_PATH,
            "veh_path": VEH_PATH,
            "analysis_id": 1,
            "m4ss_exe_path": M4SS_EXE_PATH,
            "spl_path": SPL_PATH,
            "nastran_exe": NASTRAN_EXE_PATH,
            "weight_scale": 32.2,
            "input_dict": {},
            "output_request_dict": {},
        })
        solution.add_link(
            "vehicles.mtr.wing.span",
            "openvsp_m4ss_vertical.openvsp.parm_id.HFZNSRXUBZN",
        )
        study.add_solution(solution)

The trade study drives the coupled solve.

.. code-block:: python

    def setup_tradestudy(study: Study):
        parameter_sweep = ParameterSweep({
            "vehicles.mtr.tech.Tech1": [1.0],
            "vehicles.mtr.tech.Tech2": [1.0],
        })
        study.control.add_tradestudy("ndarc_m4ss_couple", parameter_sweep)

``run_analysis.py``
-------------------

The analysis script iterates active variants, runs the coupled trade study, and writes
results back to the variant directory.

.. code-block:: python

    ACTIVE_VARIANTS = ["turbo_electric"]

    if __name__ == "__main__":
        for active_variant in ACTIVE_VARIANTS:
            variant_path = os.path.join(VARIANTS_DIR, active_variant)
            work_dir = os.path.join(VARIANTS_DIR, active_variant, "work")
            study = Study.from_json(variant_path)
            engine = Engine(study, work_dir)
            engine.run_tradestudy("ndarc_m4ss_couple", solution_name="basic")
            study.save_results_to_json(variant_path)
