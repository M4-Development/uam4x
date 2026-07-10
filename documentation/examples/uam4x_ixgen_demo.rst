uam4x IXGEN Demo
================

First, check out the example repository:

.. code-block:: bash

    git clone git@github.com:M4-Development/uam4x_ixgen_demo.git
    cd uam4x_ixgen_demo

You can also use HTTPS:

.. code-block:: bash

    git clone https://github.com/M4-Development/uam4x_ixgen_demo.git
    cd uam4x_ixgen_demo

The ``uam4x_ixgen_demo`` example demonstrates rotor-blade structural
feedback. NDARC provides blade sizing values, helper plugins convert
those values into IXGEN inputs, IXGEN returns sectional masses, and a
residual compares the IXGEN mass against the NDARC blade weight.

Run
---

.. code-block:: bash

    cd uam4x_examples/uam4x_ixgen_demo
    python branching_script.py
    python run_analysis.py

Dependencies
------------

This example is configured for ``uam4x_ndarc``, ``uam4x_ixgen``, NDARC,
and IXGEN executable paths set in ``branching_script.py``.

``custom.py``
-------------

The technology-factor subsystem feeds NDARC blade technology inputs.

.. code-block:: python

    @register_subsystem
    class NdarcTechFactorsIxgenDemo(Subsystem):
        def setup(self):
            self.add_parameter("Tech1", 1.0, float)

The conversion and helper plugins reshape NDARC outputs for IXGEN and
then convert IXGEN results back to NDARC-compatible units.

.. code-block:: python

    @register_plugin
    class Ft2Meters(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("Rotor(1).radius", float, 0.0)
            self.declare_input("Rotor(1).chord", float, 0.0)
            self.declare_output("Rotor(1).radius.meters", float, 0.0)
            self.declare_output("Rotor(1).chord.meters", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {
                "Rotor(1).radius.meters": links["Rotor(1).radius"] * 0.3048,
                "Rotor(1).chord.meters": links["Rotor(1).chord"] * 0.3048,
            }

    @register_plugin
    class CalcBladeChord(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("chord", float, 0.0)
            self.declare_input("taper", float, 0.0)
            self.declare_output("root_chord", float, 0.0)
            self.declare_output("tip_chord", float, 0.0)

        def run_plugin(self, links, work_dir):
            root_chord = 2.0 * links["chord"] / (1.0 + links["taper"])
            return {
                "root_chord": root_chord,
                "tip_chord": root_chord * links["taper"],
            }

    @register_plugin
    class CalcCrossSectionTwist(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("twist", float, 0.0)
            self.declare_output("root_twist", float, 0.0)
            self.declare_output("tip_twist", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"root_twist": 0.0, "tip_twist": links["twist"]}

Mass helper plugins convert and compare blade masses.

.. code-block:: python

    @register_plugin
    class kg_m2lbs_ft(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("region.1.inboard.mass", float, 0.0)
            self.declare_input("region.1.outboard.mass", float, 0.0)
            self.declare_input("region.2.inboard.mass", float, 0.0)
            self.declare_input("region.2.outboard.mass", float, 0.0)
            self.declare_output("region.1.inboard.mass.lbs_ft", float, 0.0)

        def run_plugin(self, links, work_dir):
            # The full script converts each kg/m value to lb/ft.
            return converted_results

    @register_plugin
    class CalcTotalMassFromCrossSection(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("region.1.inboard.mass", float, 0.0)
            self.declare_input("region.1.outboard.mass", float, 0.0)
            self.declare_input("region.2.inboard.mass", float, 0.0)
            self.declare_input("region.2.outboard.mass", float, 0.0)
            self.declare_output("total_mass", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"total_mass": sum(links.values())}

    @register_plugin
    class CalcSingleBladeWeightNdarc(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("rotors_weight", float, 0.0)
            self.declare_input("nblade", float, 2.0)
            self.declare_output("single_blade_weight", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"single_blade_weight": links["rotors_weight"] / links["nblade"]}

    @register_plugin
    class ResidualIxgenDemo(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("ixgen_weight", float, 0.0)
            self.declare_input("ndarc_weight", float, 0.0)
            self.declare_output("residual", float, 0.0)

        def run_plugin(self, links, work_dir):
            return {"residual": links["ixgen_weight"] - links["ndarc_weight"]}

``branching_script.py``
-----------------------

The model creates the quad vehicle, rotor, blade regions, and technology
factor component.

.. code-block:: python

    def setup_quad_model(study: Study):
        quad = Vehicle("quad")
        rotor1 = Rotor("rotor")
        rotor1.set_parameter("radius", 8.0)

        region1 = RotorBladeRegion("region1")
        region1.set_parameter("spanwise_start", 0.01)
        region1.set_parameter("spanwise_stop", 0.05)
        region1.set_parameter("twist_start", 0.0)
        region1.set_parameter("twist_stop", 0.0)
        region1.set_parameter("sweep", 0.0)

        region2 = RotorBladeRegion("region2")
        region2.set_parameter("spanwise_start", 0.06)
        region2.set_parameter("spanwise_stop", 1.0)
        region2.set_parameter("mainsparweb_thickness", 0.002)

        tech_factors = NdarcTechFactorsIxgenDemo("tech")
        quad.add_component(rotor1)
        quad.add_component(region1)
        quad.add_component(region2)
        quad.add_component(tech_factors)
        study.add_vehicle(quad)

The solution starts with NDARC, converts its outputs, and links them into
IXGEN.

.. code-block:: python

    def setup_solution(study: Study):
        study.solution.add_step(NdarcPlugin, "ndarc", {
            "njob_path": NJOB_PATH,
            "ndarc_exe": NDARC_EXE_PATH,
            "step_size": 0.1,
            "input_dict": {
                "Rotor(1).TECH_blade": {"type": "float", "default_value": 0.0},
                "Rotor(1).Vtip_ref": {"type": "float", "default_value": 0.0},
            },
            "output_request_dict": {
                "soln.Rotor(1).CHORD": {"type": "float", "default_value": 0.0},
                "soln.Rotor(1).TAPER": {"type": "float", "default_value": 0.0},
                "soln.Rotor(1).RADIUS": {"type": "float", "default_value": 0.0},
                "soln.Rotor(1).TWISTL": {"type": "float", "default_value": 0.0},
                "soln.Rotor(1).WEIGHT.W_ROTOR_BLADE": {
                    "type": "float",
                    "default_value": 0.0,
                },
                "soln.Rotor(1).NBLADE": {"type": "float", "default_value": 2.0},
            },
        })
        study.solution.add_link("vehicles.quad.tech.Tech1", "ndarc.Rotor(1).TECH_blade")
        study.solution.add_link("vehicles.quad.rotor.vtip", "ndarc.Rotor(1).Vtip_ref")

        study.solution.add_step(Ft2Meters, "lengthMETERS", {"input_dict": {}})
        study.solution.add_link("ndarc.soln.Rotor(1).RADIUS", "lengthMETERS.Rotor(1).radius")
        study.solution.add_link("ndarc.soln.Rotor(1).CHORD", "lengthMETERS.Rotor(1).chord")

        study.solution.add_step(CalcBladeChord, "CrossSectionChord", {})
        study.solution.add_link("lengthMETERS.Rotor(1).chord.meters", "CrossSectionChord.chord")
        study.solution.add_link("ndarc.soln.Rotor(1).TAPER", "CrossSectionChord.taper")

        study.solution.add_step(CalcCrossSectionTwist, "CrossSectionTwist", {})
        study.solution.add_link("ndarc.soln.Rotor(1).TWISTL", "CrossSectionTwist.twist")

The IXGEN step receives region geometry and returns sectional masses.

.. code-block:: python

        study.solution.add_step(IxgenPlugin, "ixgen", {
            "ixgen_path": IXGEN_INPUT_PATH,
            "ixgen_exe": IXGEN_EXE_PATH,
            "step_size": 0.1,
            "input_dict": {
                "region.1.spanwise.start": {"type": "float", "default_value": 0.0},
                "region.2.mainsparweb.layer_1.thickness": {
                    "type": "float",
                    "default_value": 0.0,
                },
                "span": {"type": "float", "default_value": 0.0},
                "region.1.chord.start": {"type": "float", "default_value": 0.0},
                "region.2.twist.stop": {"type": "float", "default_value": 0.0},
            },
            "output_request_dict": {
                "vabs.region.1.inboard.mass": {"type": "float", "default_value": 0.0},
                "vabs.region.1.outboard.mass": {"type": "float", "default_value": 0.0},
                "vabs.region.2.inboard.mass": {"type": "float", "default_value": 0.0},
                "vabs.region.2.outboard.mass": {"type": "float", "default_value": 0.0},
            },
        })
        study.solution.add_link("vehicles.quad.region1.spanwise_start", "ixgen.region.1.spanwise.start")
        study.solution.add_link("lengthMETERS.Rotor(1).radius.meters", "ixgen.span")
        study.solution.add_link("CrossSectionChord.root_chord", "ixgen.region.1.chord.start")
        study.solution.add_link("CrossSectionTwist.tip_twist", "ixgen.region.2.twist.stop")

The remaining steps convert IXGEN masses, sum them, compute NDARC
single-blade weight, and track the residual.

.. code-block:: python

        study.solution.add_step(kg_m2lbs_ft, "massLBS", {"input_dict": {}})
        study.solution.add_link("ixgen.vabs.region.1.inboard.mass", "massLBS.region.1.inboard.mass")

        study.solution.add_step(CalcTotalMassFromCrossSection, "BladeMass", {})
        study.solution.add_step(CalcSingleBladeWeightNdarc, "Rotor1BladeWeight", {})
        study.solution.add_link(
            "ndarc.soln.Rotor(1).WEIGHT.W_ROTOR_BLADE",
            "Rotor1BladeWeight.rotors_weight",
        )
        study.solution.add_link("ndarc.soln.Rotor(1).NBLADE", "Rotor1BladeWeight.nblade")

        study.solution.add_step(ResidualIxgenDemo, "residual", {})
        study.solution.add_link("BladeMass.total_mass", "residual.ixgen_weight")
        study.solution.add_link("Rotor1BladeWeight.single_blade_weight", "residual.ndarc_weight")
        study.solution.add_tracked_output("residual.residual")

The trade study varies rotor velocity.

.. code-block:: python

    def setup_tradestudy(study: Study):
        vel_trade = ParameterSweep({
            "vehicles.quad.rotor.vtip": [450, 500, 550, 600, 650, 700],
        })
        study.control.add_tradestudy("vel_trade", vel_trade)

``run_analysis.py``
-------------------

The analysis script
loads the ``quad`` variant, runs ``vel_trade``, and saves results.

.. code-block:: python

    if __name__ == "__main__":
        variant_path = os.path.join(VARIANTS_DIR, "quad")
        study = Study.from_json(variant_path)
        engine = Engine(study, os.path.join(VARIANTS_DIR, "quad", "work"))
        engine.run_tradestudy("vel_trade")
        study.save_results_to_json(variant_path)
