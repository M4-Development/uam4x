uam4x Quad
==========

First, check out the example repository:

.. code-block:: bash

    git clone git@github.com:M4-Development/uam4x_quad.git
    cd uam4x_quad

You can also use HTTPS:

.. code-block:: bash

    git clone https://github.com/M4-Development/uam4x_quad.git
    cd uam4x_quad

The ``uam4x_quad`` example demonstrates variant composition for a
quad-rotor workflow. It sweeps rotor tip speed, runs NDARC to compute
rotor radius, and sends that radius to OpenVSP through a user parameter.

Run
---

.. code-block:: bash

    cd uam4x_examples/uam4x_quad
    python branching_script.py
    python quad_rotor_analysis.py

Dependencies
------------

This example uses ``uam4x_ndarc``, ``uam4x_openvsp``, ``openvsp``, and an
NDARC executable configured in ``branching_script.py``.

``branching_script.py``
-----------------------

The base model stores rotor radius and tip speed.

.. code-block:: python

    def setup_turbo_electric_model(study: Study):
        mtr = Vehicle("mtr")
        rotor = Rotor("rotor")
        rotor.set_parameter("radius", 18.22473 / 2)
        rotor.set_parameter("vtip", 550.0)
        mtr.add_component(rotor)
        study.add_vehicle(mtr)

The default solution connects rotor tip speed into four NDARC rotor
inputs, then connects the NDARC radius output into OpenVSP.

.. code-block:: python

    from uam4x.solution import Solution

    def setup_solution(study: Study):
        solution = Solution("basic")
        solution.add_step(NdarcPlugin, "ndarc", {
            "njob_path": NJOB_PATH,
            "ndarc_exe": NDARC_EXE_PATH,
            "input_dict": {
                "ROTOR(1).Vtip_ref": {"type": "float", "default_value": 0.0},
                "ROTOR(2).Vtip_ref": {"type": "float", "default_value": 0.0},
                "ROTOR(3).Vtip_ref": {"type": "float", "default_value": 0.0},
                "ROTOR(4).Vtip_ref": {"type": "float", "default_value": 0.0},
            },
            "output_request_dict": {
                "soln.Rotor(1).RADIUS": {"type": "float", "default_value": 0.0},
            },
        })

        solution.add_step(OpenvspPlugin, "openvsp", {
            "base_vsp3_path": VSP_FILE_PATH,
            "input_dict": {
                "UserParm.Default_Group.rotor_radius": {
                    "type": "float",
                    "default_value": 0.0,
                },
            },
            "output_request_dict": {
                "Total_Wet_Area": {"type": "float", "default_value": 0.0},
                "Total_Wet_Vol": {"type": "float", "default_value": 0.0},
            },
        })

        solution.add_link("vehicles.mtr.rotor.vtip", "ndarc.ROTOR(1).Vtip_ref")
        solution.add_link("vehicles.mtr.rotor.vtip", "ndarc.ROTOR(2).Vtip_ref")
        solution.add_link("vehicles.mtr.rotor.vtip", "ndarc.ROTOR(3).Vtip_ref")
        solution.add_link("vehicles.mtr.rotor.vtip", "ndarc.ROTOR(4).Vtip_ref")
        solution.add_link(
            "ndarc.soln.Rotor(1).RADIUS",
            "openvsp.UserParm.Default_Group.rotor_radius",
        )
        solution.add_tracked_output("ndarc.soln.Rotor(1).RADIUS")
        study.add_solution(solution)

Variant modifiers replace one step while preserving the workflow shape.

.. code-block:: python

    def setup_electric_model(study: Study):
        study.get_solution("basic").add_step(NdarcPlugin, "ndarc", {
            "njob_path": ELECTRIC_NJOB_PATH,
            "ndarc_exe": NDARC_EXE_PATH,
            "input_dict": {
                "ROTOR(1).Vtip_ref": {"type": "float", "default_value": 0.0},
                "ROTOR(2).Vtip_ref": {"type": "float", "default_value": 0.0},
                "ROTOR(3).Vtip_ref": {"type": "float", "default_value": 0.0},
                "ROTOR(4).Vtip_ref": {"type": "float", "default_value": 0.0},
            },
            "output_request_dict": {
                "soln.Rotor(1).RADIUS": {"type": "float", "default_value": 0.0},
            },
        })

    def setup_at_sea_model(study: Study):
        study.get_solution("basic").add_step(OpenvspPlugin, "openvsp", {
            "base_vsp3_path": VSP_AT_SEA_FILE_PATH,
            "input_dict": {
                "UserParm.Default_Group.rotor_radius": {
                    "type": "float",
                    "default_value": 0.0,
                },
            },
            "output_request_dict": {
                "Total_Wet_Area": {"type": "float", "default_value": 0.0},
                "Total_Wet_Vol": {"type": "float", "default_value": 0.0},
            },
        })

The trade study sweeps rotor tip speed.

.. code-block:: python

    def setup_vtip_tradestudy(study: Study):
        vtip_trade = ParameterSweep({
            "vehicles.mtr.rotor.vtip": [450, 500, 550, 600, 650, 700],
        })
        study.control.add_tradestudy("vtip_trade", vtip_trade)

The entry point composes four variants. Demonstrating the reuse of functions
for multiple studies and the variant paradigm.

.. code-block:: python

    if __name__ == "__main__":
        turbo_electric_variant = [
            setup_vtip_tradestudy,
            setup_solution,
            setup_turbo_electric_model,
        ]
        electric_variant = [
            setup_vtip_tradestudy,
            setup_solution,
            setup_turbo_electric_model,
            setup_electric_model,
        ]
        turbo_electric_at_sea_variant = [
            setup_vtip_tradestudy,
            setup_solution,
            setup_turbo_electric_model,
            setup_at_sea_model,
        ]
        electric_at_sea_variant = [
            setup_vtip_tradestudy,
            setup_solution,
            setup_turbo_electric_model,
            setup_electric_model,
            setup_at_sea_model,
        ]

        framework = Variants(THIS_DIR)
        framework.add_variant("turbo_electric_variant", turbo_electric_variant)
        framework.add_variant("electric_variant", electric_variant)
        framework.add_variant("turbo_electric_at_sea_variant", turbo_electric_at_sea_variant)
        framework.add_variant("electric_at_sea_variant", electric_at_sea_variant)
        studies = framework.generate_variants()

``quad_rotor_analysis.py``
--------------------------
The various variants are then run through the analysis script,
which iterates active variants, runs the trade study, and saves results.

.. code-block:: python

    ACTIVE_VARIANTS = [
        "turbo_electric_variant",
        "electric_variant",
        "turbo_electric_at_sea_variant",
        "electric_at_sea_variant",
    ]

    if __name__ == "__main__":
        for active_variant in ACTIVE_VARIANTS:
            work_dir = os.path.join(VARIANTS_DIR, active_variant, "work")
            variant_path = os.path.join(VARIANTS_DIR, active_variant)
            study = Study.from_json(variant_path)
            engine = Engine(study, work_dir)
            engine.run_tradestudy("vtip_trade", solution_name="basic")
            study.save_results_to_json(variant_path)
