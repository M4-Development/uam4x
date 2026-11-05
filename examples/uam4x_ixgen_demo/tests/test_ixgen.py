import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from branching_script import setup_quad_model, setup_solution, setup_tradestudy

import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_ixgen_demo():
    # run branching
    quad_variant = [setup_quad_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant("quad", quad_variant)
    framework.generate_variants()

    save_path = os.path.join(THIS_DIR, "variants", "quad")
    work_dir = os.path.join(THIS_DIR, "variants", "quad", "work")

    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, work_dir)
    engine.run_tradestudy("vel_trade", solution_name="basic")

    # persist and reload results to validate JSON round-trip behavior
    study.save_results_to_json(save_path)
    study = Study.load_study_from_directory(save_path)
    solution_results = study.results.get_solution_result("basic")
    trade_results = solution_results.get_tradestudy("vel_trade")

    # reference values (4 sweep points)
    vtip_list = [500.0, 550.0, 600.0, 650.0]
    blade_weight_list = [12.032876666666667, 9.61712, 7.9156233333333335, 6.64714]
    residual_r_list = [9.15515460858432e-09, 1.5550862004777711e-09, 3.280773052002639e-08, 1.2904666567022378e-10]
    total_mass_list = [12.032948507241787, 9.61708788055274, 7.915776134185902, 6.647149646231548]
    radius_list = [8.29959, 8.149357, 8.052679, 7.980671]
    tech1_list = [0.8409516052614443, 0.7956874875109123, 0.7555989490776174,0.7203657262745317]
    # tolerances (adjust if you want tighter/looser)
    vtip_tol = 1e-12
    blade_weight_tol = 1e-9
    residual_tol = 1e-12
    mass_tol = 1e-9
    radius_tol = 1e-9
    tech_tol = 1e-7

    vtip_values = trade_results.get_variable("vehicles.quad.rotor.vtip")
    blade_weight_values = trade_results.get_variable("Rotor1BladeWeight.blade_weight")
    residual_values = trade_results.get_variable("residual.R")
    total_mass_values = trade_results.get_variable("massTotallbs.total_mass")
    radius_values = trade_results.get_variable("ndarc.soln.Rotor(1).RADIUS")
    tech1_values = trade_results.get_variable("vehicles.quad.tech.Tech1")

    assert len(vtip_values) == 4, f"Expected 4 results, got {len(vtip_values)}"

    for i in range(4):
        assert math.isclose(
            vtip_values[i],
            vtip_list[i],
            abs_tol=vtip_tol
        ), (
            f"[ixgen_demo] i={i} | vehicles.quad.rotor.vtip mismatch: "
            f"actual={vtip_values[i]}, "
            f"expected={vtip_list[i]}, abs_tol={vtip_tol}"
        )

        assert math.isclose(
            blade_weight_values[i],
            blade_weight_list[i],
            abs_tol=blade_weight_tol
        ), (
            f"[ixgen_demo] i={i} | Rotor1BladeWeight.blade_weight mismatch: "
            f"actual={blade_weight_values[i]}, "
            f"expected={blade_weight_list[i]}, abs_tol={blade_weight_tol}"
        )

        assert math.isclose(
            residual_values[i],
            residual_r_list[i],
            abs_tol=residual_tol
        ), (
            f"[ixgen_demo] i={i} | residual.R mismatch: "
            f"actual={residual_values[i]}, "
            f"expected={residual_r_list[i]}, abs_tol={residual_tol}"
        )

        assert math.isclose(
            total_mass_values[i],
            total_mass_list[i],
            abs_tol=mass_tol
        ), (
            f"[ixgen_demo] i={i} | massTotallbs.total_mass mismatch: "
            f"actual={total_mass_values[i]}, "
            f"expected={total_mass_list[i]}, abs_tol={mass_tol}"
        )

        assert math.isclose(
            radius_values[i],
            radius_list[i],
            abs_tol=radius_tol
        ), (
            f"[ixgen_demo] i={i} | ndarc.soln.Rotor(1).RADIUS mismatch: "
            f"actual={radius_values[i]}, "
            f"expected={radius_list[i]}, abs_tol={radius_tol}"
        )

        assert math.isclose(
            tech1_values[i],
            tech1_list[i],
            abs_tol=tech_tol
        ), (
            f"[ixgen_demo] i={i} | vehicles.quad.tech.Tech1 mismatch: "
            f"actual={tech1_values[i]}, "
            f"expected={tech1_list[i]}, abs_tol={tech_tol}"
        )
