import sys
from pathlib import Path
import os
import math
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine

from conf import (TURBO_ELECTRIC,TURBO_ELECTRIC_AT_SEA,
                  ELECTRIC,ELECTRIC_AT_SEA,
                  VARIANT_OPTIONS,NDARC_EXE_PATH)
from branching_script import  setup_vtip_tradestudy, setup_solution, setup_model


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")
ACTIVE_VARIANTS = [TURBO_ELECTRIC, TURBO_ELECTRIC_AT_SEA,
                    ELECTRIC, ELECTRIC_AT_SEA]

def test_quad_rotor():

    func_list = [setup_vtip_tradestudy, setup_solution,setup_model ]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant(TURBO_ELECTRIC,
                          func_list,
                          options=VARIANT_OPTIONS[TURBO_ELECTRIC])
    framework.add_variant(ELECTRIC,
                          func_list,
                          options=VARIANT_OPTIONS[ELECTRIC])
    framework.add_variant(TURBO_ELECTRIC_AT_SEA,
                          func_list,
                          options=VARIANT_OPTIONS[TURBO_ELECTRIC_AT_SEA])
    framework.add_variant(ELECTRIC_AT_SEA,
                          func_list,
                          options=VARIANT_OPTIONS[ELECTRIC_AT_SEA])
    framework.generate_variants()

    # run all variants
    variant_to_json = {}
    for active_variant in ACTIVE_VARIANTS:
        work_dir = os.path.join(VARIANTS_DIR, active_variant, "work")
        variant_path = os.path.join(VARIANTS_DIR, active_variant)
        variant_to_json[active_variant] = variant_path

        study = Study.load_study_from_directory(variant_path)
        engine = Engine(study, work_dir)
        engine.run_tradestudy("vtip_trade","basic")
        study.save_results_to_json(variant_path)

    # regression expectations (golden outputs)
    VARIANT_CASES = {
        TURBO_ELECTRIC: {
            "vehicles.mtr.rotor.vtip": [450, 500, 550, 600, 650, 700],
            "ndarc.soln.Rotor(1).RADIUS": [8.533422, 8.340016, 8.203928, 8.11678, 8.049459, 7.853601],
            "openvsp.analysis.Total_Wet_Area": [1629.941016, 1576.965112, 1546.318748, 1521.529346, 1503.835126, 1460.571266],
            "openvsp.analysis.Total_Wet_Vol": [576.796595, 575.010508, 574.070823, 573.484937, 573.217027, 571.865181],
        },
        TURBO_ELECTRIC_AT_SEA: {
            "vehicles.mtr.rotor.vtip": [450, 500, 550, 600, 650, 700],
            "ndarc.soln.Rotor(1).RADIUS": [8.533422, 8.340016, 8.203928, 8.11678, 8.049459, 7.853601],
            "openvsp.analysis.Total_Wet_Area": [1648.752978, 1599.854706, 1562.065464, 1541.736089, 1524.575694, 1474.225577],
            "openvsp.analysis.Total_Wet_Vol": [580.241118, 578.562352, 577.929055, 577.316303, 577.006674, 576.362245],
        },
        # intended failures → should not contain successful NDARC results
        ELECTRIC: None,
        ELECTRIC_AT_SEA: None,
    }

    # tolerances
    vtip_tol = 1e-3
    radius_tol = 1e-3
    openvsp_rel_tol = 0.01

    for variant, save_path in variant_to_json.items():
        study = Study.load_study_from_directory(save_path)
        results = study.results
        soln_results=results.get_solution_result("basic")
        trade_results = soln_results.get_tradestudy("vtip_trade")

        expected = VARIANT_CASES[variant]

        # Variants that are intended to fail (no valid NDARC outputs)
        if expected is None:
            assert "vtip_trade" in soln_results.tradestudies, f"{variant}: missing vtip_trade results"
            assert len(trade_results._results) > 0, f"{variant}: expected failure results"
            for result in trade_results._results:
                assert "Failure Message" in result.state, f"{variant}: missing Failure Message"
            continue

        # Successful variants: validate per-sweep-point outputs
        expected_vtips = expected["vehicles.mtr.rotor.vtip"]
        expected_radii = expected["ndarc.soln.Rotor(1).RADIUS"]
        expected_areas = expected["openvsp.analysis.Total_Wet_Area"]
        expected_volumes = expected["openvsp.analysis.Total_Wet_Vol"]

        vtip_values = trade_results.get_variable("vehicles.mtr.rotor.vtip")
        radius_values = trade_results.get_variable("ndarc.soln.Rotor(1).RADIUS")
        area_values = trade_results.get_variable("openvsp.analysis.Total_Wet_Area")
        volume_values = trade_results.get_variable("openvsp.analysis.Total_Wet_Vol")

        assert len(vtip_values) == len(expected_vtips), (
            f"{variant}: expected {len(expected_vtips)} results, got {len(vtip_values)}"
        )
        assert len(radius_values) == len(area_values) == len(volume_values) == len(expected_vtips), (
            f"{variant}: incomplete NDARC or OpenVSP probe results"
        )

        for i in range(len(expected_vtips)):
            assert vtip_values[i] is not None, f"{variant}: missing vehicles.mtr.rotor.vtip at i={i}"
            assert radius_values[i] is not None, f"{variant}: missing ndarc.soln.Rotor(1).RADIUS at i={i}"
            assert area_values[i] is not None, f"{variant}: missing openvsp.analysis.Total_Wet_Area at i={i}"
            assert volume_values[i] is not None, f"{variant}: missing openvsp.analysis.Total_Wet_Vol at i={i}"

            assert math.isclose(vtip_values[i], expected_vtips[i], abs_tol=vtip_tol), (
                f"{variant}: vtip mismatch at i={i}. got {vtip_values[i]} expected {expected_vtips[i]}"
            )

            assert math.isclose(radius_values[i], expected_radii[i], abs_tol=radius_tol), (
                f"{variant}: radius mismatch at i={i}. got {radius_values[i]} expected {expected_radii[i]}"
            )
            assert math.isclose(area_values[i], expected_areas[i], rel_tol=openvsp_rel_tol, abs_tol=0.0), (
                f"{variant}: wet area mismatch at i={i}. got {area_values[i]} expected {expected_areas[i]}"
            )
            assert math.isclose(volume_values[i], expected_volumes[i], rel_tol=openvsp_rel_tol, abs_tol=0.0), (
                f"{variant}: wet volume mismatch at i={i}. got {volume_values[i]} expected {expected_volumes[i]}"
            )
