import pytest
from uam4x.components import Wing, Rotor, Aircraft,  Parameter, get_commponent_class
from uam4x.vehicle import Vehicle
from uam4x.study import Study, Variants
import os
from uam4x.execution_engine import Engine
from uam4x.solution import Plugin, register_plugin
from uam4x.utils import get_param_type


def create_sample_problem(study: Study):
    """
    sets up a sample problem for testing

    Parameters
    ----------
    study : Study
        The study to the sample problem to
    """
    # vehicle
    mtr = Vehicle("mtr")

    # wing
    wing = Wing("wing")
    wing.set_parameter('span', 35.56)
    wing.set_parameter('area', 142.1)
    wing.set_parameter('sweep', 0.5)

    # aircraft
    aircraft = Aircraft('aircraft')
    aircraft.set_parameter('npass', 6)
    aircraft.set_parameter('wpass', 200.)

    # rotor
    rotor = Rotor("rotor")
    rotor.set_parameter("nblade", 8.0)
    vflight_rotor = rotor.add_mode('vflight')
    vflight_rotor.set_parameter('tilt', 90.)
    cruise_rotor = rotor.add_mode('cruise')
    cruise_rotor.set_parameter('tilt', 0.)

    # adds components to vehicle/study
    mtr.add_component(wing)
    mtr.add_component(rotor)
    mtr.add_component(aircraft)
    study.add_vehicle(mtr)

def setup_solution(study:Study):
    """
    sets up the solution sequence for calculating the test

    Parameters
    ----------
    study : Study
        The study to add the solution to
    """

    # Define and add test plugin
    study.solution.add_step(TestPlug,"test_plug",{})

    # Add links for test plugin
    study.solution.add_link("vehicles.mtr.wing.span", "test_plug.testin_cont")
    study.solution.add_link("vehicles.mtr.aircraft.npass", "test_plug.testin_discrete")

@register_plugin
class TestPlug(Plugin):
    """
    test plugin to check get_param_type functionality for plugin links/components

    """
    def setup_plugin(self, plugin_options):
        # declare inputs
        Plugin.declare_input(self,"testin_cont", float, 0.0)
        Plugin.declare_input(self,"testin_discrete", int, 1)

        # declare outputs
        Plugin.declare_output(self,"testout_cont", float, 0.0)
        Plugin.declare_output(self,"testout_discrete", int, 1)

    def run_plugin(self, links:dict, work_dir:str):
        results = {}
        results['testout_cont'] = 0.06
        results['testout_discrete'] = 6
        return results


def test_get_param_type(study:Study):
    """
    sets up a sample problem for testing

    Parameters
    ----------
    study : Study
        The study to the sample problem to
    """

    study = Study.from_json(variant_path)
    eng = Engine(study, os.path.join(VARIANTS_DIR, "test_variants", "work"))
    plugins = {}
    for step_name, step_info in study.solution.steps.items():
            plugin = step_info["plugin"]()
            plugin.setup_plugin(step_info["plugin_options"])
            plugins[step_name] = plugin

    # check get_param_type functionality
    assert get_param_type('vehicles.mtr.wing.span',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.wing.area',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.wing.sweep',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.aircraft.wpass',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.aircraft.npass',study, plugins) == int, f"expected int, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.rotor.nblade',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.rotor.vflight.tilt',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('vehicles.mtr.rotor.cruise.tilt',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('test_plug.testin_cont',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('test_plug.testin_discrete',study, plugins) == int, f"expected int, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('test_plug.testout_cont',study, plugins) == float, f"expected float, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"
    assert get_param_type('test_plug.testout_discrete',study, plugins) == int, f"expected int, got {eng.get_param_type('vehicles.mtr.wing.span',plugins)}"



if __name__ == '__main__':

    # setup problem
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")
    turbo_electric_variant = [create_sample_problem, setup_solution]
    framework = Variants(THIS_DIR)
    framework.add_variant('test_variants', turbo_electric_variant)
    studies = framework.generate_variants()
    variant_path = os.path.join(VARIANTS_DIR, "test_variants", "data.json")
    study = Study.from_json(variant_path)

    # run test
    test_get_param_type(study)