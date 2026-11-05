from uam4x.solution import Solution
from uam4x.control import ParameterSweep, Optimization
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Wing
from uam4x.study import Study, Variants

from uam4x_openvsp.plugin import VSPUpdatePluginGenerator
from custom import WeightingFactor, CalcObjective
from uam4x_general.plugins import ExpressionPluginGenerator
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VSP_FILE_PATH = os.path.join(THIS_DIR, "NASA_Altered_Multi-Tiltrotor.vsp3")

#Constant Names
VEH_ASPECT_RATIO="vehicles.multi_objective_vehicle.wing1.aspect_ratio"
VEH_SPAN="vehicles.multi_objective_vehicle.wing1.span"
VEH_WEIGHTING_FACTOR="vehicles.multi_objective_vehicle.weighting_factor.weighting_factor"

def setup_model(study:Study):
    veh1 = Vehicle('multi_objective_vehicle')
    wing1 = Wing('wing1')
    wing1.set_parameter('span',100.0)
    factor = WeightingFactor("weighting_factor")
    veh1.add_component(wing1)
    veh1.add_component(factor)
    study.add_vehicle(veh1)


def setup_solution(study:Study):
    #define non visual solution
    define_base_calc(study,name='no_vis')

    #define visual solution
    sol=define_base_calc(study,name='vis')
    define_visual(sol)


def define_base_calc(study:Study,name):

    solution = Solution(name)

    # Define area calculation plugin, analytical partials autocalculated using Sympy
    area_calc=ExpressionPluginGenerator('calc_area',solution=solution)
    area_calc.add_float_input('aspect_ratio')
    area_calc.add_float_input('span')
    area_calc.add_float_output('area',expr='span**2/aspect_ratio',tracked=True)

    #Add plugin to study
    area_calc.add_to_study()

    #Define links
    solution.add_link(VEH_ASPECT_RATIO,area_calc.input('aspect_ratio'))
    solution.add_link(VEH_SPAN,area_calc.input('span'))

    # Define and add objective calculation step
    solution.add_step(CalcObjective, "calc_objective")
    # Add links for objective calculation step
    solution.add_link(VEH_ASPECT_RATIO, "calc_objective.aspect_ratio")
    solution.add_link(VEH_SPAN, "calc_objective.span")
    solution.add_link("calc_area.area", "calc_objective.area")
    solution.add_link("vehicles.multi_objective_vehicle.weighting_factor.weighting_factor", "calc_objective.weighting_factor")

    # Define tracked outputs
    solution.add_tracked_output("calc_objective.CD")
    solution.add_tracked_output("calc_objective.wing_weight")
    solution.add_tracked_output("calc_objective.objective")

    study.add_solution(solution)

    return solution

def define_visual(solution):
    # Define and add openVSP visualization step
    vsp_plug=VSPUpdatePluginGenerator(name='openvsp',
                                 solution=solution,
                                 vsp3_path=VSP_FILE_PATH)
    vsp_plug.add_float_input("parm_id.HFZNSRXUBZN")
    vsp_plug.add_float_input("parm_id.YQWBLUDGDMQ")
    vsp_plug.add_to_study()

    solution.add_link(VEH_SPAN,vsp_plug.input("parm_id.HFZNSRXUBZN"))
    solution.add_link("calc_objective.chord",vsp_plug.input("parm_id.YQWBLUDGDMQ"))

def setup_tradestudy(study:Study):

    # initialize trade study

    # Initialize an optimization
    optimization = Optimization()
    # add des vars
    optimization.add_design_var(VEH_ASPECT_RATIO, lower=1., upper=10.)
    optimization.add_design_var(VEH_SPAN, lower=100., upper=1000.)

    # add constraints
    optimization.add_constraint("calc_area.area", lower=49999., upper=50001.)

    # add objectives
    optimization.add_objective('calc_objective.objective')

    # define a parameter sweep
    parameter_sweep = ParameterSweep(
        # parameters to sweep
        {
            VEH_WEIGHTING_FACTOR: [-10, -20, -30, 0, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 40000, 50000,
                                                                100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
        },
        # child sweep or optimization (performed at every point in the value)
        optimization
    )
    # add tradestudy to study
    study.control.add_tradestudy("sweep", parameter_sweep)
    study.control.add_tradestudy("optimization_only", optimization)

def compare_studies(study1, study2):
    # Compare vehicles
    assert len(study1.vehicles) == len(study2.vehicles)
    for veh1, veh2 in zip(study1.vehicles, study2.vehicles):
        if veh1.name != veh2.name:
            return False, f"Vehicle names differ: {veh1.name} and {veh2.name}"
        if len(veh1.components) != len(veh2.components):
            return False, f"Component count mismatch in vehicle {veh1.name}"
        for c1, c2 in zip(veh1.components, veh2.components):
            if c1.name != c2.name or type(c1) != type(c2):
                return False, f"Component mismatch: {c1.name}/{type(c1)} vs {c2.name}/{type(c2)}"
            # Compare parameters
            if set(c1._parameters.keys()) != set(c2._parameters.keys()):
                return False, f"Parameter keys mismatch in component {c1.name}"
            for pname in c1._parameters:
                p1, p2 = c1._parameters[pname], c2._parameters[pname]
                if p1.value != p2.value or p1.unit_type != p2.unit_type or p1.unit != p2.unit:
                    return False, f"Parameter {pname} mismatch in component {c1.name}: {p1.value}/{p1.unit_type}/{p1.unit} vs {p2.value}/{p2.unit_type}/{p2.unit}"
                # Compare modes
            if hasattr(c1, '_modes') and hasattr(c2, '_modes'):
                if set(c1._modes.keys()) != set(c2._modes.keys()):
                    return False, f"Mode keys mismatch in component {c1.name}"
                for mname in c1._modes:
                    m1, m2 = c1._modes[mname], c2._modes[mname]
                    if m1._overrides != m2._overrides:
                        return False, f"Mode override mismatch in {c1.name}.{mname}: {m1._overrides} vs {m2._overrides}"
    # Compare solution
    sol1, sol2 = study1.get_solution('no_vis'), study2.get_solution('no_vis')
    # Compare solution steps robustly (plugin as name)
    if set(sol1.steps.keys()) != set(sol2.steps.keys()):
        return False, f"Step keys mismatch: {sol1.steps.keys()} vs {sol2.steps.keys()}"
    for step_name in sol1.steps:
        s1 = sol1.steps[step_name]
        s2 = sol2.steps.get(step_name)
        if s2 is None:
            return False, f"Step {step_name} missing in loaded study"
        # Compare plugin by name
        plugin1 = s1['plugin']
        plugin2 = s2['plugin']
        name1 = plugin1.__name__ if hasattr(plugin1, '__name__') else str(plugin1)
        name2 = plugin2.__name__ if hasattr(plugin2, '__name__') else str(plugin2)
        if name1 != name2:
            return False, f"Plugin mismatch in step {step_name}: {name1} vs {name2}"
        if s1.get('plugin_options', {}) != s2.get('plugin_options', {}):
            return False, f"Plugin options mismatch in step {step_name}: {s1.get('plugin_options', {})} vs {s2.get('plugin_options', {})}"
    if sol1.tracked_outputs != sol2.tracked_outputs:
        return False, f"Tracked outputs mismatch: {sol1.tracked_outputs} vs {sol2.tracked_outputs}"
    if sol1.links != sol2.links:
        return False, f"Links mismatch: {sol1.links} vs {sol2.links}"

    # Compare control
    ctrl1, ctrl2 = study1.control, study2.control
    if ctrl1.title != ctrl2.title or ctrl1.author != ctrl2.author or ctrl1.description != ctrl2.description:
        return False, f"Control meta mismatch"
    if set(ctrl1.trade_studies.keys()) != set(ctrl2.trade_studies.keys()):
        return False, f"Trade study keys mismatch: {ctrl1.trade_studies.keys()} vs {ctrl2.trade_studies.keys()}"
    # Optionally, add deep comparison for trade_studies if needed

    return True, "Studies are equivalent"

if __name__ == "__main__":
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('demo', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","demo")
    study_from_json = Study.load_study_from_directory(save_path)

    # Compare the original and loaded study at the value level
    equal, message = compare_studies(study, study_from_json)
    if equal:
        print("SUCCESS: The original and loaded Study are equivalent at the value level.")
    else:
        print("ERROR:", message)
