from uam4x.solution import Solution
from uam4x.vehicle import Vehicle
from uam4x.study import Study, Variants
from uam4x.UQ import UniformDistribution, NormalDistribution, RegressionPCE,LatinHypercubeSampler,\
    MonteCarloSampler,GaussQuadraturePCE
from custom import BeamProblem
from uam4x_general.plugins import ExpressionPluginGenerator
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_model(study: Study):
    veh1=Vehicle('vehicle1')
    beam=BeamProblem('beam1')
    veh1.add_component(beam)
    beam.set_parameter('width',1.3)
    beam.set_parameter('height',2.0)
    beam.set_parameter('axial_load',500.0)
    beam.set_parameter('bending_moment',1000.0)
    beam.set_parameter('yield_strength',1000.0)
    study.add_vehicle(veh1)

def setup_solution(study:Study):

    solution = Solution("basic")
    failure = ExpressionPluginGenerator("calculate_failure", solution=solution)
    for name in ("height", "width", "axial_load", "bending_moment", "yield_strength"):
        failure.add_float_input(name)
    failure.add_float_output(
        "g_val",
        expr=(
            "1-4*bending_moment/(height**2*width*yield_strength)"
            "-(axial_load/(height*width*yield_strength))**2"
        ),
        tracked=True,
    )
    failure.add_to_study()
    for name in ("height", "width", "axial_load", "bending_moment", "yield_strength"):
        solution.add_link(f"vehicles.vehicle1.beam1.{name}", failure.input(name))
    study.add_solution(solution)

def setup_tradestudy(study:Study):
    # The fifth-order, two-variable surrogate has 21 basis terms.
    lhs_reg_pce_analysis = RegressionPCE(42, LatinHypercubeSampler, seed=0)
    lhs_reg_pce_analysis.add_aleatory_variable("vehicles.vehicle1.beam1.bending_moment", NormalDistribution(mean=2500., std=200.0))
    lhs_reg_pce_analysis.add_aleatory_variable("vehicles.vehicle1.beam1.yield_strength", UniformDistribution(low=2000., high=2800.))
    study.control.add_tradestudy("lhs_pce_analysis", lhs_reg_pce_analysis)

    mc_reg_pce_analysis = RegressionPCE(
        5000, MonteCarloSampler, seed=0,
        hold_out_fraction=0.3, hold_out_seed=42,
    )
    mc_reg_pce_analysis.add_aleatory_variable("vehicles.vehicle1.beam1.bending_moment", NormalDistribution(mean=2500., std=200.0))
    mc_reg_pce_analysis.add_aleatory_variable("vehicles.vehicle1.beam1.yield_strength", UniformDistribution(low=2000., high=2800.))
    study.control.add_tradestudy("mc_pce_analysis", mc_reg_pce_analysis)

    quad_pce_analysis=GaussQuadraturePCE(poly_order=5)
    quad_pce_analysis.add_aleatory_variable("vehicles.vehicle1.beam1.bending_moment", NormalDistribution(mean=2500., std=200.0))
    quad_pce_analysis.add_aleatory_variable("vehicles.vehicle1.beam1.yield_strength", UniformDistribution(low=2000., high=2800.))
    study.control.add_tradestudy("quad_pce_analysis", quad_pce_analysis)

if __name__ == "__main__":
    nipc_uq=[setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework=Variants(THIS_DIR)
    framework.add_variant("nipc_uq", nipc_uq)
    study=framework.generate_variants()[0]
    save_path=os.path.join(THIS_DIR,"variants","nipc_uq")
    study_from_json=Study.load_study_from_directory(save_path)
