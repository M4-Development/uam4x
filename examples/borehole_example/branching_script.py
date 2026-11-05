from uam4x.solution import Solution
from uam4x.UQ import RegressionPCE, LogNormalDistribution,NormalDistribution,UniformDistribution, LatinHypercubeSampler
from uam4x.vehicle import Vehicle
from uam4x.study import Study,Variants
from custom import Borehole,BoreholeAnalysis

import os
from math import comb
def setup_model(study:Study):

    vehicle=Vehicle('borehole_vehicle')
    b=Borehole('borehole_base')
    vehicle.add_component(b)
    study.add_vehicle(vehicle)

def setup_solution(study:Study):

    solution = Solution("basic")
    solution.add_step(BoreholeAnalysis,"borehole_calc")

    solution.add_link('vehicles.borehole_vehicle.borehole_base.rw','borehole_calc.rw')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.r','borehole_calc.r')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.Tu','borehole_calc.Tu')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.Hu','borehole_calc.Hu')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.Tl','borehole_calc.Tl')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.Hl','borehole_calc.Hl')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.L','borehole_calc.L')
    solution.add_link('vehicles.borehole_vehicle.borehole_base.Kw','borehole_calc.Kw')

    solution.add_tracked_output('borehole_calc.flow_rate')
    study.add_solution(solution)


def setup_tradestudy(study:Study):

    # Allow three samples per basis term for the fifth-order surrogate
    # fitted in run_post.py.

    num_samples=3*comb(5+8,5)
    uq=RegressionPCE(num_samples=num_samples,SamplingClass=LatinHypercubeSampler,seed=42)
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.rw',NormalDistribution(mean=0.1,std=0.0161812))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.r',LogNormalDistribution(mu_log=7.71,sigma_log=1.0056))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.Tu',UniformDistribution(low=63070.,high=115600.))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.Hu',UniformDistribution(low=990.,high=1100.))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.Tl',UniformDistribution(low=63.1,high=116.))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.Hl',UniformDistribution(low=700.,high=820.))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.L',UniformDistribution(low=1120.,high=1680.))
    uq.add_aleatory_variable('vehicles.borehole_vehicle.borehole_base.Kw',UniformDistribution(low=9985.,high=12045.))

    study.control.add_tradestudy('borehole_problem',uq)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    borehole=[setup_model,setup_solution,setup_tradestudy]

    framework=Variants(THIS_DIR)
    framework.add_variant('borehole', borehole)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","borehole")
    study_from_json = Study.load_study_from_directory(save_path)
