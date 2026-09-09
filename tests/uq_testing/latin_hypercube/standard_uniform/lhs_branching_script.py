from uam4x.UQ import LatinHypercubeSampler, UniformDistribution,BasicAleatory
from uam4x.vehicle import Vehicle
from uam4x.study import Study,Variants
from custom import Point,PointAnalysis
from uam4x.solution import Group, Solution
import numpy as np
import os
from math import comb
def setup_model(study:Study):

    vehicle=Vehicle('point_vehicle')
    b=Point('point_base')
    vehicle.add_component(b)
    study.add_vehicle(vehicle)

def setup_solution(study:Study):
    solution = Solution("basic")

    solution.add_step(PointAnalysis,"point_calc")

    solution.add_link('vehicles.point_vehicle.point_base.x','point_calc.x')

    solution.add_tracked_output('point_calc.x_out')
    study.add_solution(solution)


def setup_tradestudy(study:Study):

    uq=BasicAleatory(100000,SamplingClass=LatinHypercubeSampler,seed=42)
    uq.add_aleatory_variable('vehicles.point_vehicle.point_base.x',UniformDistribution(low=0.0,high=1.0))
    study.control.add_tradestudy('standard_uniform',uq)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    standard_uniform=[setup_model,setup_solution,setup_tradestudy]

    framework=Variants(THIS_DIR)
    framework.add_variant('standard_uniform', standard_uniform)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","standard_uniform")
    study_from_json = Study.load_study_from_directory(save_path)
