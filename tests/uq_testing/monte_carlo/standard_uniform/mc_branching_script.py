from uam4x.UQ import MonteCarloSampler, UniformDistribution,BasicAleatory
from uam4x.vehicle import Vehicle
from uam4x.study import Study,Variants
from custom import Point,PointAnalysis
from uam4x.solution import Group
import numpy as np
import os
from math import comb
def setup_model(study:Study):

    vehicle=Vehicle('point_vehicle')
    b=Point('point_base')
    vehicle.add_component(b)
    study.add_vehicle(vehicle)

def setup_solution(study:Study):

    study.solution.add_step(PointAnalysis,"point_calc")

    study.solution.add_link('vehicles.point_vehicle.point_base.x','point_calc.x')

    study.solution.add_tracked_output('point_calc.x_out')


def setup_tradestudy(study:Study):

    uq=BasicAleatory(100000,SamplingClass=MonteCarloSampler,seed=42)
    uq.add_aleatory_variable('vehicles.point_vehicle.point_base.x',UniformDistribution(low=0.0,high=1.0))
    study.control.add_tradestudy('standard_uniform',uq)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    standard_uniform=[setup_model,setup_solution,setup_tradestudy]

    framework=Variants(THIS_DIR)
    framework.add_variant('standard_uniform', standard_uniform)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","standard_uniform")
    study_from_json = Study.from_json(save_path)