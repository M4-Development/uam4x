import numpy as np
import os
from math import comb

from uam4x.solution import Solution
from uam4x.UQ import UniformDistribution, SobolSensitivity,SobolSampler,\
LatinHypercubeSampler,RegressionPCE
from uam4x.study import Study,Variants,Vehicle
from custom import Ishigami
from uam4x_general.plugins import ExpressionPluginGenerator

def setup_model(study:Study):

    vehicle=Vehicle('ishigami_vehicle')
    b=Ishigami('ishigami_base')
    vehicle.add_component(b)
    study.add_vehicle(vehicle)

def setup_solution(study:Study):

    solution = Solution("basic")
    ishigami=ExpressionPluginGenerator('ishigami_calc',solution=solution)
    ishigami.add_float_input('a')
    ishigami.add_float_input('b')
    ishigami.add_float_input('X1')
    ishigami.add_float_input('X2')
    ishigami.add_float_input('X3')
    ishigami.add_float_output('g',
                              expr='sin(X1)+a*sin(X2)**2+b*X3**4*sin(X1)',tracked=True)
    ishigami.add_to_study()
    for name in ('a', 'b', 'X1', 'X2', 'X3'):
        solution.add_link(
            f'vehicles.ishigami_vehicle.ishigami_base.{name}',
            ishigami.input(name),
        )
    study.add_solution(solution)

def setup_tradestudy(study:Study):


    # Use three samples per basis term for the seventh-order surrogate.

    mc_uq=SobolSensitivity(5000,SamplingClass=SobolSampler,seed=42)
    mc_uq.add_aleatory_variable('vehicles.ishigami_vehicle.ishigami_base.X1',UniformDistribution(low=-np.pi,high=np.pi))
    mc_uq.add_aleatory_variable('vehicles.ishigami_vehicle.ishigami_base.X2',UniformDistribution(low=-np.pi,high=np.pi))
    mc_uq.add_aleatory_variable('vehicles.ishigami_vehicle.ishigami_base.X3',UniformDistribution(low=-np.pi,high=np.pi))

    study.control.add_tradestudy('mc_ishigami_problem',mc_uq)


    num_samps=3*comb(7+3,7)
    pce_uq=RegressionPCE(SamplingClass=LatinHypercubeSampler,
                         num_samples=num_samps,seed=42)
    pce_uq.add_aleatory_variable('vehicles.ishigami_vehicle.ishigami_base.X1',UniformDistribution(low=-np.pi,high=np.pi))
    pce_uq.add_aleatory_variable('vehicles.ishigami_vehicle.ishigami_base.X2',UniformDistribution(low=-np.pi,high=np.pi))
    pce_uq.add_aleatory_variable('vehicles.ishigami_vehicle.ishigami_base.X3',UniformDistribution(low=-np.pi,high=np.pi))

    study.control.add_tradestudy('pce_ishigami_problem',pce_uq)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    ishigami=[setup_model,setup_solution,setup_tradestudy]

    framework=Variants(THIS_DIR)
    framework.add_variant('ishigami', ishigami)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","ishigami")
    study_from_json = Study.load_study_from_directory(save_path)
