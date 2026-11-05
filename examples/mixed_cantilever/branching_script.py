from uam4x.solution import Solution
from uam4x.UQ import NormalDistribution,BasicMixedIVP,MixedCSVPoints,\
UniformDistribution, LatinHypercubeSampler,HaltonSampler,MixedRegressionPCE
from uam4x.vehicle import Vehicle
from uam4x.study import Study,Variants
from custom import CantileverBase,CantiliverDeflection
import numpy as np
import os
from math import comb

def setup_model(study:Study):

    vehicle=Vehicle('base')
    cb=CantileverBase('cantilever_base')
    vehicle.add_component(cb)
    study.add_vehicle(vehicle)

def setup_solution(study:Study):

    solution = Solution("basic")
    solution.add_step(CantiliverDeflection,'cantilever_deflection')
    solution.add_link("vehicles.base.cantilever_base.M","cantilever_deflection.M")
    solution.add_link("vehicles.base.cantilever_base.Y","cantilever_deflection.Y")
    solution.add_link("vehicles.base.cantilever_base.T","cantilever_deflection.T")
    solution.add_link("vehicles.base.cantilever_base.b","cantilever_deflection.b")
    solution.add_link("vehicles.base.cantilever_base.h","cantilever_deflection.h")
    solution.add_link("vehicles.base.cantilever_base.P","cantilever_deflection.P")

    solution.add_tracked_output("cantilever_deflection.g")
    study.add_solution(solution)


def setup_tradestudy(study:Study):

    ivp=BasicMixedIVP(50,LatinHypercubeSampler,
                      1000,HaltonSampler,seed=42)
    ivp.add_aleatory_variable("vehicles.base.cantilever_base.M",NormalDistribution(mean=2500.,std=200.))
    ivp.add_aleatory_variable("vehicles.base.cantilever_base.Y",UniformDistribution(low=2000.,high=2800.))
    ivp.add_epistemic_variable("vehicles.base.cantilever_base.T",np.array([0.9,1.1]))

    study.control.add_tradestudy('mc_cantilever_problem',ivp)

    num_samples=2*comb(3+2,2)
    ivp=MixedRegressionPCE(LatinHypercubeSampler,num_samples=num_samples,seed=42)
    ivp.add_aleatory_variable("vehicles.base.cantilever_base.M",NormalDistribution(mean=2500.,std=200.))
    ivp.add_aleatory_variable("vehicles.base.cantilever_base.Y",UniformDistribution(low=2000.,high=2800.))
    ivp.add_epistemic_variable("vehicles.base.cantilever_base.T",np.array([0.9,1.1]))

    study.control.add_tradestudy('pce_cantilever_problem',ivp)


    csv_file=os.path.join(THIS_DIR,'m4_ref','samples.csv')
    ivp=MixedCSVPoints(csv_file)
    ivp.add_aleatory_variable("vehicles.base.cantilever_base.M",NormalDistribution(mean=2500.,std=200.))
    ivp.add_aleatory_variable("vehicles.base.cantilever_base.Y",UniformDistribution(low=2000.,high=2800.))
    ivp.add_epistemic_variable("vehicles.base.cantilever_base.T",np.array([0.9,1.1]))
    study.control.add_tradestudy('m4_samples',ivp)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    mixed_prob=[setup_model,setup_solution,setup_tradestudy]


    framework=Variants(THIS_DIR)
    framework.add_variant("mixed_cantilever", mixed_prob)
    study = framework.generate_variants()[0]

    save_path = os.path.join(THIS_DIR, "variants","mixed_cantilever")
    study_from_json = Study.load_study_from_directory(save_path)
