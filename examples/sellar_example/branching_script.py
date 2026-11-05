import numpy as np
import os

from uam4x.control import Optimization
from uam4x.vehicle import Vehicle
from uam4x.study import Study,Variants
from uam4x.solution import Group, Solution

from custom import SellarBase,SellarDis1,SellarDis2,SellarObjective


def setup_model(study:Study):

    vehicle=Vehicle('sellar_vehicle')
    sb=SellarBase('sellar_base')
    sb.set_parameter('x',1.)
    sb.set_parameter('z',np.array([5.0,2.0]))
    vehicle.add_component(sb)
    study.add_vehicle(vehicle)

def setup_solution(study:Study):
    sellar_basic=Solution('sellar_basic')

    sellar_cycle=Group()
    sellar_cycle.add_step(SellarDis1,'sellardis1')
    sellar_cycle.add_step(SellarDis2,'sellardis2')
    sellar_cycle.declare_cycle_step_order(['sellardis1','sellardis2'])
    sellar_cycle.set_cycle_solver("NonlinearBlockGS")

    sellar_basic.add_step(sellar_cycle,'sellar_cycle')

    #internal links
    sellar_cycle.add_link('sellardis1.y1','sellardis2.y1')
    sellar_cycle.add_link('sellardis2.y2','sellardis1.y2')


    #external links

    #Sellar base--> Sellar dis1
    sellar_basic.add_link('vehicles.sellar_vehicle.sellar_base.x','sellar_cycle.sellardis1.x')
    sellar_basic.add_link('vehicles.sellar_vehicle.sellar_base.z','sellar_cycle.sellardis1.z')

    sellar_basic.add_link('vehicles.sellar_vehicle.sellar_base.z','sellar_cycle.sellardis2.z')

    sellar_basic.add_step(SellarObjective,'sellar_objective')

    sellar_basic.add_link('vehicles.sellar_vehicle.sellar_base.x','sellar_objective.x')
    sellar_basic.add_link('sellar_cycle.sellardis1.y1','sellar_objective.y1')
    sellar_basic.add_link('sellar_cycle.sellardis2.y2','sellar_objective.y2')
    sellar_basic.add_link('vehicles.sellar_vehicle.sellar_base.z','sellar_objective.z')

    sellar_basic.add_tracked_output('sellar_objective.obj')
    sellar_basic.add_tracked_output('sellar_objective.con1')
    sellar_basic.add_tracked_output('sellar_objective.con2')
    sellar_basic.add_tracked_output('vehicles.sellar_vehicle.sellar_base.z[1]')

    study.add_solution(sellar_basic)


def setup_tradestudy(study:Study):

    optimization=Optimization()
    optimization.add_optimizer_option('ACC',1e-8)
    optimization.set_driver("pyOptSparseDriver")
    optimization.add_design_var('vehicles.sellar_vehicle.sellar_base.x',lower=0.0,upper=10.)
    optimization.add_design_var('vehicles.sellar_vehicle.sellar_base.z',lower=np.array([0.,0.]),upper=np.array([10.,10.]))

    optimization.add_constraint('sellar_objective.con1',upper=0.)
    optimization.add_constraint('sellar_objective.con2',upper=0.)

    optimization.add_objective('sellar_objective.obj')

    study.control.add_tradestudy('sellar_problem',optimization)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    sellar=[setup_model,setup_solution,setup_tradestudy]

    framework=Variants(THIS_DIR)
    framework.add_variant('sellar', sellar)
    study = framework.generate_variants()[0]

    save_path = os.path.join(THIS_DIR, "variants","sellar")
    study_from_json = Study.load_study_from_directory(save_path)