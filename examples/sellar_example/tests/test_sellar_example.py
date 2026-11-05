import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from branching_script import setup_model,setup_solution,setup_tradestudy
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_sellar():
    #run branching:
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('sellar', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants",'sellar')


    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path,"work"))
    engine.run_tradestudy("sellar_problem", solution_name="sellar_basic")

    solution_results=study.results.get_solution_result("sellar_basic")
    trade=solution_results.get_tradestudy("sellar_problem")

    #reference data:
    ref_z0=1.9776388839296362
    ref_z1=0.0
    ref_z=[1.9776388839296362, 0.0]
    ref_x=0.0
    ref_obj=3.183393953277392
    ref_con1=0.0
    ref_con2=-20.24472223214072

    #current data:
    z0=trade.get_variable('vehicles.sellar_vehicle.sellar_base.z[0]')
    z1=trade.get_variable('vehicles.sellar_vehicle.sellar_base.z[1]')
    z=trade.get_variable('vehicles.sellar_vehicle.sellar_base.z')
    x=trade.get_variable('vehicles.sellar_vehicle.sellar_base.x')
    obj=trade.get_variable('sellar_objective.obj')
    con1=trade.get_variable('sellar_objective.con1')
    con2=trade.get_variable('sellar_objective.con2')

    assert np.isclose(z0,ref_z0,10**-8)
    assert np.isclose(z1,ref_z1,10**-8)
    assert np.allclose(z,ref_z,10**-8)
    assert np.isclose(x,ref_x,10**-8)
    assert np.isclose(obj,ref_obj,10**-8)
    assert np.isclose(con1,ref_con1,10**-8)
    assert np.isclose(con2,ref_con2,10**-8)


if __name__=="__main__":
    test_sellar()
