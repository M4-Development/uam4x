import os
from uam4x.study import Study


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "sellar")
    study = Study.load_study_from_directory(study_dir)
    solution_results = study.results.get_solution_result("sellar_basic")
    trade = solution_results.get_tradestudy("sellar_problem")

    out_z=trade.get_variable('vehicles.sellar_vehicle.sellar_base.z')
    out_z0=trade.get_variable('vehicles.sellar_vehicle.sellar_base.z[0]')
    out_z1=trade.get_variable('vehicles.sellar_vehicle.sellar_base.z[1]')

    print(f'The z des var: {out_z}')
    print(f'Index view z0: {out_z0}')
    print(f'Index view z1: {out_z1}')

    out_x=trade.get_variable('vehicles.sellar_vehicle.sellar_base.x')
    print(f'The x des var: {out_x}')

    obj=trade.get_variable('sellar_objective.obj')
    print(f'The objective function: {obj}')

    con1=trade.get_variable('sellar_objective.con1')
    print(f'The constraint 1 value: {con1}')

    con2=trade.get_variable('sellar_objective.con2')
    print(f'The constraint 2 value: {con2}')
