import os

# import openvsp_config
# openvsp_config.LOAD_FACADE = True
# openvsp_config.LOAD_GRAPHICS = True

from uam4x.study import Study
from uam4x.post import BasicPost

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":

    # vsp.StartGUI()
    study_dir=os.path.join(THIS_DIR, "variants", "demo")
    study = Study.load_study_from_directory(study_dir)

    soln_res=study.results.get_solution_result("no_vis")
    #Sweep trade

    sweep_ts=soln_res.get_tradestudy('sweep')
    ar=sweep_ts.get_variable("vehicles.multi_objective_vehicle.wing1.aspect_ratio")
    span=sweep_ts.get_variable("vehicles.multi_objective_vehicle.wing1.span")
    obj=sweep_ts.get_variable("calc_objective.objective")
    print(f'Aspect Ratio: {ar}, Span: {span}, Objective: {obj} for sweep trade study.')

    post=BasicPost(sweep_ts)
    post.scatter(var1="calc_objective.CD",
                 var2="calc_objective.wing_weight",
                 title="Pareto Front Plot of Drag Coefficient vs. Wing Weight", xlabel="Drag Coefficient", ylabel="Wing Weight",save_dir=THIS_DIR)


    #Optimization trade
    opt_ts=soln_res.get_tradestudy('optimization_only')
    ar_opt=opt_ts.get_variable("vehicles.multi_objective_vehicle.wing1.aspect_ratio")
    span_opt=opt_ts.get_variable("vehicles.multi_objective_vehicle.wing1.span")
    obj_opt=opt_ts.get_variable("calc_objective.objective")
    print(f'Aspect Ratio: {ar_opt}, Span: {span_opt}, Objective: {obj_opt} for optimization trade study.')

    # single point trade
    sp_ts=soln_res.get_tradestudy('Default State')
    ar_sp=sp_ts.get_variable("vehicles.multi_objective_vehicle.wing1.aspect_ratio")
    span_sp=sp_ts.get_variable("vehicles.multi_objective_vehicle.wing1.span")
    obj_sp=sp_ts.get_variable("calc_objective.objective")
    print(f'Aspect Ratio: {ar_sp}, Span: {span_sp}, Objective: {obj_sp} for single point trade study.')

