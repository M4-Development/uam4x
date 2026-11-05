import os
from uam4x.study import Study
import numpy as np
from uam4x.post import MixedRegressionPCEPost

def compute_rel_error(ref,new_val,tol=10**-6):
    if np.abs(ref-new_val)<tol:
        print(f'Difference less than tolerance, taking absolute difference')
        return new_val-ref
    else:
        return np.abs((new_val-ref)/ref)*100


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "mixed_cantilever")
    study=Study.load_study_from_directory(study_dir)
    solution_results=study.results.get_solution_result("basic")
    trade=solution_results.get_tradestudy('pce_cantilever_problem')

    post=MixedRegressionPCEPost(trade)

    surrogate=post.generate_surrogate("cantilever_deflection.g", poly_order=2)
    surrogate.sample(50,1000)
    surrogate.plot_horsetails(save_dir=study_dir)
    surrogate.get_plevel_bounds(0.0265)
    surrogate.plot_pbox(save_dir=study_dir)
    surrogate.plot_total_order_sobol(save_dir=study_dir)

    sample_inp=surrogate.new_sample_points[0]
    inp_dict={
        "vehicles.base.cantilever_base.M":sample_inp[1],
        "vehicles.base.cantilever_base.Y":sample_inp[2],
        "vehicles.base.cantilever_base.T":sample_inp[0]
    }
    pred_val=surrogate.predict(inp_dict)
    #check the polynomial model
    poly_surrogate, var_map = surrogate.get_symbolic_metamodel("physical")
    poly_inp_dict={
        'x0':sample_inp[0],
        'x1':sample_inp[1],
        'x2':sample_inp[2]}
    poly_est=poly_surrogate.subs(poly_inp_dict)
    #Compare values to M4 UQ, instead use predict on M4 sample points
    m4_ref_in=os.path.join(THIS_DIR,'m4_ref','Samples.dat')
    with open(m4_ref_in,'r') as f:
        in_lines=f.readlines()

    m4_ref_out=os.path.join(THIS_DIR,'m4_ref','SystemOutputValues.dat')
    with open(m4_ref_out,'r') as f:
        out_lines=f.readlines()

    data=[]
    for in_line,out_line in zip(in_lines,out_lines):
        in_line=in_line.strip()
        in_line=in_line.split(',')
        out_line=float(out_line.strip())
        data.append([float(val) for val in in_line]+[out_line])

    all_vals=[]
    rel_error=[]
    for point in data:
        comp_vals=[]
        var_dict={
            "vehicles.base.cantilever_base.M":point[0],
            "vehicles.base.cantilever_base.Y":point[1],
            "vehicles.base.cantilever_base.T":point[2]
        }

        comp_vals.append(point[3])
        surrogate_val=surrogate.predict(var_dict)[0]
        comp_vals.append(surrogate_val)
        all_vals.append(comp_vals)
        rel_error.append(compute_rel_error(point[3],surrogate_val))

    print(f'Max relative error between M4 surrogate and uam4x surrogate is {np.max(np.array(rel_error)):.3f}%')


    #Compare sobol indices:
    m4_sobol=os.path.join(THIS_DIR,'m4_ref','SobolMSRV_0.dat')
    with open(m4_sobol,'r') as f:
        sobol_lines=f.readlines()

    data=[]
    for line in sobol_lines:
        line=line.strip()
        line=line.split(',')
        data.append([float(value) for value in line])

    #note the first row corresponds to young modulus
    young_sobol=surrogate.get_total_order_sobol( "vehicles.base.cantilever_base.Y")
    print(f'Rel error in sobol indices for Youngs Modulus '
           f'between m4 and uam4x: {compute_rel_error(data[0][1],young_sobol):.2f}%')
    #second row corresponds to moment
    moment_sobol=surrogate.get_total_order_sobol( "vehicles.base.cantilever_base.M")
    print(f'Rel error in sobol indices for Moment '
           f'between m4 and uam4x: {compute_rel_error(data[1][1],moment_sobol):.2f}%')
