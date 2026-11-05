import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from uam4x.post import MixedRegressionPCEPost, BasicMixedPost
from branching_script import setup_model,setup_solution,setup_tradestudy
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def relative_error(value,reference):
    if np.isclose(reference,0.0):
        return np.nan
    else:
        return np.abs((value-reference)/(reference))*100.


def test_mixed_cantilever():
    #run branching:
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('mixed_cantilever', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","mixed_cantilever")

    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path,"work"))
    trades=["pce_cantilever_problem","mc_cantilever_problem"]



    solution_results = None
    for trade in trades:
        engine.run_tradestudy(
            trade, solution_name="basic", create_step_work_dirs=False
        )
        if solution_results is None:
            solution_results = study.results.get_solution_result("basic")
        ts=solution_results.get_tradestudy(trade)
        if trade in ["pce_cantilever_problem","m4_samples"]:
            post=MixedRegressionPCEPost(ts)
            surrogate=post.generate_surrogate("cantilever_deflection.g", poly_order=2)
            #M4 sample input points used to build the surrogate:
            m4_ref_in=os.path.join(THIS_DIR,'..','m4_ref','Samples.dat')
            with open(m4_ref_in,'r') as f:
                in_lines=f.readlines()

            #Corresponding model evaluations:
            m4_ref_out=os.path.join(THIS_DIR,'..','m4_ref','SystemOutputValues.dat')
            with open(m4_ref_out,'r') as f:
                out_lines=f.readlines()

            data=[]
            for in_line,out_line in zip(in_lines,out_lines):
                in_line=in_line.strip()
                in_line=in_line.split(',')
                out_line=float(out_line.strip())
                data.append([float(val) for val in in_line]+[out_line])

            all_vals=[]
            if trade=="pce_cantilever_problem":
                high_error=10.0
                low_error=2.0
            else:
                high_error=1.5
                low_error=0.4
            for i,point in enumerate(data):
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
                #most values are 1%, one outlier at 10%
                if i!=11:
                    assert relative_error(surrogate_val,point[3])<low_error
                else:
                    assert relative_error(surrogate_val,point[3])<high_error


            #Compare sobol indices:
            m4_sobol=os.path.join(THIS_DIR,'..','m4_ref','SobolMSRV_0.dat')
            with open(m4_sobol,'r') as f:
                sobol_lines=f.readlines()

            data=[]
            for line in sobol_lines:
                line=line.strip()
                line=line.split(',')
                data.append([float(value) for value in line])

            #note the first row corresponds to young modulus
            young_sobol=surrogate.get_total_order_sobol( "vehicles.base.cantilever_base.Y")
            assert relative_error(young_sobol,data[0][1])<0.05
            #second row corresponds to moment
            moment_sobol=surrogate.get_total_order_sobol( "vehicles.base.cantilever_base.M")
            assert relative_error(data[1][1],moment_sobol)<0.5

        else:
            post=BasicMixedPost(ts)
            #Compare the p-levels with M4 UQ
            m4_ref=os.path.join(THIS_DIR,'..','m4_ref','ProbabilityLevels_0.dat')
            with open(m4_ref,'r') as f:
                lines=f.readlines()

            data=[]
            for line in lines:
                line=line.strip()
                line=line.split(',')
                data.append([float(val) for val in line])

            for i,point in enumerate(data):
                plevel=point[0]
                m4_low=point[1]
                m4_high=point[2]
                uam4x_low,uam4x_high=post.get_plevel_bounds("cantilever_deflection.g",plevel)
                if i==0:
                    #first point is at 0.025 where rel error is quite high (probably surrogate limitation)
                    assert relative_error(uam4x_low,m4_low)<32.
                    assert relative_error(uam4x_high,m4_high)<21.
                else:
                    #error outside of the low tail is much better between raw sampling and surrogate
                    assert relative_error(uam4x_low,m4_low)<4.0
                    assert relative_error(uam4x_high,m4_high)<1.5


if __name__=="__main__":
    test_mixed_cantilever()
