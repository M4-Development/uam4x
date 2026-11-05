import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from uam4x.post import RegressionPCEPost
from branching_script import setup_model,setup_solution,setup_tradestudy
import os
import numpy as np
import scipy.stats as stats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VSP_FILE_PATH = os.path.join(THIS_DIR, "NASA_Altered_Multi-Tiltrotor.vsp3")
def relative_error(value,reference):
    if np.isclose(reference,0.0):
        return np.nan
    else:
        return np.abs((value-reference)/(reference))*100.

def test_borehole():
    #run branching:
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('borehole', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","borehole")


    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path,"work"))
    engine.run_tradestudy(
        "borehole_problem", solution_name="basic", create_step_work_dirs=False
    )

    uam4x_reference_data={
            'analytical_mean':72.89569974764134,
            'analytical_variance':788.5542545498289,
            'analytical_std':28.081208210293035,
            'sampled_mean':72.8635958139405,
            'sampled_variance':788.3300716651863,
            'sampled_std':28.07721623781792
        }

    #Monte carlo reference:
    ref_path=os.path.join(THIS_DIR,'..','full_analysis.csv')
    reference_data=np.genfromtxt(ref_path,delimiter=',',skip_header=1)

    #get post data:
    solution_results=study.results.get_solution_result("basic")
    ts=solution_results.get_tradestudy('borehole_problem')
    post=RegressionPCEPost(ts)
    surrogate=post.generate_surrogate('borehole_calc.flow_rate', poly_order=5)
    analytical_mean=float(surrogate.analytical_mean)
    analytical_variance=float(surrogate.analytical_variance)
    analytical_std=float(surrogate.analytical_std)

    #test the surrogate performance on the reference monte carlo points
    predicted_outs=[]
    monte_carlo_out=[]
    for row in reference_data:
        rw,r,Tu,Hu,Tl,Hl,L,Kw=row[:8]
        monte_carlo_out.append(row[8])

        sample_input_prediction={
            'vehicles.borehole_vehicle.borehole_base.rw':rw,
            'vehicles.borehole_vehicle.borehole_base.r':r,
            'vehicles.borehole_vehicle.borehole_base.Tu':Tu,
            'vehicles.borehole_vehicle.borehole_base.Hu':Hu,
            'vehicles.borehole_vehicle.borehole_base.Tl':Tl,
            'vehicles.borehole_vehicle.borehole_base.Hl':Hl,
            'vehicles.borehole_vehicle.borehole_base.L':L,
            'vehicles.borehole_vehicle.borehole_base.Kw':Kw
        }
        out=surrogate.predict(sample_input_prediction)[0]
        predicted_outs.append(out)

    predicted_outs=np.array(predicted_outs, dtype=float)
    monte_carlo_out=np.array(monte_carlo_out, dtype=float)

    # Monte Carlo moments from reference outputs
    monte_carlo_mean=float(np.mean(monte_carlo_out))
    monte_carlo_std=float(np.std(monte_carlo_out))
    monte_carlo_variance=float(np.var(monte_carlo_out))

    #compare the same analysis to known benchmark
    assert relative_error(analytical_mean, uam4x_reference_data['analytical_mean'])<=1e-5
    assert relative_error(analytical_variance, uam4x_reference_data['analytical_variance'])<=1e-5
    assert relative_error(analytical_std, uam4x_reference_data['analytical_std'])<=1e-5

    #compare the monte carlo to analytical surrogate
    assert relative_error(analytical_mean, monte_carlo_mean)<=0.5
    assert relative_error(analytical_variance, monte_carlo_variance)<=0.5
    assert relative_error(analytical_std, monte_carlo_std)<=0.5


    # RMS error between surrogate predictions and Monte Carlo outputs at the same monte carlo input points
    #compared to mean, error is low
    value_rms_error=float(np.sqrt(np.mean((monte_carlo_out - predicted_outs)**2)))
    assert value_rms_error<1.0


    # Quantile and probability comparisons against the Monte Carlo reference file
    res=stats.ecdf(predicted_outs).cdf
    predicted_quantiles=np.array(res.quantiles, dtype=float)
    #quantiles precomputed from monte carlo outputs in reference file
    monte_carlo_quantiles=np.array(reference_data[:,9], dtype=float)
    quantile_rms_error=float(np.sqrt(np.mean((monte_carlo_quantiles-predicted_quantiles)**2)))
    #note min quantile value>5.0 and 280. so very low rms error
    assert quantile_rms_error<0.5
if __name__=="__main__":
    test_borehole()
