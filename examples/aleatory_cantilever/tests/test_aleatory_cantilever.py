import sys
from pathlib import Path
import os
import numpy as np
import re
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from branching_script import setup_model,setup_solution,setup_tradestudy

from uam4x.post import GaussQuadtraturePCEPost, RegressionPCEPost
import scipy.stats as stats
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VSP_FILE_PATH = os.path.join(THIS_DIR, "NASA_Altered_Multi-Tiltrotor.vsp3")



def extract_first_float(text):
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        raise ValueError(f"No numeric value found in: {text}")
    return float(match.group(0))

def relative_error(value,reference):
    if np.isclose(reference,0.0):
        return np.nan
    else:
        return np.abs((value-reference)/(reference))*100.

def test_aleatory_cantilever():
    #run branching:
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('nipc_uq', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","nipc_uq")


    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path,"work"))
    trades=["lhs_pce_analysis","mc_pce_analysis","quad_pce_analysis"]
    #validated reference
    trade_to_regression_data={
        'lhs_pce_analysis':{
            'analytical_mean':0.18456856164819732,
            'analytical_variance':0.010626277947905946,
        },
        'quad_pce_analysis':{
            'analytical_mean':0.18456853517212182,
            'analytical_variance':0.010626307451286758
        },
        'mc_pce_analysis':{
            'analytical_mean':0.1845685562794462,
            'analytical_variance':0.010626306167891888
        },
        'm4_uq':{
            'analytical_mean':0.18456868641329277,
            'analytical_variance':0.01062612148568702,
            'p_level_0.025':-0.02291118198976921,
            'p_level_0.975':0.3655238319863137,
            'pearson_corr_yield':0.7755555635624036,
            'spearman_corr_yield':0.7854420838336834,
            'pearson_corr_moment':-0.6298096606120087,
            'spearman_corr_moment':-0.6015098585883943
        }


    }

    #M4UQ data:
    base_m4dir=os.path.join(ROOT, 'm4uq_ref', 'OutputFolder')
    #get the variance stuff
    out_file = os.path.join(base_m4dir,'MuSigmaVar.dat')
    with open(out_file, "r") as f:
        f.readline()
        lines = f.readlines()
    m4uq_mean = extract_first_float(lines[0].strip())
    m4uq_variance = extract_first_float(lines[2].strip())
    m4uq_std = float(np.sqrt(m4uq_variance))

    # Parse reference points
    normalized_sample_file=os.path.join(base_m4dir,'PDFs_CDFs','PDF_CDF_Samples.dat')
    with open(normalized_sample_file, "r") as f:
        lines = f.readlines()

    moments = []
    yield_stresses = []
    for line in lines:
        line = line.strip().split(",")
        moments.append(float(line[0]))
        yield_stresses.append(float(line[1]))

    # transform standard normal N(0,1)-->N(2500.,200)
    untransformed_moment=moments
    moments = np.array(moments) * 200.0 + 2500.0
    # transform uniform Unif(-1,1)-->Unif(2000,2800)
    untransformed_yield_stresses=yield_stresses
    yield_stresses = np.array(yield_stresses) * (2800.0 - 2000.0) / 2.0 + 2400.0


    mean_tol=1e-5
    var_tol=1e-5
    quantile_tol=1e-3
    solution_results = None
    for trade in trades:
        engine.run_tradestudy(
            trade, solution_name="basic", create_step_work_dirs=False
        )
        if solution_results is None:
            solution_results = study.results.get_solution_result("basic")
        ts=solution_results.get_tradestudy(trade)
        if 'quad' in trade:
            post=GaussQuadtraturePCEPost(ts)
            surrogate=post.generate_surrogate('calculate_failure.g_val')
        else:
            post=RegressionPCEPost(ts)
            surrogate=post.generate_surrogate('calculate_failure.g_val', poly_order=5)
        analytical_mean=surrogate.analytical_mean
        analytical_var=surrogate.analytical_variance
        assert np.isclose(analytical_mean,
                          trade_to_regression_data[trade]['analytical_mean'],
                          mean_tol)
        assert np.isclose(analytical_var,
                          trade_to_regression_data[trade]['analytical_variance'],
                          var_tol)

        #Direct m4uq, uam4x comparison:
        m4_mean=trade_to_regression_data['m4_uq']['analytical_mean']
        m4_var=trade_to_regression_data['m4_uq']['analytical_variance']
        #allow for less than half a percent relative error (it's actually less)
        assert relative_error(analytical_mean,m4_mean)<=0.5
        assert relative_error(analytical_var,m4_var)<=0.5

        prediction_vals=[]
        for moment,yield_stress in zip(moments,yield_stresses):
            input_dict = {
            "vehicles.vehicle1.beam1.bending_moment": moment,
            "vehicles.vehicle1.beam1.yield_strength": yield_stress,
            }
            prediction=surrogate.predict(input_dict)
            prediction_vals.append(prediction)
        prediction_vals=np.array(prediction_vals).reshape(-1)

        #Quantiles:
        low_quantile=np.quantile(prediction_vals,0.025)
        high_quantile=np.quantile(prediction_vals,0.975)

        #allow for 0.5% error between uam4x and m4uq reference
        assert relative_error(low_quantile,trade_to_regression_data['m4_uq']['p_level_0.025'])<=0.5
        assert relative_error(high_quantile,trade_to_regression_data['m4_uq']['p_level_0.975'])<=0.5

        #correlations
        pearson_corr_yield, _ = stats.pearsonr(yield_stresses,prediction_vals)
        spearman_corr_yield, _ = stats.spearmanr( yield_stresses,prediction_vals)
        pearson_corr_moment, _ = stats.pearsonr( moments,prediction_vals)
        spearman_corr_moment, _ = stats.spearmanr(moments, prediction_vals)
        assert relative_error(pearson_corr_yield,trade_to_regression_data['m4_uq']['pearson_corr_yield'])<=0.005
        assert relative_error(spearman_corr_yield,trade_to_regression_data['m4_uq']['spearman_corr_yield'])<=0.005
        assert relative_error(pearson_corr_moment,trade_to_regression_data['m4_uq']['pearson_corr_moment'])<=0.005
        assert relative_error(spearman_corr_moment,trade_to_regression_data['m4_uq']['spearman_corr_moment'])<=0.005



if __name__=="__main__":
    test_aleatory_cantilever()
