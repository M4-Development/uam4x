import os
from uam4x.study import Study,Variants
from uam4x.execution_engine import Engine
import numpy as np
import scipy.stats as stats
import pytest
from halton_branching_script import setup_model,setup_solution,setup_tradestudy
import openturns as ot
from uam4x.post import BasicAleatoryPost
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STACKED_COMPARISON_PATH = os.path.join(THIS_DIR, "predicted_vs_mc_stacked.csv")

def _relative_error(estimate, reference,tol=10**-6):
    if reference<tol:
        return np.abs(estimate-reference)
    else:
        return (estimate-reference)/reference*100.

def test_halton_uniform():
    study_dir=os.path.join(THIS_DIR, "variants", "standard_uniform")
    standard_uniform=[setup_model,setup_solution,setup_tradestudy]
    framework=Variants(THIS_DIR)
    framework.add_variant('standard_uniform', standard_uniform)
    framework.generate_variants()
    study=Study.load_study_from_directory(study_dir)
    engine=Engine(study,os.path.join(study_dir,'work'))
    engine.run_tradestudy('standard_uniform', solution_name="basic", create_step_work_dirs=False,
                          print_log_info=False)
    study.save_results_to_json(study_dir)
    study=Study.load_study_from_directory(study_dir)
    #start asserting things
    solution_results = study.results.get_solution_result("basic")
    ts = solution_results.get_tradestudy('standard_uniform')
    uq_post=BasicAleatoryPost(ts)

    x=ts.get_variable('point_calc.x_out')
    assert len(x)==100000
    #Analytical mean: 0.5
    analytical_mu=0.5
    computed_mu=uq_post.compute_mean('point_calc.x_out')
    #relative calc
    assert _relative_error(computed_mu,analytical_mu)<1.
    #Analytical variance is 1/12
    analytical_variance=1./12.
    computed_var=uq_post.compute_variance('point_calc.x_out')
    #this is relative error as a percent
    assert _relative_error(computed_var,analytical_variance)<=1
    #Analytical skew is 0.0
    analytical_skew=0.
    computed_skew=uq_post.compute_skewness('point_calc.x_out')
    #absolute here
    assert _relative_error(computed_skew,analytical_skew)<10**-3
    #Excess kurtosis analytical E(X-mu/sigma)^4-3
    analytical_kurtosis=-6./5.
    computed_kurtosis=uq_post.compute_kurtosis('point_calc.x_out')
    #relative here
    assert _relative_error(computed_kurtosis,analytical_kurtosis)<1.

    #confirm seeding is deterministic:
    csv_path=os.path.join(THIS_DIR,'reference_values.csv')
    ref_vals=np.genfromtxt(csv_path,delimiter=',',skip_header=1)
    assert max(x-ref_vals)==0

    #analytical quantiles are F^-1_X(x)=x for 0<=x<=1
    #for standard uniform X~Unif(0,1)
    analytical_quantiles=np.linspace(0.,1.,21)
    for i,quantile in enumerate(analytical_quantiles):
        uam4x_quantile=uq_post.compute_quantile('point_calc.x_out',quantile)
        if i==0:
            assert _relative_error(uam4x_quantile,quantile)<10**-4
        else:
            assert _relative_error(uam4x_quantile,quantile)<1.

    #for X~Unif(a,b) F_X(x)=x/(b-a) so for standard
    #Uniform, its F_X(x)=x as well
    analytical_cdf=np.linspace(0.,1.,21)
    for i,probability in enumerate(analytical_cdf):
        uam4x_quantile=uq_post.compute_cdf_value('point_calc.x_out',probability)
        if i==0:
            assert _relative_error(uam4x_quantile,probability)<10**-4
        else:
            assert _relative_error(uam4x_quantile,probability)<1.
if __name__=="__main__":
    test_halton_uniform()
