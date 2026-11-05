import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from uam4x.post import RegressionPCEPost, SobolSensitivityPost
from branching_script import setup_model,setup_solution,setup_tradestudy
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def relative_error(value,reference):
    if np.isclose(reference,0.0):
        return np.nan
    else:
        return np.abs((value-reference)/(reference))*100.


def test_ishigami():
    #run branching:
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('ishigami', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","ishigami")

    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path,"work"))
    trades=["pce_ishigami_problem","mc_ishigami_problem"]

    a=7.0
    b=0.1
    trade_ref={
        "pce_ishigami_problem":{
            'S1':0.31679525,
            'S2':0.43774365,
            'S3':0.00007397,
            'ST1':0.55865402,
            'ST2':0.45340246,
            'ST3':0.24300350,
        },
        "mc_ishigami_problem":{
            'S1':0.31125862,
            'S2':0.44399855,
            'S3':0.00326209,
            'ST1':0.55999029,
            'ST2':0.44309153,
            'ST3':0.24376269 ,
        }
    }

    reference_mean=float(a/2.)
    reference_variance=float(1./2.+a**2/8.+b**2*np.pi**8/18.+b*np.pi**4/5.)
    reference_std=float(np.sqrt(reference_variance))

    V1_analytical=1./2.*(1.+b*np.pi**4/5.)**2
    V2_analytical=a**2/8.
    V13_analytical=b**2*np.pi**8*8./225.

    S1_reference=float(V1_analytical/reference_variance)
    S2_reference=float(V2_analytical/reference_variance)
    S3_reference=0.0
    ST1_reference=float((V1_analytical+V13_analytical)/reference_variance)
    ST2_reference=S2_reference
    ST3_reference=float(V13_analytical/reference_variance)

    solution_results = None
    for trade in trades:
        engine.run_tradestudy(
            trade, solution_name="basic", create_step_work_dirs=False
        )
        if solution_results is None:
            solution_results = study.results.get_solution_result("basic")
        ts=solution_results.get_tradestudy(trade)

        if trade=='pce_ishigami_problem':
            post=RegressionPCEPost(ts)
            surrogate=post.generate_surrogate('ishigami_calc.g', poly_order=7)
            analytical_mean=float(surrogate.analytical_mean)
            analytical_variance=float(surrogate.analytical_variance)
            analytical_std=float(surrogate.analytical_std)
            S1=float(surrogate.get_first_order_sobol('vehicles.ishigami_vehicle.ishigami_base.X1'))
            S2=float(surrogate.get_first_order_sobol('vehicles.ishigami_vehicle.ishigami_base.X2'))
            S3=float(surrogate.get_first_order_sobol('vehicles.ishigami_vehicle.ishigami_base.X3'))
            ST1=float(surrogate.get_total_order_sobol('vehicles.ishigami_vehicle.ishigami_base.X1'))
            ST2=float(surrogate.get_total_order_sobol('vehicles.ishigami_vehicle.ishigami_base.X2'))
            ST3=float(surrogate.get_total_order_sobol('vehicles.ishigami_vehicle.ishigami_base.X3'))

            assert relative_error(analytical_mean,reference_mean)<=2.0
            assert relative_error(analytical_variance,reference_variance)<=1.0
            assert relative_error(analytical_std,reference_std)<=0.5

            assert relative_error(S1,S1_reference)<=1.0
            assert relative_error(S2,S2_reference)<=2.0
            assert np.isclose(S3,S3_reference,atol=10**-4)
            assert relative_error(ST1,ST1_reference)<=1.0
            assert relative_error(ST2,ST2_reference)<=3.0
            assert relative_error(ST3,ST3_reference)<=1.0

            assert np.isclose(S1,trade_ref[trade]['S1'],atol=10**-4)
            assert np.isclose(S2,trade_ref[trade]['S2'],atol=10**-4)
            assert np.isclose(S3,trade_ref[trade]['S3'],atol=10**-4)
            assert np.isclose(ST1,trade_ref[trade]['ST1'],atol=10**-4)
            assert np.isclose(ST2,trade_ref[trade]['ST2'],atol=10**-4)
            assert np.isclose(ST3,trade_ref[trade]['ST3'],atol=10**-4)
        else:
            post=SobolSensitivityPost(ts)
            first_order_terms=post.get_first_order_indices('ishigami_calc.g')
            total_order_terms=post.get_total_order_indices('ishigami_calc.g')
            sampled_mean=float(post.compute_mean('ishigami_calc.g'))
            sampled_variance=float(post.compute_variance('ishigami_calc.g',method='population'))
            sampled_std=float(post.compute_std('ishigami_calc.g',method='population'))
            S1=float(first_order_terms['vehicles.ishigami_vehicle.ishigami_base.X1'])
            S2=float(first_order_terms['vehicles.ishigami_vehicle.ishigami_base.X2'])
            S3=float(first_order_terms['vehicles.ishigami_vehicle.ishigami_base.X3'])
            ST1=float(total_order_terms['vehicles.ishigami_vehicle.ishigami_base.X1'])
            ST2=float(total_order_terms['vehicles.ishigami_vehicle.ishigami_base.X2'])
            ST3=float(total_order_terms['vehicles.ishigami_vehicle.ishigami_base.X3'])

            assert relative_error(sampled_mean,reference_mean)<=1.0
            assert relative_error(sampled_variance,reference_variance)<=1.0
            assert relative_error(sampled_std,reference_std)<=1.0

            assert relative_error(S1,S1_reference)<=1.0
            assert relative_error(S2,S2_reference)<=3.0
            assert np.isclose(S3,S3_reference,atol=10**-2)
            assert relative_error(ST1,ST1_reference)<=1.0
            assert relative_error(ST2,ST2_reference)<=1.0
            assert relative_error(ST3,ST3_reference)<=1.0

            assert np.isclose(S1,trade_ref[trade]['S1'],atol=10**-4)
            assert np.isclose(S2,trade_ref[trade]['S2'],atol=10**-4)
            assert np.isclose(S3,trade_ref[trade]['S3'],atol=10**-4)
            assert np.isclose(ST1,trade_ref[trade]['ST1'],atol=10**-4)
            assert np.isclose(ST2,trade_ref[trade]['ST2'],atol=10**-4)
            assert np.isclose(ST3,trade_ref[trade]['ST3'],atol=10**-4)


if __name__=="__main__":
    test_ishigami()
