import os
from uam4x.study import Study
from uam4x.post import SobolSensitivityPost
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STACKED_COMPARISON_PATH = os.path.join(THIS_DIR, "predicted_vs_mc_stacked.csv")
BASIC_SOBOL_EXCEL_PATH = os.path.join(
    THIS_DIR, "variants", "ishigami", "basic_sobol_results.xlsx"
)

def _relative_error(estimate, reference):
    if reference == 0.0:
        return np.nan
    return (estimate - reference) / reference

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "ishigami")

    study=Study.load_study_from_directory(os.path.join(study_dir))
    solution_results=study.results.get_solution_result("basic")
    ts=solution_results.get_tradestudy('mc_ishigami_problem')
    uq_post=SobolSensitivityPost(ts)

    uq_post.plot_total_order_indices('ishigami_calc.g',
                                     save_dir=study_dir)
    first_order_terms=uq_post.get_first_order_indices('ishigami_calc.g')
    total_order_terms=uq_post.get_total_order_indices('ishigami_calc.g')

    S1=first_order_terms['vehicles.ishigami_vehicle.ishigami_base.X1']
    S2=first_order_terms['vehicles.ishigami_vehicle.ishigami_base.X2']
    S3=first_order_terms['vehicles.ishigami_vehicle.ishigami_base.X3']

    ST1=total_order_terms['vehicles.ishigami_vehicle.ishigami_base.X1']
    ST2=total_order_terms['vehicles.ishigami_vehicle.ishigami_base.X2']
    ST3=total_order_terms['vehicles.ishigami_vehicle.ishigami_base.X3']
    #Analytical values:
    a=ts.get_variable('vehicles.ishigami_vehicle.ishigami_base.a')[0]
    b=ts.get_variable('vehicles.ishigami_vehicle.ishigami_base.b')[0]
    expectation_g=a/2
    var_g=1./2.+a**2/8+b**2*np.pi**8/18.+b*np.pi**4/5

    V1_analytical=1./2.*(1+b*np.pi**4/5.)**2

    V2_analytical=a**2/8.

    V13_analytical=b**2*np.pi**8*8./225.

    V3_analytical=V12_analytical=V23_analytical=V123_analytical=0.

    #Sobol indices First order
    S1_analytical=V1_analytical/var_g
    S2_analytical=V2_analytical/var_g
    S3_analytical=0.

    #Total order
    ST1_analytical=(V1_analytical+V13_analytical)/var_g

    ST2_analytical=S2_analytical

    ST3_analytical=V13_analytical/var_g

    print("\nSobol sensitivity comparison (computed vs analytical)")
    print("-" * 95)
    print(f"{'Index':<6}{'Computed':>16}{'Analytical':>16}{'Abs Error':>16}{'Rel Error [%]':>18}")
    print("-" * 95)

    comparisons = [
        ("S1", S1, S1_analytical),
        ("S2", S2, S2_analytical),
        ("S3", S3, S3_analytical),
        ("ST1", ST1, ST1_analytical),
        ("ST2", ST2, ST2_analytical),
        ("ST3", ST3, ST3_analytical),
    ]

    comparison_rows = []
    for name, computed, analytical in comparisons:
        abs_err = computed - analytical
        rel_err = _relative_error(computed, analytical)
        rel_err_str = f"{100.0 * rel_err: .6f}" if not np.isnan(rel_err) else "N/A"
        print(f"{name:<6}{computed:>16.8f}{analytical:>16.8f}{abs_err:>16.8f}{rel_err_str:>18}")
        comparison_rows.append(
            {
                "index": name,
                "computed": computed,
                "analytical": analytical,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    first_order_df = pd.DataFrame(
        [
            {"variable": variable, "first_order_index": value}
            for variable, value in first_order_terms.items()
        ]
    )
    total_order_df = pd.DataFrame(
        [
            {"variable": variable, "total_order_index": value}
            for variable, value in total_order_terms.items()
        ]
    )

    with pd.ExcelWriter(BASIC_SOBOL_EXCEL_PATH, engine="xlsxwriter") as writer:
        comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
        first_order_df.to_excel(writer, sheet_name="FirstOrder", index=False)
        total_order_df.to_excel(writer, sheet_name="TotalOrder", index=False)

    print(f"\nSaved Excel Sobol report to: {BASIC_SOBOL_EXCEL_PATH}")
