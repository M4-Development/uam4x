import os
from uam4x.study import Study
import numpy as np
from uam4x.post import BasicMixedPost
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_rel_error(ref,new_val,tol=10**-6):
    if np.abs(ref-new_val)<tol:
        return new_val-ref
    else:
        return np.abs((new_val-ref)/ref)*100


if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "mixed_cantilever")
    study=Study.load_study_from_directory(study_dir)
    solution_results=study.results.get_solution_result("basic")
    trade=solution_results.get_tradestudy('mc_cantilever_problem')

    post=BasicMixedPost(trade)
    post.get_plevel_bounds("cantilever_deflection.g",0.0265)
    post.plot_pbox("cantilever_deflection.g",save_dir=study_dir)
    post.plot_horsetails("cantilever_deflection.g",save_dir=study_dir)

    #Compare the p-levels with M4 UQ
    m4_ref=os.path.join(THIS_DIR,'m4_ref','ProbabilityLevels_0.dat')
    with open(m4_ref,'r') as f:
        lines=f.readlines()

    data=[]
    for line in lines:
        line=line.strip()
        line=line.split(',')
        data.append([float(val) for val in line])

    p_levels=[]
    m4_lows=[]
    m4_highs=[]
    uam4x_lows=[]
    uam4x_highs=[]
    low_error=[]
    high_error=[]
    for point in data:
        plevel=point[0]
        m4_low=point[1]
        m4_high=point[2]
        low,high=post.get_plevel_bounds("cantilever_deflection.g",plevel)

        m4_lows.append(m4_low)
        m4_highs.append(m4_high)
        uam4x_lows.append(low)
        uam4x_highs.append(high)
        low_error.append(compute_rel_error(m4_low,low))
        high_error.append(compute_rel_error(m4_high,high))
        p_levels.append(plevel)


    print("\nProbability level comparison (uam4x vs. M4)")
    print("-" * 120)
    print(
        f"{'p-level':<10}"
        f"{'uam4x low':>16}"
        f"{'M4 low':>16}"
        f"{'low err [%]':>16}"
        f"{'uam4x high':>16}"
        f"{'M4 high':>16}"
        f"{'high err [%]':>16}"
    )
    print("-" * 120)

    for plevel, low, m4_low, low_err, high, m4_high, high_err in zip(
        p_levels, uam4x_lows, m4_lows, low_error, uam4x_highs, m4_highs, high_error
    ):
        print(
            f"{plevel:<10.4f}"
            f"{low:>16.8f}"
            f"{m4_low:>16.8f}"
            f"{low_err:>16.6f}"
            f"{high:>16.8f}"
            f"{m4_high:>16.8f}"
            f"{high_err:>16.6f}"
        )

    print("-" * 120)
    print(
        f"Worst low-bound relative error:  {np.max(np.abs(low_error)):.6f}%\n"
        f"Worst high-bound relative error: {np.max(np.abs(high_error)):.6f}%"
    )
