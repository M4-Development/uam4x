import os
from uam4x.study import Study
import numpy as np
import scipy.stats as stats
from uam4x.post import RegressionPCEPost

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STACKED_COMPARISON_PATH = os.path.join(THIS_DIR, "predicted_vs_mc_stacked.csv")

def _relative_error(estimate, reference):
    if reference == 0.0:
        return np.nan
    return 100.0 * (estimate - reference) / reference

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "borehole")
    study=Study.load_study_from_directory(study_dir)
    solution_results=study.results.get_solution_result("basic")
    ts=solution_results.get_tradestudy('borehole_problem')
    uq_post=RegressionPCEPost(ts)

    #Generate surrogate model for output
    surrogate_model=uq_post.generate_surrogate('borehole_calc.flow_rate', poly_order=5)
    uq_post.compute_mean('borehole_calc.flow_rate')
    #Get the analytical values from the surrogate model:
    analytical_mean=float(surrogate_model.analytical_mean)
    analytical_std=float(surrogate_model.analytical_std)
    analytical_variance=float(surrogate_model.analytical_variance)

    #sample a bunch of points and predict outputs
    new_samples, predicted_outputs=surrogate_model.sample(num_samples=5000,method='LHS')

    #numerical mean, std, variance on the predictions
    sampled_mean=float(np.mean(predicted_outputs))
    sampled_std=float(np.std(predicted_outputs))
    sampled_var=float(np.var(predicted_outputs))

    ref_path=os.path.join(THIS_DIR,'full_analysis.csv')
    reference_data=np.genfromtxt(ref_path,delimiter=',',skip_header=1)

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
        out=surrogate_model.predict(sample_input_prediction)[0]
        predicted_outs.append(out)

    predicted_outs=np.array(predicted_outs, dtype=float)
    monte_carlo_out=np.array(monte_carlo_out, dtype=float)

    # Monte Carlo moments from reference outputs
    monte_carlo_mean=float(np.mean(monte_carlo_out))
    monte_carlo_std=float(np.std(monte_carlo_out))
    monte_carlo_variance=float(np.var(monte_carlo_out))

    # Analytical moments vs sampled-prediction moments
    print(f"Analytical vs Sampled Mean: analytical={analytical_mean:.12g}, sampled={sampled_mean:.12g}, relative_error={_relative_error(sampled_mean, analytical_mean):.12g}%")
    print(f"Analytical vs Sampled Std: analytical={analytical_std:.12g}, sampled={sampled_std:.12g}, relative_error={_relative_error(sampled_std, analytical_std):.12g}%")
    print(f"Analytical vs Sampled Variance: analytical={analytical_variance:.12g}, sampled={sampled_var:.12g}, relative_error={_relative_error(sampled_var, analytical_variance):.12g}%")

    # Analytical moments vs Monte Carlo moments
    print(f"Analytical vs Monte Carlo Mean: analytical={analytical_mean:.12g}, monte_carlo={monte_carlo_mean:.12g}, relative_error={_relative_error(monte_carlo_mean, analytical_mean):.12g}%")
    print(f"Analytical vs Monte Carlo Std: analytical={analytical_std:.12g}, monte_carlo={monte_carlo_std:.12g}, relative_error={_relative_error(monte_carlo_std, analytical_std):.12g}%")
    print(f"Analytical vs Monte Carlo Variance: analytical={analytical_variance:.12g}, monte_carlo={monte_carlo_variance:.12g}, relative_error={_relative_error(monte_carlo_variance, analytical_variance):.12g}%")

    # RMS error between surrogate predictions and Monte Carlo outputs at the same points
    value_rms_error=float(np.sqrt(np.mean((monte_carlo_out - predicted_outs)**2)))
    print(f"RMS Error (surrogate predictions vs Monte Carlo outputs): {value_rms_error:.12g}")

    # Quantile and probability comparisons against the Monte Carlo reference file
    res=stats.ecdf(predicted_outs).cdf
    predicted_quantiles=np.array(res.quantiles, dtype=float)
    monte_carlo_quantiles=np.array(reference_data[:,9], dtype=float)
    quantile_rms_error=float(np.sqrt(np.mean((monte_carlo_quantiles-predicted_quantiles)**2)))

    print(f"Quantile RMS Error (surrogate ECDF quantiles vs Monte Carlo): {quantile_rms_error:.12g}")

    stacked_data=np.column_stack((
        reference_data[:, :8],
        monte_carlo_out,
        predicted_outs,
        monte_carlo_quantiles,
        predicted_quantiles,
    ))
    np.savetxt(
        STACKED_COMPARISON_PATH,
        stacked_data,
        delimiter=",",
        header="rw,r,Tu,Hu,Tl,Hl,L,Kw,mc_output,predicted_output,mc_quantiles,predicted_quantiles",
    )
    print(f"Saved stacked comparison CSV: {STACKED_COMPARISON_PATH}")
