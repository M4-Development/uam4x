from uam4x.study import Study
from uam4x.post import GaussQuadtraturePCEPost,RegressionPCEPost
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def relative_error(value, reference):
    if np.isclose(reference, 0.0):
        return np.nan
    return (value - reference) / reference


def extract_first_float(text):
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        raise ValueError(f"No numeric value found in: {text}")
    return float(match.group(0))


if __name__ == "__main__":
    save_path = os.path.join(THIS_DIR, "variants", "nipc_uq")

    #--------------------------------------Run NIPC UQ analysis
    study = Study.load_study_from_directory(save_path)
    soln_results=study.results.get_solution_result('basic')
    # PCE regression tradestudy using LHS

    lhs_nipc_ts = soln_results.get_tradestudy("lhs_pce_analysis")
    lhs_nipc_post=RegressionPCEPost(lhs_nipc_ts)
    lhs_nipc_surrogate = lhs_nipc_post.generate_surrogate("calculate_failure.g_val", poly_order=5)
    lhs_mean = float(lhs_nipc_surrogate.analytical_mean)
    lhs_std = float(lhs_nipc_surrogate.analytical_std)
    print(f"mean of LHS PCE is {lhs_mean}")
    print(f"variance of LHS PCE is {lhs_std**2}")

    # PCE regression tradestudy using MC
    mc_nipc_ts = soln_results.get_tradestudy("mc_pce_analysis")
    mc_nipc_post=RegressionPCEPost(mc_nipc_ts)
    mc_nipc_surrogate = mc_nipc_post.generate_surrogate("calculate_failure.g_val", poly_order=5)
    mse=mc_nipc_surrogate.get_surrogate_mse()
    r2=mc_nipc_surrogate.get_surrogate_r2()
    rel_mse=mc_nipc_surrogate.get_surrogate_rel_mse()
    mc_mean = float(mc_nipc_surrogate.analytical_mean)
    mc_std = float(mc_nipc_surrogate.analytical_std)
    print(f"mean of MC PCE is {mc_mean}")
    print(f"variance of MC PCE is {mc_std**2}")

    # PCE gauss quadrature
    quad_ts = soln_results.get_tradestudy("quad_pce_analysis")
    quad_nipc_post = GaussQuadtraturePCEPost(quad_ts)
    quad_nipc_surrogate = quad_nipc_post.generate_surrogate("calculate_failure.g_val")
    quad_mean = float(quad_nipc_surrogate.analytical_mean)
    quad_std = float(quad_nipc_surrogate.analytical_std)
    print(f"mean of quad PCE is {quad_mean}")
    print(f"variance of quad PCE is {quad_std**2}")


    #Another form of surrogate validation:
    # Use the independently generated MC samples as the test
    #set and run them on the LHS surrogate.
    x1=np.array(mc_nipc_ts.get_variable("vehicles.vehicle1.beam1.bending_moment"))[:,None]
    x2=np.array(mc_nipc_ts.get_variable("vehicles.vehicle1.beam1.yield_strength"))[:,None]
    X_test=np.hstack((x1,x2))
    y_test=np.array(mc_nipc_ts.get_variable("calculate_failure.g_val"))
    print(f"LHS PCE MSE: {lhs_nipc_surrogate.get_surrogate_mse(X_test,y_test)}")
    print(f"LHS PCE R2: {lhs_nipc_surrogate.get_surrogate_r2(X_test,y_test)}")
    print(f"LHS PCE Relative MSE: {lhs_nipc_surrogate.get_surrogate_rel_mse(X_test,y_test)}")
    # Compare to m4uq mean/std
    base_m4dir=os.path.join(THIS_DIR,'m4uq_ref','OutputFolder')

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

    lhs_nipc_prediction = []
    mc_nipc_prediction = []
    quad_prediction = []
    for moment, yield_stress in zip(moments, yield_stresses):
        input_dict = {
            "vehicles.vehicle1.beam1.bending_moment": moment,
            "vehicles.vehicle1.beam1.yield_strength": yield_stress,
        }
        lhs_nipc_prediction.append(lhs_nipc_surrogate.predict(input_dict))
        mc_nipc_prediction.append(mc_nipc_surrogate.predict(input_dict))
        quad_prediction.append(quad_nipc_surrogate.predict(input_dict))

    lhs_output = np.asarray(lhs_nipc_prediction).reshape(-1)
    mc_output = np.asarray(mc_nipc_prediction).reshape(-1)
    quad_output = np.asarray(quad_prediction).reshape(-1)

    data_by_method = {
        "lhs": lhs_output,
        "mc": mc_output,
        "quad": quad_output,
    }

    # Compare confidence intervals
    quantiles = {"lhs": [], "mc": [], "quad": []}
    for name, data in data_by_method.items():
        high = float(np.quantile(data, 0.975))
        low = float(np.quantile(data, 0.025))
        quantiles[name] = [low, high]

    ci_file = os.path.join(base_m4dir,"ConfidenceIntervals.dat")
    with open(ci_file, "r") as f:
        f.readline()
        line = f.readline().strip()
        line = line[1:-1].split(",")
        low = float(line[0])
        high = float(line[1])

    quantiles["m4_uq"] = [low, high]

    # Compare correlation coefficients
    correlations = {
        "lhs": {"pearson": {}, "spearman": {}},
        "mc": {"pearson": {}, "spearman": {}},
        "quad": {"pearson": {}, "spearman": {}},
        "m4_uq": {"pearson": {}, "spearman": {}},
    }

    for method_name, output_values in data_by_method.items():
        correlations[method_name]["pearson"][("moment", "output")] = stats.pearsonr(
            moments, output_values
        )[0]
        correlations[method_name]["spearman"][("moment", "output")] = stats.spearmanr(
            moments, output_values
        )[0]
        correlations[method_name]["pearson"][("yield_stress", "output")] = stats.pearsonr(
            yield_stresses, output_values
        )[0]
        correlations[method_name]["spearman"][("yield_stress", "output")] = stats.spearmanr(
            yield_stresses, output_values
        )[0]

    correlation_file = os.path.join(base_m4dir,"Correlation","NIPCCorrelations.dat")
    with open(correlation_file, "r") as f:
        lines = f.readlines()

    pearson_moment = float(lines[2].strip().split("\t")[1])
    pearson_yield = float(lines[3].strip().split("\t")[1])
    spearman_moment = float(lines[7].strip().split("\t")[1])
    spearman_yield = float(lines[8].strip().split("\t")[1])
    correlations["m4_uq"]["pearson"][("moment", "output")] = pearson_moment
    correlations["m4_uq"]["spearman"][("moment", "output")] = spearman_moment
    correlations["m4_uq"]["pearson"][("yield_stress", "output")] = pearson_yield
    correlations["m4_uq"]["spearman"][("yield_stress", "output")] = spearman_yield

    print("correlations")
    print(correlations)

    # ---------------- Excel report ----------------
    page1_rows = [
        {
            "method": "m4_uq",
            "mean": m4uq_mean,
            "std_dev": m4uq_std,
            "relative_error_mean": 0.0,
            "relative_error_std_dev": 0.0,
        },
        {
            "method": "lhs",
            "mean": lhs_mean,
            "std_dev": lhs_std,
            "relative_error_mean": relative_error(lhs_mean, m4uq_mean),
            "relative_error_std_dev": relative_error(lhs_std, m4uq_std),
        },
        {
            "method": "mc",
            "mean": mc_mean,
            "std_dev": mc_std,
            "relative_error_mean": relative_error(mc_mean, m4uq_mean),
            "relative_error_std_dev": relative_error(mc_std, m4uq_std),
        },
        {
            "method": "quad",
            "mean": quad_mean,
            "std_dev": quad_std,
            "relative_error_mean": relative_error(quad_mean, m4uq_mean),
            "relative_error_std_dev": relative_error(quad_std, m4uq_std),
        },
    ]
    page1_df = pd.DataFrame(page1_rows)

    ci_rows = []
    ref_low, ref_high = quantiles["m4_uq"]
    for method in ["m4_uq", "lhs", "mc", "quad"]:
        low_ci, high_ci = quantiles[method]
        ci_rows.append(
            {
                "method": method,
                "ci_low": low_ci,
                "ci_high": high_ci,
                "relative_error_ci_low": relative_error(low_ci, ref_low),
                "relative_error_ci_high": relative_error(high_ci, ref_high),
            }
        )
    ci_df = pd.DataFrame(ci_rows)

    corr_rows = []
    inputs = ["moment", "yield_stress"]
    metrics = ["pearson", "spearman"]
    for method in ["m4_uq", "lhs", "mc", "quad"]:
        for metric in metrics:
            for inp in inputs:
                ref_value = correlations["m4_uq"][metric][(inp, "output")]
                value = correlations[method][metric][(inp, "output")]
                corr_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "input": inp,
                        "reference_m4_uq": ref_value,
                        "value": value,
                        "relative_error": relative_error(value, ref_value),
                    }
                )
    corr_df = pd.DataFrame(corr_rows)

    excel_out = os.path.join(THIS_DIR, "variants", "nipc_uq", "m4uq_surrogate_comparison.xlsx")
    with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
        page1_df.to_excel(writer, sheet_name="Page1_MeanStd", index=False)
        ci_df.to_excel(writer, sheet_name="Page2_CI_Corr", index=False, startrow=0)
        corr_df.to_excel(
            writer,
            sheet_name="Page2_CI_Corr",
            index=False,
            startrow=len(ci_df) + 3,
        )

    print(f"Saved Excel comparison report to: {excel_out}")
