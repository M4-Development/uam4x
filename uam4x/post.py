from __future__ import annotations
import matplotlib.pyplot as plt
import openturns as ot
import numpy as np
from uam4x.UQ import LatinHypercubeSampler, SobolSampler,MonteCarloSampler,\
    HaltonSampler,NormalDistribution,UniformDistribution,LogNormalDistribution
import scipy.stats as stats
from math import comb

#--------------------------Basic Post Process Class
class BasicPost:
    def __init__(self, tradestudy: Tradestudy):
        self.tradestudy = tradestudy

    def scatter(
        self, var1, var2, xlabel=None, ylabel=None, title="default_title", save_dir=None
    ):
        """Plot a scatter chart for two variables across stored result points.

        Parameters
        ----------
        var1 : Any
            Variable name to plot on the x-axis.
        var2 : Any
            Variable name to plot on the y-axis.
        xlabel : Any, optional
            Input value for `xlabel`.
        ylabel : Any, optional
            Input value for `ylabel`.
        title : Any, optional
            Input value for `title`.
        save_dir : Any, optional
            Input value for `save_dir`.

        Returns
        -------
        Any
            Matplotlib axes object for the generated plot.

        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ts=self.tradestudy
        x_values = ts.get_variable(var1)
        y_values = ts.get_variable(var2)
        ax.scatter(x_values, y_values, color="blue", alpha=0.7)

        ax.set_title(title, fontsize=16, fontweight="bold")
        if xlabel is None:
            xlabel = var1
        if ylabel is None:
            ylabel = var2
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{title.replace(' ','_')}.png", dpi=300, bbox_inches="tight"
            )
        return ax

    def linear_regression(self,var1,var2):
        x_values = self.tradestudy.get_variable(var1)
        y_values = self.tradestudy.get_variable(var2)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        return slope, intercept, r_value, p_value, std_err



#--------------------------Aleatory Post Process Classes
# Define post process classes:
class BasicAleatoryPost:
    """Represent the basicstatspost component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize object state and configuration.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        self.tradestudy = tradestudy

    def compute_mean(self, var_name):
        """Compute the sample mean for a variable across trade-study results.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        vals = self.tradestudy.get_variable(var_name)
        return np.average(vals)

    def compute_variance(self, var_name, method="sample"):
        """Compute the sample or population variance for a variable.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        method : Any, optional
            Method identifier controlling algorithm behavior.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        vals = self.tradestudy.get_variable(var_name)
        sample = ["s", "sample"]
        population = ["p", "pop", "population"]
        if method in sample:
            return float(np.var(vals, ddof=1))
        elif method in population:
            return float(np.var(vals, ddof=0))
        else:
            raise ValueError(
                f"Unsupported variance method '{method}'. "
                "Expected one of: 's', 'sample', 'p', 'pop', 'population'."
            )

    def compute_std(self, var_name, method="sample"):
        """Compute the sample or population standard deviation for a variable.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        method : Any, optional
            Method identifier controlling algorithm behavior.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        vals = self.tradestudy.get_variable(var_name)
        sample = ["s", "sample"]
        population = ["p", "pop", "population"]
        if method in sample:
            return float(np.std(vals, ddof=1))
        elif method in population:
            return float(np.std(vals, ddof=0))
        else:
            raise ValueError(
                f"Unsupported standard deviation method '{method}'. "
                "Expected one of: 's', 'sample', 'p', 'pop', 'population'."
            )

    def compute_skewness(self, var_name, bias=True):

        # uses Fisher-Pearson
        """Compute Fisher-Pearson skewness for a variable.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        bias : Any, optional
            Input value for `bias`.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        return stats.skew(vals, nan_policy="raise", bias=bias)

    def compute_kurtosis(self, var_name, fisher=True, bias=True):
        # uses fisher or pearson
        """Compute kurtosis for a variable using SciPy conventions.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        fisher : Any, optional
            Input value for `fisher`.
        bias : Any, optional
            Input value for `bias`.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        return stats.kurtosis(vals, nan_policy="raise", bias=bias, fisher=fisher)

    def plot_pdf(
        self,
        var_name,
        x_axis_title=None,
        plot_title="",
        method="histogram",
        bins=20,
        save_dir=None,
    ):
        """Plot a PDF from previously generated surrogate samples.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        x_axis_title : Any, optional
            Input value for `x_axis_title`.
        plot_title : Any, optional
            Input value for `plot_title`.
        method : Any, optional
            Method identifier controlling algorithm behavior.
        bins : Any, optional
            Number of bins used for histogram-based plots.
        save_dir : Any, optional
            Optional directory where plots are saved.

        Returns
        -------
        Any
            Plot artifact (axes object or plotted data).

        """
        histo = ["histogram", "h"]
        kernal = ["kde", "kernel_density_estimate", "kernel", "k"]
        if method in histo:
            vals = np.array(self.tradestudy.get_variable(var_name))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(vals, bins=bins, color="blue", edgecolor="black")
            ax.set_title(
                plot_title if plot_title else f"PDF of {var_name}",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel(
                x_axis_title if x_axis_title is not None else var_name, fontsize=14
            )
            ax.set_ylabel("Frequency", fontsize=14)
        elif method in kernal:
            vals = np.array(self.tradestudy.get_variable(var_name))
            fig, ax = plt.subplots(figsize=(10, 6))
            kde = stats.gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(x_range, kde(x_range), color="blue")
            ax.set_title(
                plot_title if plot_title else f"PDF of {var_name}",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel(
                x_axis_title if x_axis_title is not None else var_name, fontsize=14
            )
            ax.set_ylabel("Density", fontsize=14)
        else:
            raise ValueError(
                f"Unsupported PDF plotting method '{method}'. "
                "Expected one of: 'histogram', 'h', 'kde', "
                "'kernel_density_estimate', 'kernel', 'k'."
            )
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/PDF_{var_name.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return ax

    def plot_cdf(self, var_name, x_axis_title=None, plot_title="", save_dir=None):
        """Plot a CDF from previously generated surrogate samples.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        x_axis_title : Any, optional
            Input value for `x_axis_title`.
        plot_title : Any, optional
            Input value for `plot_title`.
        save_dir : Any, optional
            Optional directory where plots are saved.

        Returns
        -------
        Any
            Plot artifact (axes object or plotted data).

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        ecdf = stats.ecdf(vals)
        quantiles = ecdf.cdf.quantiles
        probabilities = ecdf.cdf.probabilities
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(quantiles, probabilities, color="blue")
        if x_axis_title is None:
            x_axis_title = var_name
        if plot_title == "":
            plot_title = f"CDF of {x_axis_title}"
        ax.set_title(plot_title, fontsize=16, fontweight="bold")
        ax.set_xlabel(x_axis_title, fontsize=14)
        ax.set_ylabel("Probability", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return np.hstack((quantiles, probabilities))

    def get_cdf_function(self, var_name, method="empirical"):
        """Return a callable CDF approximation for a variable.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        method : Any, optional
            Method identifier controlling algorithm behavior.

        Returns
        -------
        Any
            Requested value.

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        ecdf = stats.ecdf(vals)
        quantiles = ecdf.cdf.quantiles
        probabilities = ecdf.cdf.probabilities
        if method == "empirical":

            def cdf(x):
                """Execute cdf.

                Parameters
                ----------
                x : Any
                    Input value for `x`.

                Returns
                -------
                Any
                    Return value produced by this operation.

                """
                return ecdf.cdf.evaluate(x)

        elif method == "smooth":

            def cdf(x):
                """Execute cdf.

                Parameters
                ----------
                x : Any
                    Input value for `x`.

                Returns
                -------
                Any
                    Return value produced by this operation.

                """
                return np.interp(x, quantiles, probabilities, left=0.0, right=1.0)

        else:
            raise ValueError(
                f"Unsupported CDF method '{method}'. Expected 'empirical' or 'smooth'."
            )
        return cdf

    def compute_cdf_value(self, var_name, value, method="empirical"):
        """Evaluate the CDF value at a specified point.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        value : Any
            Value at which to evaluate a distribution/CDF.
        method : Any, optional
            Method identifier controlling algorithm behavior.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        ecdf = stats.ecdf(vals)
        quantiles = ecdf.cdf.quantiles
        probabilities = ecdf.cdf.probabilities
        if method == "empirical":
            return ecdf.cdf.evaluate(value)
        elif method == "smooth":
            return np.interp(value, quantiles, probabilities, left=0.0, right=1.0)
        else:
            raise ValueError(
                f"Unsupported CDF method '{method}'. Expected 'empirical' or 'smooth'."
            )

    def compute_quantile(self, var_name, quantile, method="inverted_cdf"):
        """Compute a quantile value for a variable distribution.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        quantile : Any
            Requested quantile level in the interval [0, 1].
        method : Any, optional
            Method identifier controlling algorithm behavior.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError(
                f"Quantile must be between 0 and 1 inclusive; received {quantile}."
            )
        methods = [
            "inverted_cdf",
            "averaged_inverted_cdf",
            "closest_observation",
            "interpolated_inverted_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
        ]

        if method not in methods:
            raise ValueError(
                f"Unsupported quantile method '{method}'. Expected one of: "
                + ", ".join(f"'{name}'" for name in methods)
                + "."
            )

        vals = np.array(self.tradestudy.get_variable(var_name))

        return np.quantile(vals, quantile, method=method)

    def compute_spearman_correlation(self, var_name1, var_name2):
        # NEED TO DETERMINE NAN POLICY
        """Compute Spearman rank correlation between two variables.

        Parameters
        ----------
        var_name1 : Any
            First variable name for correlation calculations.
        var_name2 : Any
            Second variable name for correlation calculations.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        var1_vals = np.array(self.tradestudy.get_variable(var_name1))
        var2_vals = np.array(self.tradestudy.get_variable(var_name2))
        significance_res, p_val = stats.spearmanr(var1_vals, var2_vals)
        # SHOULD HANDLE >3 VARS REQUIRES MATRIX
        return significance_res, p_val

    def compute_pearson_correlation(self, var_name1, var_name2):
        # NEED TO DETERMINE NAN POLICY
        """Compute Pearson linear correlation between two variables.

        Parameters
        ----------
        var_name1 : Any
            First variable name for correlation calculations.
        var_name2 : Any
            Second variable name for correlation calculations.

        Returns
        -------
        Any
            Computed statistic or metric.

        """
        var1_vals = np.array(self.tradestudy.get_variable(var_name1))
        var2_vals = np.array(self.tradestudy.get_variable(var_name2))
        significance_res, p_val = stats.pearsonr(var1_vals, var2_vals)
        # SHOULD HANDLE >3 VARS REQUIRES MATRIX
        return significance_res, p_val

class RegressionPCEPost(BasicAleatoryPost):
    """Represent the regressionpcepost component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize object state and configuration.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        super().__init__(tradestudy)

    def generate_surrogate(self, output_name):
        """Build and return a polynomial-chaos surrogate for one output variable.

        Parameters
        ----------
        output_name : Any
            Output variable name used for surrogate or Sobol calculations.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        num_samples = ts.UQ_workflow.num_samples
        aleatory_var_to_dist = ts.aleatory_var_to_dist
        poly_factories = []
        abscissas = np.zeros((num_samples, len(aleatory_var_to_dist)))
        joint_list = []
        for i, (name, distribution) in enumerate(aleatory_var_to_dist.items()):
            if isinstance(distribution, LogNormalDistribution):
                # special transformation of log-->normal since no analytical
                # polynomials for log (was getting error)

                # transform distribution, let Y~LN(mu,sigma). Y=exp(X) for X~N(mu,sigma)
                # then this says the log of Y is normally distributed with mean mu and
                # standard deviation sigma (the log distribution has different expectation)
                # and variance
                ot_dist = ot.Normal(distribution.mu_log, distribution.sigma_log)
                # first center the log distribution by getting rid of the gamma shift
                # then take the natural log, X=ln(Y-gamma)
                x_phys = np.array(self.tradestudy.get_variable(name))
                z = x_phys - distribution.gamma
                if np.any(z) <= 0.0:
                    raise ValueError(
                        f"LogNormal variable '{name}' contains values at or below its "
                        f"shift gamma={distribution.gamma}; cannot apply log transform."
                    )

                new_vals = np.log(z)
                abscissas[:, i] = new_vals
                poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
            else:
                ot_dist = distribution._OT_Dist()
                poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
                abscissas[:, i] = np.array(self.tradestudy.get_variable(name))

            joint_list.append(ot_dist)

        # Data as OT samples
        abscissas = ot.Sample(abscissas)
        outputs = ot.Sample(
            np.array(self.tradestudy.get_variable(output_name))[:, None]
        )

        # Joint distribution based on independent input aleatory dists
        joint = ot.JointDistribution(joint_list)

        # Enumeration function
        enum = ot.LinearEnumerateFunction(len(aleatory_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis = ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order = ts.UQ_workflow.poly_order
        basis_size = enum.getBasisSizeFromTotalDegree(poly_order)
        if basis_size != comb(poly_order + len(aleatory_var_to_dist), poly_order):
            raise ValueError(
                f"Unexpected basis size {basis_size} for poly order {poly_order} and num aleatory {len(aleatory_var_to_dist)}"
            )

        # Selects first P elemens of the polynomial chaos basis according to enumeration function
        truncation_strategy = ot.FixedStrategy(product_basis, basis_size)

        # This is regression based PCE, so projection is done via least squares minimization
        projection_strategy = ot.LeastSquaresStrategy()

        algo = ot.FunctionalChaosAlgorithm(
            abscissas, outputs, joint, truncation_strategy, projection_strategy
        )
        algo.run()
        res = algo.getResult()

        aleatory_var_to_dist = ts.aleatory_var_to_dist

        return AleatorySurrogate(aleatory_var_to_dist, res)


class GaussQuadtraturePCEPost(BasicAleatoryPost):
    """Represent the gaussquadtraturepcepost component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize object state and configuration.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        super().__init__(tradestudy)

    def generate_surrogate(self, output_name):
        """Build and return a polynomial-chaos surrogate for one output variable.

        Parameters
        ----------
        output_name : Any
            Output variable name used for surrogate or Sobol calculations.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        aleatory_var_to_dist = ts.aleatory_var_to_dist
        num_aleatory = len(aleatory_var_to_dist)
        num_samples = (ts.UQ_workflow.poly_order + 1) ** num_aleatory
        # get the weights:
        weights = []
        for result in self.tradestudy._results:
            weights.append(result.state["weight"])
        weights = ot.Point(weights)
        poly_factories = []
        # matrix of input values
        abscissas = np.zeros((num_samples, len(aleatory_var_to_dist)))
        joint_list = []
        for i, (name, distribution) in enumerate(aleatory_var_to_dist.items()):
            if isinstance(distribution, LogNormalDistribution):
                # special transformation of log-->normal since no analytical
                # polynomials for log (was getting error)

                # transform distribution, let Y~LN(mu,sigma). Y=exp(X) for X~N(mu,sigma)
                # then this says the log of Y is normally distributed with mean mu and
                # standard deviation sigma (the log distribution has different expectation)
                # and variance
                ot_dist = ot.Normal(distribution.mu_log, distribution.sigma_log)
                # first center the log distribution by getting rid of the gamma shift
                # then take the natural log, X=ln(Y-gamma)
                x_phys = np.array(self.tradestudy.get_variable(name))
                z = x_phys - distribution.gamma
                if np.any(z) <= 0.0:
                    raise ValueError(
                        f"LogNormal variable '{name}' contains values at or below its "
                        f"shift gamma={distribution.gamma}; cannot apply log transform."
                    )

                new_vals = np.log(z)
                abscissas[:, i] = new_vals
                poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
            else:
                ot_dist = distribution._OT_Dist()
                poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
                abscissas[:, i] = np.array(self.tradestudy.get_variable(name))

            joint_list.append(ot_dist)

        # Data as OT samples
        abscissas = ot.Sample(abscissas)
        outputs = ot.Sample(
            np.array(self.tradestudy.get_variable(output_name))[:, None]
        )

        # Joint distribution based on independent input aleatory dists
        joint = ot.JointDistribution(joint_list)

        # Enumeration function
        enum = ot.LinearEnumerateFunction(len(aleatory_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis = ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order = ts.UQ_workflow.poly_order
        basis_size = enum.getBasisSizeFromTotalDegree(poly_order)
        if basis_size != comb(poly_order + len(aleatory_var_to_dist), poly_order):
            raise ValueError(
                f"Unexpected basis size {basis_size} for poly order {poly_order} and num aleatory {len(aleatory_var_to_dist)}"
            )

        # Selects first P elemens of the polynomial chaos basis according to enumeration function
        truncation_strategy = ot.FixedStrategy(product_basis, basis_size)

        # Gaussian PCE using integration
        projection_strategy = ot.IntegrationStrategy()

        algo = ot.FunctionalChaosAlgorithm(
            abscissas, weights, outputs, joint, truncation_strategy, projection_strategy
        )
        algo.run()
        res = algo.getResult()

        aleatory_var_to_dist = ts.aleatory_var_to_dist

        return AleatorySurrogate(aleatory_var_to_dist, res)


class AleatorySurrogate:
    """Represent the surrogate component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, aleatory_var_to_dist, result):
        # NEED TO ADD A VARIABLE NAME TO DISTRIBUTION FOR SPECIAL TRANSFORMS
        # FROM FAKE SPACE BACK TO PHYSICAL INTERNALLY
        """Initialize object state and configuration.

        Parameters
        ----------
        aleatory_var_to_dist : Any
            Mapping from aleatory variable names to distributions.
        result : Any
            OpenTurns functional-chaos result object.

        """
        self.aleatory_var_to_dist = aleatory_var_to_dist
        self._ot_res = result
        self.new_sample_points = None
        self.new_sample_values = None
        self._Chaos_SI=ot.FunctionalChaosSobolIndices(self._ot_res)

    def get_first_order_sobol(self,var_name):
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            if var_name==name:
                return self._Chaos_SI.getSobolIndex(i)

    def get_total_order_sobol(self,var_name):
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            if var_name==name:
                return self._Chaos_SI.getSobolTotalIndex(i)

    def get_interaction_sobol(self,var_names:list):
        inds=[]
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        return self._Chaos_SI.getSobolGroupedIndex(inds)

    def get_total_interaction_sobol(self,var_names:list):
        inds=[]
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        return self._Chaos_SI.getSobolGroupedTotalIndex(inds)

    def plot_total_order_sobol(self,plot_title="",save_dir=None):
        name_to_inds={}
        names=[]
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            name_to_inds[name]=self._Chaos_SI.getSobolTotalIndex(i)
            names.append(name.split('.')[-1])
        fig,ax=plt.subplots(figsize=(10,6))
        ax.grid(True,which="both",linestyle="--",linewidth=0.5)
        ax.bar(names,list(name_to_inds.values()))
        ax.set_xlabel(f'Variable Names')
        ax.set_ylabel(f'Sensitivity Metrics')

        if plot_title=="":
            plot_title = f'Total Sensitivity Indices of Surrogate'
        ax.set_title(plot_title)
        if save_dir is not None:
            plt.savefig(
                rf"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def predict(self, name_to_value_dict):
        # Note, name_to_value_dict should be PHYSICAL inputs, not
        # probabalistic based inputs
        """Evaluate the surrogate at one dictionary of input variable values.

        Parameters
        ----------
        name_to_value_dict : Any
            Mapping from variable names to surrogate input values.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        vals = []
        for name, distribution in self.aleatory_var_to_dist.items():
            if isinstance(distribution, LogNormalDistribution):
                transformed_val = np.log(name_to_value_dict[name] - distribution.gamma)
                vals.append(transformed_val)
            else:
                vals.append(name_to_value_dict[name])
        point = ot.Point(vals)
        return self._ot_res.getMetaModel()(point)

    @property
    def analytical_mean(self):
        """Execute analytical mean.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return ot.FunctionalChaosRandomVector(self._ot_res).getMean()[0]

    @property
    def analytical_variance(self):
        """Execute analytical variance.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        ot_var = np.array(ot.FunctionalChaosRandomVector(self._ot_res).getCovariance())[
            0
        ][0]
        coeffs = np.array(self._ot_res.getCoefficients())
        # skip first term which is mean
        variance = np.sum(coeffs[1:] ** 2)
        if np.abs(ot_var - variance) > 10**-6:
            raise ValueError(
                f"Analytical variance mismatch: OpenTurns returned {ot_var}, "
                f"but the coefficient-based variance is {variance}."
            )
        return variance

    @property
    def analytical_std(self):
        """Execute analytical std.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return np.sqrt(self.analytical_variance)

    def sample(self, num_samples, method="MC"):
        """Generate new surrogate samples using a selected sampling strategy.

        Parameters
        ----------
        num_samples : Any
            Number of samples to generate.
        method : Any, optional
            Method identifier controlling algorithm behavior.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        model = self._ot_res.getMetaModel()
        # need to add method here?
        dist = self._ot_res.getDistribution()
        if method == "MC":
            experiment = ot.MonteCarloExperiment(dist, num_samples)
            self.new_sample_points = np.array(experiment.generate())
        elif method == "LHS":
            experiment = ot.LHSExperiment(dist, num_samples)
            self.new_sample_points = np.array(experiment.generate())
        elif method == "Sobol":
            sobolseq = ot.SobolSequence(dist.getDimension())
            experiment = ot.LowDiscrepancyExperiment(sobolseq, dist, num_samples)
            self.new_sample_points = np.array(experiment.generate())
        elif method == "Halton":
            haltonseq = ot.HaltonSequence(dist.getDimension())
            experiment = ot.LowDiscrepancyExperiment(haltonseq, dist, num_samples)
            self.new_sample_points = np.array(experiment.generate())

        Y = model(self.new_sample_points)

        # NOTE, SAMPLE POINTS RETURNED ARE PHYSICAL POINTS, NOT Probability space points
        # in the case of the lognormal distribution. Basic Uniform and Normal
        # probability space is equal to the physical since no transformaton occurs.
        for i, (name, distribution) in enumerate(self.aleatory_var_to_dist.items()):
            if isinstance(distribution, LogNormalDistribution):
                # points are in normal space X~N(mu,sigma) transform to Y=exp(X)+gamma
                # where gamma is the shift
                self.new_sample_points[:, i] = (
                    np.exp(self.new_sample_points[:, i]) + distribution.gamma
                )

        self.new_sample_values = np.array(Y)
        return np.array(self.new_sample_points), np.array(self.new_sample_values)

    def plot_cdf(self, xlabel=None,plot_title="",save_dir=None):
        """Plot a CDF from previously generated surrogate samples.

        Parameters
        ----------
        save_dir : Any, optional
            Optional directory where plots are saved.

        Returns
        -------
        Any
            Plot artifact (axes object or plotted data).

        """
        if self.new_sample_values is None:
            raise ValueError("Must generate samples before plotting cdf")
        vals = self.new_sample_values.flatten()
        ecdf = stats.ecdf(vals)
        quantiles = ecdf.cdf.quantiles
        probabilities = ecdf.cdf.probabilities
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(quantiles, probabilities, color="blue")
        if plot_title=="":
            plot_title=f"CDF of surrogate output"
        if xlabel is None:
            xlabel="Output value"

        ax.set_title(plot_title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Probability", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png", dpi=300, bbox_inches="tight"
            )

        return np.hstack((quantiles, probabilities))

    def plot_pdf(self, bins=20, xlabel=None, plot_title="", save_dir=None):
        """Plot a PDF from previously generated surrogate samples.

        Parameters
        ----------
        bins : Any, optional
            Number of bins used for histogram-based plots.
        save_dir : Any, optional
            Optional directory where plots are saved.

        Returns
        -------
        Any
            Plot artifact (axes object or plotted data).

        """
        if self.new_sample_values is None:
            raise ValueError("Must generate samples before plotting pdf")
        vals = self.new_sample_values.flatten()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(vals, bins=bins, color="blue", edgecolor="black")
        if plot_title == "":
            plot_title = "PDF of surrogate output"
        if xlabel is None:
            xlabel = "Output value"
        ax.set_title(plot_title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

        return ax


class SobolSensitivityPost(BasicAleatoryPost):
    """Represent the sobolsensitivitypost component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    _METHOD_TO_CLASS = {
        "Saltelli": ot.SaltelliSensitivityAlgorithm,
        "Jansen": ot.JansenSensitivityAlgorithm,
        "MauntzKucherenko": ot.MauntzKucherenkoSensitivityAlgorithm,
        "Martinez": ot.MartinezSensitivityAlgorithm,
    }

    def __init__(self, tradestudy):
        """Execute init.

        Parameters
        ----------
        tradestudy : Any
            Input value for `tradestudy`.

        """
        super().__init__(tradestudy)

    def generate_Sobol_indices(self, output_name, algo="Jansen"):
        # Output values
        """Compute Sobol indices for one model output.

        Parameters
        ----------
        output_name : Any
            Output variable name used for surrogate or Sobol calculations.
        algo : Any, optional
            Sobol index algorithm name supported by OpenTurns.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        outs = np.array(self.tradestudy.get_variable(output_name))[:, None]

        # Overarching UQ distributor class
        UQ_Distrubutor = self.tradestudy.study.control.trade_studies[
            self.tradestudy.trade_name
        ]
        aleatory_dict = UQ_Distrubutor.aleatory_var_to_dist

        # Samples run
        num_samples = UQ_Distrubutor.UQ_workflow.num_samples
        # Number of aleatory vars
        # Calculates the number of samples needed to preallocate array shape
        second_order = UQ_Distrubutor.UQ_workflow.computeSecondOrder
        num_aleatory = len(aleatory_dict)
        if not second_order or num_aleatory == 2:
            num_sobol_samples = num_samples * (2 + num_aleatory)
        else:
            num_sobol_samples = num_samples * (2 + 2 * num_aleatory)

        input_matrix = np.zeros((num_sobol_samples, num_aleatory))
        name_to_col = {}
        col_to_name = {}
        for col, aleatory_var_name in enumerate(aleatory_dict.keys()):
            name_to_col[aleatory_var_name] = col
            col_to_name[col] = aleatory_var_name
            input_matrix[:, col] = self.tradestudy.get_variable(aleatory_var_name)
        input_matrix = ot.Sample(input_matrix)
        outs = ot.Sample(outs)
        ot_algo = self._METHOD_TO_CLASS[algo]
        otSobol = ot_algo(input_matrix, outs, num_samples)

        results = {"first_order": {}, "second_order": {}, "total_order": {}}

        # First order indices, one output per input distribution
        first_order = otSobol.getFirstOrderIndices()
        for name, col in name_to_col.items():
            results["first_order"][name] = first_order[col]

        # Returns symmetric matrix (A=A^t) of all S(i,j) pairs
        if second_order:
            second_order = otSobol.getSecondOrderIndices()
            for i in range(num_aleatory):
                for j in range(i + 1, num_aleatory):
                    name_i = col_to_name[i]
                    name_j = col_to_name[j]
                    results["second_order"][(name_i, name_j)] = float(
                        second_order[i, j]
                    )

        # Total: Si+Sum(j,d)(Si,j)(j!=i)+Sum(j<=k)(Si,j,k)(i!=j<k)+...
        # one variable per input distribution
        total_order = otSobol.getTotalOrderIndices()
        for name, col in name_to_col.items():
            results["total_order"][name] = total_order[col]

        #should add plotting so its easy to visualize relative sensitivity

        return results, otSobol

    def get_first_order_indices(self, output_name, algo="Jansen"):
        """Return first-order Sobol indices for one output.

        Parameters
        ----------
        output_name : Any
            Output variable name used for surrogate or Sobol calculations.
        algo : Any, optional
            Sobol index algorithm name supported by OpenTurns.

        Returns
        -------
        Any
            Requested value.

        """
        results, _ = self.generate_Sobol_indices(output_name, algo)
        return results["first_order"]

    def get_second_order_indices(self, output_name, algo="Jansen"):
        """Return second-order Sobol indices for one output.

        Parameters
        ----------
        output_name : Any
            Output variable name used for surrogate or Sobol calculations.
        algo : Any, optional
            Sobol index algorithm name supported by OpenTurns.

        Returns
        -------
        Any
            Requested value.

        """
        UQ_Distrubutor = self.tradestudy.study.control.trade_studies[
            self.tradestudy.trade_name
        ]
        if not UQ_Distrubutor.UQ_workflow.computeSecondOrder:
            raise ValueError(f"This run did not support second order Sobol")

        results, _ = self.generate_Sobol_indices(output_name, algo)

        return results["second_order"]

    def get_total_order_indices(self, output_name, algo="Jansen"):
        """Return total-order Sobol indices for one output.

        Parameters
        ----------
        output_name : Any
            Output variable name used for surrogate or Sobol calculations.
        algo : Any, optional
            Sobol index algorithm name supported by OpenTurns.

        Returns
        -------
        Any
            Requested value.

        """
        results, _ = self.generate_Sobol_indices(output_name, algo)
        return results["total_order"]

    def plot_total_order_indices(self,output_name,plot_title="",algo="Jansen",save_dir=None):
        results,_=self.generate_Sobol_indices(output_name,algo)
        total_results=results["total_order"]
        fig,ax=plt.subplots(figsize=(10,6))
        names=[name.split('.')[-1] for name in list(total_results.keys())]
        values=list(total_results.values())
        ax.grid(True,which="both",linestyle="--",linewidth=0.5)
        ax.bar(names,values)
        ax.set_xlabel(f'Variable Names')
        ax.set_ylabel(f'Sensitivity Metrics')
        if plot_title=="":
            plot_title = f'Total Sensitivity Indices of {output_name}'
        ax.set_title(plot_title)
        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

#--------------------------Epistemic Post Process Classes
class BasicEpistemicPost:
    def __init__(self,tradestudy:Tradestudy):
        self.tradestudy=tradestudy

    def compute_bounds(self,var_name):
        vals = np.array(self.tradestudy.get_variable(var_name), dtype=float)
        #max loc
        max_val=np.max(vals)
        max_loc=np.argmax(vals)
        max_state=self.tradestudy._results[max_loc].state
        #min loc
        min_val=np.min(vals)
        min_loc=np.argmin(vals)
        min_state=self.tradestudy._results[min_loc].state

        return {
            'maximum':{
                'value':max_val,
                'state':max_state
            },
            'minimum':{
                'value':min_val,
                'state':min_state
            }
        }
#--------------------------Mixed Aleatory-Epistemic Post Process Classes
class BasicMixedPost:
    def __init__(self, tradestudy: Tradestudy):
        """Initialize object state and configuration.
        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.
        """
        self.tradestudy = tradestudy

    def _generate_output_var_data(self,var_name):
        trade_name=self.tradestudy.trade_name
        trade_base=self.tradestudy.study.control.trade_studies[trade_name]
        #special case for IVP problems with ad hoc state generation
        ivp_meta=[]
        meta_to_epistemic_state={}
        epistemic_vars=list(trade_base.epistemic_var_to_vals.keys())
        for result in self.tradestudy._results:
            state_meta=int(result.state["__IVP__STATE__META"])
            ivp_meta.append(int(state_meta))
            if state_meta not in meta_to_epistemic_state:
                epistemic_state={}
                for epistemic_var in epistemic_vars:
                    epistemic_state[epistemic_var]=result.state[epistemic_var]
                meta_to_epistemic_state[state_meta]=epistemic_state

        ivp_meta=np.array(ivp_meta)

        u,inv=np.unique(ivp_meta,return_inverse=True)
        indices_per_value = {
            val: np.where(inv == i)[0]
            for i, val in enumerate(u)
            }

        out_vals=np.array(self.tradestudy.get_variable(var_name),dtype=float)
        if not np.all(np.isfinite(out_vals)):
            num_bad=int(np.size(out_vals)-np.count_nonzero(np.isfinite(out_vals)))
            raise ValueError(
                f"Found {num_bad} non-finite values for '{var_name}' in IVP results. "
                "Cannot compute ECDF-based IVP post metrics with NaN/Inf values."
            )
        meta_to_cdf={}

        #numbers of columns is the epistemic states
        num_cols=len(meta_to_epistemic_state)
        #number of rows is total states/epistemic states
        if not ivp_meta.shape[0]%num_cols==0:
            raise ValueError(
                "IVP result grouping is inconsistent: the total number of states "
                f"({ivp_meta.shape[0]}) is not divisible by the number of "
                f"epistemic states ({num_cols})."
            )

        num_rows=int(ivp_meta.shape[0]/num_cols)
        quantiles_data=np.zeros((num_rows,num_cols))
        probabilities_data=np.zeros((num_rows,num_cols))
        for epistemic_meta,inds in indices_per_value.items():
            vals=out_vals[inds]
            ecdf=stats.ecdf(vals)
            meta_to_cdf[epistemic_meta]=ecdf
            quantiles_data[:,epistemic_meta]=ecdf.cdf.quantiles
            probabilities_data[:,epistemic_meta]=ecdf.cdf.probabilities

        return meta_to_cdf,quantiles_data,probabilities_data


    def plot_pbox(self,var_name,x_axis_title=None,plot_title="",save_dir=None):

        _,quantiles_data,probabilities_data=self._generate_output_var_data(var_name=var_name)

        max_quantiles=np.max(quantiles_data,axis=1)
        #gives column with max
        max_quantiles_locs=np.argmax(quantiles_data,axis=1)
        max_quantile_probabilities=probabilities_data[np.arange(max_quantiles.shape[0]),max_quantiles_locs]

        min_quantiles=np.min(quantiles_data,axis=1)
        min_quantiles_locs=np.argmin(quantiles_data,axis=1)
        min_quantile_probabilities=probabilities_data[np.arange(min_quantiles.shape[0]),min_quantiles_locs]

        #plot data
        fig,ax=plt.subplots(figsize=(10, 6))
        ax.plot(max_quantiles,max_quantile_probabilities,color="red",label="max envelope")
        ax.plot(min_quantiles,min_quantile_probabilities,color='blue',label="min envelope")
        if x_axis_title is None:
            x_axis_title=var_name
        if plot_title=="":
            plot_title= f"P-box Plot of {x_axis_title}"
        ax.set_title(plot_title,fontsize=16,fontweight="bold")
        ax.set_xlabel(x_axis_title,fontsize=14)
        ax.set_ylabel('Probability',fontsize=14)
        ax.grid(True,which="both",linestyle="--",linewidth=0.5)
        ax.legend()

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def get_plevel_bounds(self,var_name,p_level):
        _,quantiles,probabilities=self._generate_output_var_data(var_name)
        if p_level<0.0 or p_level>1.0:
            raise ValueError(
                f"p_level must be between 0 and 1 inclusive; received {p_level}."
            )


        #get probability row, note structure of creation results
        # in consistent probability matrix (each column is identical):
        probabilities=probabilities[:,0]
        le=np.where(probabilities<=p_level)[0]
        ge=np.where(probabilities>=p_level)[0]

        if le.size==0:
            # clamp to left boundary
            min_loc=0
            max_loc=0
        elif ge.size==0:
            # clamp to right boundary
            min_loc=probabilities.shape[0]-1
            max_loc=probabilities.shape[0]-1
        else:
            min_loc=le[-1]
            max_loc=ge[0]

        if min_loc==max_loc:
            max_quantile=np.max(quantiles[max_loc])
            min_quantile=np.min(quantiles[min_loc])
        else:
            #handle interpolation of curves first,
            #then take envelope, handle edge cases
            #handle max
            low_mat=quantiles[min_loc]
            high_mat=quantiles[max_loc]
            numerator=(high_mat-low_mat)*(p_level-probabilities[min_loc])
            denominator=(probabilities[max_loc]-probabilities[min_loc])
            interp_curves=numerator/denominator+low_mat

            max_quantile=np.max(interp_curves)
            min_quantile=np.min(interp_curves)

        return (min_quantile,max_quantile)


    def plot_horsetails(self,var_name,x_axis_title=None,plot_title="",save_dir=None):
        _,quantiles,probabilities=self._generate_output_var_data(var_name)
        fig,ax=plt.subplots(figsize=(10, 6))
        ax.plot(quantiles,probabilities,color="blue")
        if x_axis_title is None:
            x_axis_title=var_name
        if plot_title=="":
            plot_title= f"Horsetail Plot of {x_axis_title}"
        ax.set_title(plot_title,fontsize=16,fontweight="bold")
        ax.set_xlabel(x_axis_title,fontsize=14)
        ax.set_ylabel('Probability',fontsize=14)
        ax.grid(True,which="both",linestyle="--",linewidth=0.5)


        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )


class MixedRegressionPCEPost(BasicAleatoryPost):
    def __init__(self, tradestudy: Tradestudy):
        super().__init__(tradestudy)

    def _get_mixed_var_to_dist(self):
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        mixed_var_to_dist = {}
        for name, interval in ts.epistemic_var_to_vals.items():
            mixed_var_to_dist[name] = UniformDistribution(interval[0], interval[1])
        for name, distribution in ts.aleatory_var_to_dist.items():
            mixed_var_to_dist[name] = distribution
        return mixed_var_to_dist

    def generate_surrogate(self, output_name):
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        num_samples = ts.UQ_workflow.num_samples
        all_var_to_dist = self._get_mixed_var_to_dist()
        poly_factories = []
        abscissas = np.zeros((num_samples, len(all_var_to_dist)))
        joint_list = []
        for i, (name, distribution) in enumerate(all_var_to_dist.items()):
            if isinstance(distribution, LogNormalDistribution):
                # special transformation of log-->normal since no analytical
                # polynomials for log (was getting error)

                # transform distribution, let Y~LN(mu,sigma). Y=exp(X) for X~N(mu,sigma)
                # then this says the log of Y is normally distributed with mean mu and
                # standard deviation sigma (the log distribution has different expectation)
                # and variance
                ot_dist = ot.Normal(distribution.mu_log, distribution.sigma_log)
                # first center the log distribution by getting rid of the gamma shift
                # then take the natural log, X=ln(Y-gamma)
                x_phys = np.array(self.tradestudy.get_variable(name))
                z = x_phys - distribution.gamma
                if np.any(z) <= 0.0:
                    raise ValueError(
                        f"LogNormal variable '{name}' contains values at or below its "
                        f"shift gamma={distribution.gamma}; cannot apply log transform."
                    )

                new_vals = np.log(z)
                abscissas[:, i] = new_vals
                poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
            else:
                ot_dist = distribution._OT_Dist()
                poly_factories.append(ot.StandardDistributionPolynomialFactory(ot_dist))
                abscissas[:, i] = np.array(self.tradestudy.get_variable(name))

            joint_list.append(ot_dist)

        # Data as OT samples
        abscissas = ot.Sample(abscissas)
        outputs = ot.Sample(
            np.array(self.tradestudy.get_variable(output_name))[:, None]
        )

        # Joint distribution based on independent input aleatory dists
        joint = ot.JointDistribution(joint_list)

        # Enumeration function
        enum = ot.LinearEnumerateFunction(len(all_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis = ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order = ts.UQ_workflow.poly_order
        basis_size = enum.getBasisSizeFromTotalDegree(poly_order)
        if basis_size != comb(poly_order + len(all_var_to_dist), poly_order):
            raise ValueError(
                f"Unexpected basis size {basis_size} for poly order {poly_order} and num aleatory {len(all_var_to_dist)}"
            )

        # Selects first P elemens of the polynomial chaos basis according to enumeration function
        truncation_strategy = ot.FixedStrategy(product_basis, basis_size)

        # This is regression based PCE, so projection is done via least squares minimization
        projection_strategy = ot.LeastSquaresStrategy()

        algo = ot.FunctionalChaosAlgorithm(
            abscissas, outputs, joint, truncation_strategy, projection_strategy
        )
        algo.run()
        res = algo.getResult()

        epistemic_var_to_interval=ts.epistemic_var_to_vals
        aleatory_var_to_dist=ts.aleatory_var_to_dist

        return MixedSurrogate(all_var_to_dist, res,epistemic_var_to_interval,aleatory_var_to_dist)

class MixedSurrogate:

    _META_STATE_NAME="__IVP__STATE__META"
    def __init__(self, all_var_to_dist, result,epistemic_var_to_interval,
                 aleatory_var_to_dist):
        # NEED TO ADD A VARIABLE NAME TO DISTRIBUTION FOR SPECIAL TRANSFORMS
        # FROM FAKE SPACE BACK TO PHYSICAL INTERNALLY
        """Initialize object state and configuration.
        Parameters
        ----------
        aleatory_var_to_dist : Any
            Mapping from aleatory variable names to distributions.
        result : Any
            OpenTurns functional-chaos result object.
        """
        self.epistemic_var_to_interval=epistemic_var_to_interval
        self.aleatory_var_to_dist=aleatory_var_to_dist
        self.all_var_to_dist = all_var_to_dist
        self._ot_res = result
        self.new_sample_points = None
        self.new_sample_values = None
        self._Chaos_SI=ot.FunctionalChaosSobolIndices(self._ot_res)


    def predict(self, name_to_value_dict):
        # Note, name_to_value_dict should be PHYSICAL inputs, not
        # probabalistic based inputs
        """Evaluate the surrogate at one dictionary of input variable values.
        Parameters
        ----------
        name_to_value_dict : Any
            Mapping from variable names to surrogate input values.
        Returns
        -------
        Any
            Return value produced by this method.
        """
        vals = []
        for name, distribution in self.all_var_to_dist.items():
            if isinstance(distribution, LogNormalDistribution):
                transformed_val = np.log(name_to_value_dict[name] - distribution.gamma)
                vals.append(transformed_val)
            else:
                vals.append(name_to_value_dict[name])
        point = ot.Point(vals)
        return self._ot_res.getMetaModel()(point)


    def sample(self, num_epistemic_samples,
                num_aleatory_samples,method="MC"):
        """Generate new surrogate samples using a selected sampling strategy.
        Parameters
        ----------
        num_samples : Any
            Number of samples to generate.
        method : Any, optional
            Method identifier controlling algorithm behavior.
        Returns
        -------
        Any
            Return value produced by this method.
        """
        aleatory_to_distribution=self.aleatory_var_to_dist
        epistemic_to_vals=self.epistemic_var_to_interval
        all_var_to_dists=self.all_var_to_dist
        model = self._ot_res.getMetaModel()

        local_epistemic_to_distribution={}
        for name,dist in all_var_to_dists.items():
            if name in epistemic_to_vals:
                local_epistemic_to_distribution[name]=dist

        local_aleatory_to_distribution={}
        for name,dist in all_var_to_dists.items():
            if name in aleatory_to_distribution:
                local_aleatory_to_distribution[name]=dist

        if method == "MC":
            SamplingClass=MonteCarloSampler
        elif method == "LHS":
            SamplingClass=LatinHypercubeSampler
        elif method == "Sobol":
            SamplingClass=SobolSampler
        elif method == "Halton":
            SamplingClass=HaltonSampler
        else:
            raise ValueError(
                f"Unsupported sampling method '{method}'. "
                "Expected one of: 'MC', 'LHS', 'Sobol', 'Halton'."
            )

        epistemic_samples, epistemic_var_name_to_col =SamplingClass.sample(
            local_epistemic_to_distribution,num_epistemic_samples
            )

        epistemic_with_meta=np.hstack((np.arange(epistemic_samples.shape[0])[:,None],epistemic_samples,))
        #then generate aleatory inner loop
        all_aleatory_states=np.zeros((num_aleatory_samples*epistemic_samples.shape[0],
                                      len(aleatory_to_distribution)))
        row=0
        for i in range(num_epistemic_samples):
            aleatory_samples, aleatory_var_name_to_col=SamplingClass.sample(
               local_aleatory_to_distribution,num_aleatory_samples
            )
            all_aleatory_states[row:row+num_aleatory_samples,:]=aleatory_samples
            row+=num_aleatory_samples

        all_vars_to_col={self._META_STATE_NAME:0}
        for name,col in epistemic_var_name_to_col.items():
            all_vars_to_col[name]=col+1

        for name,col in aleatory_var_name_to_col.items():
            #account for meta offset
            all_vars_to_col[name]=col+(len(epistemic_var_name_to_col)+1)

        epistemic_states=np.repeat(epistemic_with_meta,
                                   num_aleatory_samples,
                                   axis=0)

        #do not include meta data for sample
        self._new_sample_points_with_meta=epistemic_states
        self.new_sample_points=np.hstack((epistemic_states[:,1:],all_aleatory_states))

        Y = model(self.new_sample_points)
        # NOTE, SAMPLE POINTS RETURNED ARE PHYSICAL POINTS, NOT Probability space points
        # in the case of the lognormal distribution. Basic Uniform and Normal
        # probability space is equal to the physical since no transformaton occurs.
        for i, (name, distribution) in enumerate(self.all_var_to_dist.items()):
            if isinstance(distribution, LogNormalDistribution):
                # points are in normal space X~N(mu,sigma) transform to Y=exp(X)+gamma
                # where gamma is the shift
                self.new_sample_points[:, i] = (
                    np.exp(self.new_sample_points[:, i]) + distribution.gamma
                )

        self.new_sample_values = np.array(Y)

        return np.array(self.new_sample_points), np.array(self.new_sample_values)


    def _aggregate_predicted_vals(self):
        ivp_meta=self._new_sample_points_with_meta[:,0]
        u,inv=np.unique(ivp_meta,return_inverse=True)
        indices_per_value = {
            val: np.where(inv == i)[0]
            for i, val in enumerate(u)
            }
        out_vals=self.new_sample_values
        num_cols=u.shape[0]
        num_rows=int(ivp_meta.shape[0]/num_cols)
        quantiles_data=np.zeros((num_rows,num_cols))
        probabilities_data=np.zeros((num_rows,num_cols))
        meta_to_cdf={}
        for epistemic_meta,inds in indices_per_value.items():
            vals=out_vals[inds][:,0]
            ecdf=stats.ecdf(vals)
            meta_to_cdf[epistemic_meta]=ecdf
            epistemic_meta=int(epistemic_meta)
            quantiles_data[:,epistemic_meta]=ecdf.cdf.quantiles
            probabilities_data[:,epistemic_meta]=ecdf.cdf.probabilities

        return meta_to_cdf,quantiles_data,probabilities_data

    def plot_horsetails(self,xlabel=None,plot_title="", save_dir=None):
        """Plot a CDF from previously generated surrogate samples.

        Parameters
        ----------
        save_dir : Any, optional
            Optional directory where plots are saved.

        Returns
        -------
        Any
            Plot artifact (axes object or plotted data).

        """
        if self.new_sample_values is None:
            raise ValueError("Must generate samples before plotting cdf")
        _,quantiles,probabilities=self._aggregate_predicted_vals()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(quantiles, probabilities, color="blue")
        if plot_title=="":
            plot_title=f"Horsetail plot of surrogate output"
        if xlabel is None:
            xlabel="Output value"

        ax.set_title(plot_title,fontsize=16,fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Probability", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)


        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png", dpi=300, bbox_inches="tight"
            )

    def plot_pbox(self, xlabel=None, plot_title="", save_dir=None):

        _,quantiles_data,probabilities_data=self._aggregate_predicted_vals()

        max_quantiles=np.max(quantiles_data,axis=1)
        #gives column with max
        max_quantiles_locs=np.argmax(quantiles_data,axis=1)
        max_quantile_probabilities=probabilities_data[np.arange(max_quantiles.shape[0]),max_quantiles_locs]

        min_quantiles=np.min(quantiles_data,axis=1)
        min_quantiles_locs=np.argmin(quantiles_data,axis=1)
        min_quantile_probabilities=probabilities_data[np.arange(min_quantiles.shape[0]),min_quantiles_locs]

        #plot data
        fig,ax=plt.subplots(figsize=(10, 6))
        ax.plot(max_quantiles,max_quantile_probabilities,color="red",label="max envelope")
        ax.plot(min_quantiles,min_quantile_probabilities,color='blue',label="min envelope")

        if plot_title == "":
            plot_title = "P-box Plot of surrogate model"
        if xlabel is None:
            xlabel = "Output value"
        ax.set_title(plot_title,fontsize=16,fontweight="bold")
        ax.set_xlabel(xlabel,fontsize=14)
        ax.set_ylabel('Probability',fontsize=14)
        ax.grid(True,which="both",linestyle="--",linewidth=0.5)
        ax.legend()

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def get_plevel_bounds(self,p_level):
        _,quantiles,probabilities=self._aggregate_predicted_vals()
        if p_level<0.0 or p_level>1.0:
            raise ValueError(
                f"p_level must be between 0 and 1 inclusive; received {p_level}."
            )


        #get probability row, note structure of creation results
        # in consistent probability matrix (each column is identical):
        probabilities=probabilities[:,0]
        le=np.where(probabilities<=p_level)[0]
        ge=np.where(probabilities>=p_level)[0]

        if le.size==0:
            # clamp to left boundary
            min_loc=0
            max_loc=0
        elif ge.size==0:
            # clamp to right boundary
            min_loc=probabilities.shape[0]-1
            max_loc=probabilities.shape[0]-1
        else:
            min_loc=le[-1]
            max_loc=ge[0]

        if min_loc==max_loc:
            max_quantile=np.max(quantiles[max_loc])
            min_quantile=np.min(quantiles[min_loc])
        else:
            #handle interpolation of curves first,
            #then take envelope, handle edge cases
            #handle max
            low_mat=quantiles[min_loc]
            high_mat=quantiles[max_loc]
            numerator=(high_mat-low_mat)*(p_level-probabilities[min_loc])
            denominator=(probabilities[max_loc]-probabilities[min_loc])
            interp_curves=numerator/denominator+low_mat

            max_quantile=np.max(interp_curves)
            min_quantile=np.min(interp_curves)

        return (min_quantile,max_quantile)

    def get_first_order_sobol(self,var_name):
        for i,name in enumerate(self.all_var_to_dist.keys()):
            if var_name==name:
                return self._Chaos_SI.getSobolIndex(i)

    def get_total_order_sobol(self,var_name):
        for i,name in enumerate(self.all_var_to_dist.keys()):
            if var_name==name:
                return self._Chaos_SI.getSobolTotalIndex(i)

    def get_interaction_sobol(self,var_names:list):
        inds=[]
        for i,name in enumerate(self.all_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        return self._Chaos_SI.getSobolGroupedIndex(inds)

    def get_total_interaction_sobol(self,var_names:list):
        inds=[]
        for i,name in enumerate(self.all_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        return self._Chaos_SI.getSobolGroupedTotalIndex(inds)

    def plot_total_order_sobol(self,plot_title="",save_dir=None):
        name_to_inds={}
        names=[]
        for i,name in enumerate(self.all_var_to_dist.keys()):
            name_to_inds[name]=self._Chaos_SI.getSobolTotalIndex(i)
            names.append(name.split('.')[-1])
        fig,ax=plt.subplots(figsize=(10,6))
        ax.grid(True,which="both",linestyle="--",linewidth=0.5)
        ax.bar(names,list(name_to_inds.values()))
        ax.set_xlabel(f'Variable Names')
        ax.set_ylabel(f'Sensitivity Metrics')

        if plot_title=="":
            plot_title = f'Total Sensitivity Indices of Surrogate'
        ax.set_title(plot_title)
        if save_dir is not None:
            plt.savefig(
                rf"{save_dir}/{plot_title.replace(' ','_')}.png",
                dpi=300,
                bbox_inches="tight",
            )