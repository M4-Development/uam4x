from __future__ import annotations
import matplotlib.pyplot as plt
import openturns as ot
import numpy as np
from uam4x.UQ import LatinHypercubeSampler, SobolSampler,MonteCarloSampler,\
    HaltonSampler,NormalDistribution,UniformDistribution,LogNormalDistribution
import scipy.stats as stats
from math import comb
import sympy as sp
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split



#--------------------------Basic Post Process Class
class BasicPost:
    """Provide basic post-processing utilities for a trade study.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize a basic post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        self.tradestudy = tradestudy
        trade_name=self.tradestudy.trade_name
        self.post_storage=tradestudy.study._post_data[trade_name]

    def scatter(
        self, var1, var2, xlabel=None, ylabel=None, title="default_title", save_dir=None
    ):
        """Plot a scatter chart for two trade-study variables.

        Parameters
        ----------
        var1 : str
            Variable name to plot on the x-axis.
        var2 : str
            Variable name to plot on the y-axis.
        xlabel : str, optional
            Label for the x-axis. If omitted, ``var1`` is used.
        ylabel : str, optional
            Label for the y-axis. If omitted, ``var2`` is used.
        title : str, optional
            Plot title and, when saving, the base filename.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object for the generated scatter plot.

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
        """Fit a linear regression between two trade-study variables.

        Parameters
        ----------
        var1 : str
            Variable name used as the independent variable.
        var2 : str
            Variable name used as the dependent variable.

        Returns
        -------
        slope : float
            Slope of the fitted regression line.
        intercept : float
            Intercept of the fitted regression line.
        r_value : float
            Pearson correlation coefficient.
        p_value : float
            Two-sided p-value for a hypothesis test whose null hypothesis is
            that the slope is zero.
        std_err : float
            Standard error of the estimated slope.

        """
        x_values = self.tradestudy.get_variable(var1)
        y_values = self.tradestudy.get_variable(var2)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        return slope, intercept, r_value, p_value, std_err



#--------------------------Aleatory Post Process Classes
# Define post process classes:
class BasicAleatoryPost:
    """Post-process aleatory trade-study results.

    This class provides descriptive statistics, empirical distribution tools,
    and correlation metrics for variables stored in a completed aleatory trade
    study.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize an aleatory post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        self.tradestudy = tradestudy
        trade_name=self.tradestudy.trade_name
        self.post_storage=tradestudy.study._post_data[trade_name]

    def compute_mean(self, var_name):
        """Compute the sample mean for a variable across trade-study results.

        Parameters
        ----------
        var_name : str
            Variable name to analyze.

        Returns
        -------
        float
            Arithmetic mean of the stored variable values.

        """
        vals = self.tradestudy.get_variable(var_name)
        return np.average(vals)

    def compute_variance(self, var_name, method="sample"):
        """Compute the sample or population variance for a variable.

        Parameters
        ----------
        var_name : str
            Variable name to analyze.
        method : {"sample", "s", "population", "pop", "p"}, optional
            Variance convention. Sample variance uses ``ddof=1`` and population
            variance uses ``ddof=0``.

        Returns
        -------
        float
            Variance of the stored variable values.

        Raises
        ------
        ValueError
            If ``method`` is not a supported variance convention.

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
        var_name : str
            Variable name to analyze.
        method : {"sample", "s", "population", "pop", "p"}, optional
            Standard-deviation convention. Sample standard deviation uses
            ``ddof=1`` and population standard deviation uses ``ddof=0``.

        Returns
        -------
        float
            Standard deviation of the stored variable values.

        Raises
        ------
        ValueError
            If ``method`` is not a supported standard-deviation convention.

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
        var_name : str
            Variable name to analyze.
        bias : bool, optional
            If False, apply SciPy's bias correction.

        Returns
        -------
        float
            Skewness of the stored variable values.

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        return stats.skew(vals, nan_policy="raise", bias=bias)

    def compute_kurtosis(self, var_name, fisher=True, bias=True):
        # uses fisher or pearson
        """Compute kurtosis for a variable using SciPy conventions.

        Parameters
        ----------
        var_name : str
            Variable name to analyze.
        fisher : bool, optional
            If True, use Fisher's definition where a normal distribution has
            kurtosis 0. If False, use Pearson's definition where a normal
            distribution has kurtosis 3.
        bias : bool, optional
            If False, apply SciPy's bias correction.

        Returns
        -------
        float
            Kurtosis of the stored variable values.

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
        """Plot an estimated probability density for a variable.

        Parameters
        ----------
        var_name : str
            Variable name to analyze.
        x_axis_title : str, optional
            Label for the x-axis. If omitted, ``var_name`` is used.
        plot_title : str, optional
            Plot title. If omitted, a title is generated from ``var_name``.
        method : {"histogram", "h", "kde", "kernel_density_estimate", "kernel", "k"}, optional
            Density estimation method. Histogram methods plot counts; kernel
            methods plot a Gaussian KDE.
        bins : int, optional
            Number of bins used for histogram-based plots.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object for the generated PDF plot.

        Raises
        ------
        ValueError
            If ``method`` is not a supported PDF plotting method.

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
        """Plot the empirical cumulative distribution for a variable.

        Parameters
        ----------
        var_name : str
            Variable name to analyze.
        x_axis_title : str, optional
            Label for the x-axis. If omitted, ``var_name`` is used.
        plot_title : str, optional
            Plot title. If omitted, a title is generated from ``x_axis_title``.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        ndarray
            Array containing the empirical CDF quantiles followed by the
            corresponding probabilities.

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
        var_name : str
            Variable name to analyze.
        method : {"empirical", "smooth"}, optional
            CDF approximation method. ``"empirical"`` uses SciPy's empirical CDF
            evaluator, and ``"smooth"`` linearly interpolates between empirical
            quantiles.

        Returns
        -------
        callable
            Function that evaluates the selected CDF approximation.

        Raises
        ------
        ValueError
            If ``method`` is not ``"empirical"`` or ``"smooth"``.

        """
        vals = np.array(self.tradestudy.get_variable(var_name))
        ecdf = stats.ecdf(vals)
        quantiles = ecdf.cdf.quantiles
        probabilities = ecdf.cdf.probabilities
        if method == "empirical":

            def cdf(x):
                """Evaluate the empirical CDF.

                Parameters
                ----------
                x : float or array-like
                    Value or values at which to evaluate the CDF.

                Returns
                -------
                float or ndarray
                    Empirical CDF value or values.

                """
                return ecdf.cdf.evaluate(x)

        elif method == "smooth":

            def cdf(x):
                """Evaluate the interpolated CDF.

                Parameters
                ----------
                x : float or array-like
                    Value or values at which to evaluate the CDF.

                Returns
                -------
                float or ndarray
                    Interpolated CDF value or values.

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
        var_name : str
            Variable name to analyze.
        value : float or array-like
            Value at which to evaluate a distribution/CDF.
        method : {"empirical", "smooth"}, optional
            CDF approximation method.

        Returns
        -------
        float or ndarray
            CDF value or values evaluated at ``value``.

        Raises
        ------
        ValueError
            If ``method`` is not ``"empirical"`` or ``"smooth"``.

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
        var_name : str
            Variable name to analyze.
        quantile : float
            Requested quantile level in the interval [0, 1].
        method : str, optional
            NumPy quantile method. Supported values are ``"inverted_cdf"``,
            ``"averaged_inverted_cdf"``, ``"closest_observation"``,
            ``"interpolated_inverted_cdf"``, ``"hazen"``, ``"weibull"``,
            ``"linear"``, ``"median_unbiased"``, and ``"normal_unbiased"``.

        Returns
        -------
        float
            Quantile value for the stored variable values.

        Raises
        ------
        ValueError
            If ``quantile`` is outside [0, 1] or ``method`` is unsupported.

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
        var_name1 : str
            First variable name for correlation calculations.
        var_name2 : str
            Second variable name for correlation calculations.

        Returns
        -------
        correlation : float
            Spearman rank correlation coefficient.
        p_value : float
            Two-sided p-value for a hypothesis test whose null hypothesis is
            that the variables are uncorrelated.

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
        var_name1 : str
            First variable name for correlation calculations.
        var_name2 : str
            Second variable name for correlation calculations.

        Returns
        -------
        correlation : float
            Pearson correlation coefficient.
        p_value : float
            Two-sided p-value for a hypothesis test whose null hypothesis is
            that the variables are uncorrelated.

        """
        var1_vals = np.array(self.tradestudy.get_variable(var_name1))
        var2_vals = np.array(self.tradestudy.get_variable(var_name2))
        significance_res, p_val = stats.pearsonr(var1_vals, var2_vals)
        # SHOULD HANDLE >3 VARS REQUIRES MATRIX
        return significance_res, p_val

class RegressionPCEPost(BasicAleatoryPost):
    """Post-process regression PCE trade-study results.

    This class builds polynomial-chaos surrogate models from completed
    :class:`uam4x.UQ.RegressionPCE` trade-study results.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize a regression PCE post-processing helper.

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
        output_name : str
            Output variable name used to fit the surrogate.

        Returns
        -------
        RegressionSurrogate
            Surrogate model fitted to ``output_name``.

        Notes
        -----
        Log-normal input variables are transformed to their underlying normal
        variables before fitting the OpenTURNS functional chaos model. When
        the workflow was configured with a nonzero holdout fraction, the
        fitted surrogate stores the held-out inputs and outputs for later
        validation metrics. For Monte Carlo training samples this acts like a
        standard random train/test split. For structured space-filling designs
        such as Latin hypercube or Sobol sampling, the resulting holdout
        metrics can be optimistic because both subsets come from the same
        original design rather than independent validation samples.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        num_samples = len(self.tradestudy._results)
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

        outputs=np.array(self.tradestudy.get_variable(output_name))[:, None]
        X_test=None
        y_test=None
        # Handle hold out case
        if ts.hold_out_fraction > 0.0:
            X_train,X_test,y_train,y_test = train_test_split(abscissas,
                                                        outputs,
                                                        test_size=ts.hold_out_fraction,
                                                        random_state=ts.hold_out_seed)
            abscissas = ot.Sample(X_train)
            outputs = ot.Sample(y_train)
        else:
            abscissas = ot.Sample(abscissas)
            outputs = ot.Sample(outputs)

        # Joint distribution based on independent input aleatory dists
        joint = ot.JointDistribution(joint_list)

        # Enumeration function
        enum = ot.LinearEnumerateFunction(len(aleatory_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis = ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order = ts.poly_order
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

        return RegressionSurrogate(aleatory_var_to_dist, res,self.post_storage,X_test,y_test)


class GaussQuadtraturePCEPost(BasicAleatoryPost):
    """Post-process Gauss quadrature PCE trade-study results.

    This class builds polynomial-chaos surrogate models from completed
    :class:`uam4x.UQ.GaussQuadraturePCE` trade-study results using quadrature
    weights stored in each state.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize a Gauss quadrature PCE post-processing helper.

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
        output_name : str
            Output variable name used to fit the surrogate.

        Returns
        -------
        AleatorySurrogate
            Surrogate model fitted to ``output_name``.

        Notes
        -----
        The fitted OpenTURNS functional chaos model uses the quadrature
        ``weight`` stored in each trade-study state. Log-normal input variables
        are transformed to their underlying normal variables before fitting.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        aleatory_var_to_dist = ts.aleatory_var_to_dist
        num_aleatory = len(aleatory_var_to_dist)
        num_samples = (ts.poly_order + 1) ** num_aleatory
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
        poly_order = ts.poly_order
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

        return AleatorySurrogate(aleatory_var_to_dist, res,self.post_storage)


class AleatorySurrogate:
    """Represent an aleatory polynomial-chaos surrogate.

    The surrogate wraps an OpenTURNS functional-chaos result and provides
    prediction, analytical moments, Sobol indices, symbolic metamodels, and
    sampling-based plots.

    Parameters
    ----------
    aleatory_var_to_dist : dict
        Mapping from aleatory variable names to distribution objects.
    result : openturns.FunctionalChaosResult
        OpenTURNS functional-chaos result object.
    post_storage : dict
        Post-processing storage dictionary associated with the trade study.

    """

    def __init__(self, aleatory_var_to_dist, result,post_storage):
        """Initialize an aleatory surrogate.

        Parameters
        ----------
        aleatory_var_to_dist : dict
            Mapping from aleatory variable names to distribution objects.
        result : openturns.FunctionalChaosResult
            OpenTURNS functional-chaos result object.
        post_storage : dict
            Post-processing storage dictionary associated with the trade study.

        """
        self.aleatory_var_to_dist = aleatory_var_to_dist
        self._ot_res = result
        self.new_sample_points = None
        self.new_sample_values = None
        self._Chaos_SI=ot.FunctionalChaosSobolIndices(self._ot_res)
        self.post_storage=post_storage


    def get_symbolic_metamodel(self, model_type='physical'):
        """Return a symbolic expression for the surrogate metamodel.

        Parameters
        ----------
        model_type : {"physical", "probabilistic"}, optional
            Coordinate system for the symbolic expression. ``"physical"``
            expresses the model in physical input variables. ``"probabilistic"``
            returns the OpenTURNS probability-space expression.

        Returns
        -------
        model : sympy expression
            Symbolic surrogate expression.
        aleatory_to_sympy_map : dict
            Mapping from aleatory variable names to SymPy symbols.

        Raises
        ------
        ValueError
            If ``model_type`` is not ``"physical"`` or ``"probabilistic"``.

        """
        if model_type == 'physical':
            return self._get_physical_symbolic_metamodel()
        elif model_type in ('probabilistic', 'probabalistic'):
            return self._get_probability_symbolic_metamodel()
        else:
            raise ValueError(
                "model_type must be either 'physical' or 'probabilistic'."
            )

    def _get_probability_symbolic_metamodel(self):
        """Build the symbolic metamodel in probability space.

        Returns
        -------
        model : sympy expression
            Symbolic expression in OpenTURNS basis variables.
        aleatory_to_sympy_map : dict
            Mapping from aleatory variable names to SymPy symbols.

        """
        result=self._ot_res
        coeffs=np.array(result.getCoefficients()).reshape(-1)
        indices=result.getIndices()
        basis=result.getOrthogonalBasis()

        aleatory_to_sympy_map={}
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            aleatory_to_sympy_map[name]=sp.symbols(f'x{i}')

        model=0
        for coeff,ind in zip(coeffs,indices):
            term=str(basis.build(ind))
            term=sp.sympify(term)
            term=coeff*term
            model+=term
        return model, aleatory_to_sympy_map

    def _get_physical_symbolic_metamodel(self):
        """Build the symbolic metamodel in physical input variables.

        Returns
        -------
        model : sympy expression
            Symbolic expression in physical aleatory variables.
        aleatory_to_sympy_map : dict
            Mapping from aleatory variable names to SymPy symbols.

        Raises
        ------
        ValueError
            If an input distribution type cannot be transformed to the
            OpenTURNS basis variable.

        """
        result=self._ot_res
        coeffs=np.array(result.getCoefficients()).reshape(-1)
        indices=result.getIndices()
        basis=result.getOrthogonalBasis()

        aleatory_to_sympy_map={}
        physical_to_standard_map={}
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            aleatory_to_sympy_map[name]=sp.symbols(f'x{i}')
            physical_to_standard_map[sp.symbols(f'x{i}')]=sp.symbols(f'z{i}')
        # OpenTurns builds the orthogonal basis in its stochastic input space.
        # Re-express those basis variables in terms of the physical variables so
        # symbolic substitution matches `predict(...)`.
        #That is the pce is in terms of the z variables (standard space)
        #need to represent z in terms of x
        standard_to_physical_subs = {}
        for i, (name, distribution) in enumerate(self.aleatory_var_to_dist.items()):
            physical_symbol = aleatory_to_sympy_map[name]
            standard_symbol = sp.symbols(f"z{i}")

            if isinstance(distribution, UniformDistribution):

                #mapping from X~Unif(a,b)-->Z~Unif(-1,1)
                midpoint=(distribution.low+distribution.high)/2
                half_range=(distribution.high-distribution.low)/2
                transformed_symbol=(physical_symbol-midpoint)/half_range
            elif isinstance(distribution, NormalDistribution):
                #mapping from X~Norm(mu,sigma)-->Z~Norm(0,1)
                transformed_symbol=(physical_symbol-distribution.mean)/distribution.std
            elif isinstance(distribution, LogNormalDistribution):
                #mapping from X~Lognormal(mu,sigma)-->Z~Norm(0,1)
                #Map X~Lognormal(mu,sigma)-->Y~Norm(mu,sigma)
                transformed_symbol=(sp.ln(physical_symbol-distribution.gamma)-distribution.mu_log)/distribution.sigma_log
            else:
                raise ValueError(f'Invalid distribution')

            standard_to_physical_subs[standard_symbol] = transformed_symbol

        model=0
        for coeff,ind in zip(coeffs,indices):
            term=str(basis.build(ind))
            term=sp.sympify(term)
            term=term.subs(physical_to_standard_map)
            term=term.subs(standard_to_physical_subs)
            term=coeff*term
            model+=term
        return model, aleatory_to_sympy_map


    def get_first_order_sobol(self,var_name):
        """Return the first-order Sobol index for one variable.

        Parameters
        ----------
        var_name : str
            Aleatory variable name.

        Returns
        -------
        float
            First-order Sobol sensitivity index.

        """
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            if var_name==name:
                val=self._Chaos_SI.getSobolIndex(i)
                return val


    def get_total_order_sobol(self,var_name):
        """Return the total-order Sobol index for one variable.

        Parameters
        ----------
        var_name : str
            Aleatory variable name.

        Returns
        -------
        float
            Total-order Sobol sensitivity index.

        """
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            if var_name==name:
                val=self._Chaos_SI.getSobolTotalIndex(i)
                return val

    def get_interaction_sobol(self,var_names:list):
        """Return the grouped Sobol index for a set of variables.

        Parameters
        ----------
        var_names : list of str
            Aleatory variable names included in the interaction group.

        Returns
        -------
        float
            Grouped Sobol sensitivity index.

        """
        inds=[]
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)
        vals=self._Chaos_SI.getSobolGroupedIndex(inds)
        names=tuple(var_names)
        return vals

    def get_total_interaction_sobol(self,var_names:list):
        """Return the grouped total-order Sobol index for a set of variables.

        Parameters
        ----------
        var_names : list of str
            Aleatory variable names included in the interaction group.

        Returns
        -------
        float
            Grouped total-order Sobol sensitivity index.

        """
        inds=[]
        for i,name in enumerate(self.aleatory_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        vals=self._Chaos_SI.getSobolGroupedTotalIndex(inds)
        names=tuple(var_names)
        return vals

    def plot_total_order_sobol(self,plot_title="",save_dir=None):
        """Plot total-order Sobol indices for all aleatory variables.

        Parameters
        ----------
        plot_title : str, optional
            Plot title. If omitted, a default title is used.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        """
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
        # probabilistic based inputs
        """Evaluate the surrogate at one dictionary of input variable values.

        Parameters
        ----------
        name_to_value_dict : dict
            Mapping from aleatory variable names to physical input values.

        Returns
        -------
        openturns.Point
            Surrogate prediction at the provided input point.

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
        """Analytical mean of the surrogate output.

        Returns
        -------
        float
            Mean computed from the OpenTURNS functional-chaos random vector.

        """
        mean=ot.FunctionalChaosRandomVector(self._ot_res).getMean()[0]
        return mean


    @property
    def analytical_variance(self):
        """Analytical variance of the surrogate output.

        Returns
        -------
        float
            Variance computed from the polynomial-chaos coefficients.

        Raises
        ------
        ValueError
            If the coefficient-based variance does not match the OpenTURNS
            covariance result.

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
        """Analytical standard deviation of the surrogate output.

        Returns
        -------
        float
            Square root of :attr:`analytical_variance`.

        """
        std=np.sqrt(self.analytical_variance)
        return std

    def sample(self, num_samples, method="MC"):
        """Generate new surrogate samples using a selected sampling strategy.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        method : {"MC", "LHS", "Sobol", "Halton"}, optional
            Sampling method used to generate surrogate input points.

        Returns
        -------
        sample_points : ndarray
            Sampled input points in physical variable space.
        sample_values : ndarray
            Surrogate output values evaluated at ``sample_points``.

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
        xlabel : str, optional
            Label for the x-axis. If omitted, ``"Output value"`` is used.
        plot_title : str, optional
            Plot title. If omitted, a default title is used.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        ndarray
            Array containing the empirical CDF quantiles followed by the
            corresponding probabilities.

        Raises
        ------
        ValueError
            If surrogate samples have not been generated with :meth:`sample`.

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
        bins : int, optional
            Number of bins used for histogram-based plots.
        xlabel : str, optional
            Label for the x-axis. If omitted, ``"Output value"`` is used.
        plot_title : str, optional
            Plot title. If omitted, a default title is used.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object for the generated PDF plot.

        Raises
        ------
        ValueError
            If surrogate samples have not been generated with :meth:`sample`.

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

class RegressionSurrogate(AleatorySurrogate):
    """Represent a regression-fitted aleatory polynomial-chaos surrogate.

    When the parent workflow used a nonzero holdout fraction, the surrogate
    stores the held-out inputs and outputs and exposes validation metrics on
    that test split. Those metrics are most directly interpretable when the
    parent regression workflow used Monte Carlo sampling; they may be
    optimistic for space-filling samplers such as Latin hypercube or Sobol.
    The metric methods also accept external physical-space validation inputs
    and outputs.

    """

    def __init__(self, aleatory_var_to_dist, result,post_storage,X_test=None,y_test=None):
        super().__init__(aleatory_var_to_dist, result,post_storage)
        if X_test is not None:
            self.X_test=X_test
        if y_test is not None:
            self.y_test=y_test

    def get_surrogate_mse(self,X=None,y=None):
        """Return the mean-squared error for the surrogate.

        Parameters
        ----------
        X : array-like or None, optional
            Validation input points in physical variable space. When omitted,
            the stored internal holdout split is used. When provided, ``X``
            must be a 2D array-like object with rows as validation samples
            and columns ordered to match ``self.aleatory_var_to_dist``.
        y : array-like or None, optional
            Validation outputs corresponding to ``X``. Must be provided
            together with ``X``.

        Returns
        -------
        float
            Mean-squared error evaluated on either the stored holdout split or
            the provided external validation data.

        Raises
        ------
        ValueError
            If no holdout split was configured and no external validation data
            were provided, if only one of ``X`` and ``y`` was provided, or if
            ``X`` is not a 2D array-like object, or if ``X`` does not contain
            one column per aleatory variable in surrogate order.

        """
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Both X and y must be provided together.")
        if X is None or y is None:
            if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
                raise ValueError("No hold out was specified for this surrogate, cannot compute MSE.")
            inputs=ot.PointCollection(self.X_test)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(self.y_test, evaluated_vals)
        else:
            #assumes provided order of X matches the provided order of aleatory variables in the surrogate
            X_new=np.copy(X)
            if X_new.ndim != 2:
                raise ValueError("X must be a 2D array-like object with rows as samples and columns as variables.")
            if X_new.shape[1] != len(self.aleatory_var_to_dist):
                raise ValueError("X must have one column per aleatory variable in surrogate order.")
            y_new=np.asarray(y)
            if y_new.shape[0] != X_new.shape[0]:
                raise ValueError("y must contain one validation output per row of X.")
            for i, (name,dist) in enumerate(self.aleatory_var_to_dist.items()):
                if isinstance(dist, LogNormalDistribution):
                    # transform X~Lognormal(mu,sigma)-->Z~Norm(mu,sigma)
                    if np.any(X_new[:, i] - dist.gamma <= 0.0):
                        raise ValueError(
                            f"LogNormal variable '{name}' contains values at or below its "
                            f"shift gamma={dist.gamma}; cannot apply log transform."
                        )
                    z = np.log(X_new[:, i] - dist.gamma)
                    X_new[:, i] = z
            inputs=ot.PointCollection(X_new)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(y_new, evaluated_vals)
        return mse

    def get_surrogate_r2(self,X=None,y=None):
        """Return the coefficient of determination for the surrogate.

        Parameters
        ----------
        X : array-like or None, optional
            Validation input points in physical variable space. When omitted,
            the stored internal holdout split is used. When provided, ``X``
            must be a 2D array-like object with rows as validation samples
            and columns ordered to match ``self.aleatory_var_to_dist``.
        y : array-like or None, optional
            Validation outputs corresponding to ``X``. Must be provided
            together with ``X``.

        Returns
        -------
        float
            R-squared evaluated on either the stored holdout split or the
            provided external validation data.

        Raises
        ------
        ValueError
            If no holdout split was configured and no external validation data
            were provided, if only one of ``X`` and ``y`` was provided, or if
            ``X`` is not a 2D array-like object, or if ``X`` does not contain
            one column per aleatory variable in surrogate order.

        """
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Both X and y must be provided together.")
        if X is None or y is None:
            if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
                raise ValueError("No hold out was specified for this surrogate, cannot compute R2.")
            inputs=ot.PointCollection(self.X_test)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            r2=r2_score(self.y_test, evaluated_vals)
        else:
            #assumes provided order of X matches the provided order of aleatory variables in the surrogate
            X_new=np.copy(X)
            if X_new.ndim != 2:
                raise ValueError("X must be a 2D array-like object with rows as samples and columns as variables.")
            if X_new.shape[1] != len(self.aleatory_var_to_dist):
                raise ValueError("X must have one column per aleatory variable in surrogate order.")
            y_new=np.asarray(y)
            if y_new.shape[0] != X_new.shape[0]:
                raise ValueError("y must contain one validation output per row of X.")
            for i, (name,dist) in enumerate(self.aleatory_var_to_dist.items()):
                if isinstance(dist, LogNormalDistribution):
                    # transform X~Lognormal(mu,sigma)-->Z~Norm(mu,sigma)
                    if np.any(X_new[:, i] - dist.gamma <= 0.0):
                        raise ValueError(
                            f"LogNormal variable '{name}' contains values at or below its "
                            f"shift gamma={dist.gamma}; cannot apply log transform."
                        )
                    z = np.log(X_new[:, i] - dist.gamma)
                    X_new[:, i] = z
            inputs=ot.PointCollection(X_new)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            r2=r2_score(y_new, evaluated_vals)
        return r2

    def get_surrogate_rel_mse(self,X=None,y=None):
        """Return MSE normalized by the validation-output standard deviation.

        Parameters
        ----------
        X : array-like or None, optional
            Validation input points in physical variable space. When omitted,
            the stored internal holdout split is used. When provided, ``X``
            must be a 2D array-like object with rows as validation samples
            and columns ordered to match ``self.aleatory_var_to_dist``.
        y : array-like or None, optional
            Validation outputs corresponding to ``X``. Must be provided
            together with ``X``.

        Returns
        -------
        float
            Mean-squared error divided by the standard deviation of the stored
            holdout outputs or the provided external validation outputs.

        Raises
        ------
        ValueError
            If no holdout split was configured and no external validation data
            were provided, if only one of ``X`` and ``y`` was provided, or if
            ``X`` is not a 2D array-like object, if ``X`` does not contain one
            column per aleatory variable in surrogate order, or if the
            validation outputs have zero standard deviation.

        """
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Both X and y must be provided together.")
        if X is None or y is None:
            if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
                raise ValueError("No hold out was specified for this surrogate, cannot compute relative MSE.")
            inputs=ot.PointCollection(self.X_test)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(self.y_test, evaluated_vals)
            std=np.std(self.y_test)
        else:
            #assumes provided order of X matches the provided order of aleatory variables in the surrogate
            X_new=np.copy(X)
            if X_new.ndim != 2:
                raise ValueError("X must be a 2D array-like object with rows as samples and columns as variables.")
            if X_new.shape[1] != len(self.aleatory_var_to_dist):
                raise ValueError("X must have one column per aleatory variable in surrogate order.")
            y_new=np.asarray(y)
            if y_new.shape[0] != X_new.shape[0]:
                raise ValueError("y must contain one validation output per row of X.")
            for i, (name,dist) in enumerate(self.aleatory_var_to_dist.items()):
                if isinstance(dist, LogNormalDistribution):
                    # transform X~Lognormal(mu,sigma)-->Z~Norm(mu,sigma)
                    if np.any(X_new[:, i] - dist.gamma <= 0.0):
                        raise ValueError(
                            f"LogNormal variable '{name}' contains values at or below its "
                            f"shift gamma={dist.gamma}; cannot apply log transform."
                        )
                    z = np.log(X_new[:, i] - dist.gamma)
                    X_new[:, i] = z
            inputs=ot.PointCollection(X_new)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(y_new, evaluated_vals)
            std=np.std(y_new)
        if std == 0:
            raise ValueError("Standard deviation of validation outputs is zero, cannot compute relative MSE.")
        rel_mse=mse/std
        return rel_mse

class SobolSensitivityPost(BasicAleatoryPost):
    """Post-process Sobol sensitivity trade-study results.

    This class computes Sobol sensitivity indices from the structured sample
    design generated by :class:`uam4x.UQ.SobolSensitivity`.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    _METHOD_TO_CLASS = {
        "Saltelli": ot.SaltelliSensitivityAlgorithm,
        "Jansen": ot.JansenSensitivityAlgorithm,
        "MauntzKucherenko": ot.MauntzKucherenkoSensitivityAlgorithm,
        "Martinez": ot.MartinezSensitivityAlgorithm,
    }

    def __init__(self, tradestudy):
        """Initialize a Sobol sensitivity post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        super().__init__(tradestudy)

    def generate_Sobol_indices(self, output_name, algo="Jansen"):
        # Output values
        """Compute Sobol indices for one model output.

        Parameters
        ----------
        output_name : str
            Output variable name used for Sobol calculations.
        algo : {"Jansen", "Saltelli", "Martinez", "MauntzKucherenko"}, optional
            OpenTURNS Sobol sensitivity algorithm to use.

        Returns
        -------
        results : dict
            Dictionary with ``"first_order"``, ``"second_order"``, and
            ``"total_order"`` entries. First- and total-order entries are keyed
            by variable name; second-order entries are keyed by variable-name
            tuples.
        otSobol : object
            OpenTURNS Sobol sensitivity algorithm object.

        """
        outs = np.array(self.tradestudy.get_variable(output_name))[:, None]

        # Sobol workflow configuration for this trade study.
        uq_distributor = self.tradestudy.study.control.trade_studies[
            self.tradestudy.trade_name
        ]
        aleatory_dict = uq_distributor.aleatory_var_to_dist

        # Number of model evaluations used by the Sobol design.
        num_samples = len(self.tradestudy._results)
        # Number of aleatory vars
        # Calculates the number of samples needed to preallocate array shape
        second_order = uq_distributor.computeSecondOrder
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
        output_name : str
            Output variable name used for Sobol calculations.
        algo : {"Jansen", "Saltelli", "Martinez", "MauntzKucherenko"}, optional
            OpenTURNS Sobol sensitivity algorithm to use.

        Returns
        -------
        dict
            First-order Sobol indices keyed by aleatory variable name.

        """
        results, _ = self.generate_Sobol_indices(output_name, algo)
        return results["first_order"]

    def get_second_order_indices(self, output_name, algo="Jansen"):
        """Return second-order Sobol indices for one output.

        Parameters
        ----------
        output_name : str
            Output variable name used for Sobol calculations.
        algo : {"Jansen", "Saltelli", "Martinez", "MauntzKucherenko"}, optional
            OpenTURNS Sobol sensitivity algorithm to use.

        Returns
        -------
        dict
            Second-order Sobol indices keyed by aleatory variable-name tuples.

        Raises
        ------
        ValueError
            If the underlying Sobol workflow was not configured to compute
            second-order indices.

        """
        uq_distributor = self.tradestudy.study.control.trade_studies[
            self.tradestudy.trade_name
        ]
        if not uq_distributor.computeSecondOrder:
            raise ValueError(f"This run did not support second order Sobol")

        results, _ = self.generate_Sobol_indices(output_name, algo)

        return results["second_order"]

    def get_total_order_indices(self, output_name, algo="Jansen"):
        """Return total-order Sobol indices for one output.

        Parameters
        ----------
        output_name : str
            Output variable name used for Sobol calculations.
        algo : {"Jansen", "Saltelli", "Martinez", "MauntzKucherenko"}, optional
            OpenTURNS Sobol sensitivity algorithm to use.

        Returns
        -------
        dict
            Total-order Sobol indices keyed by aleatory variable name.

        """
        results, _ = self.generate_Sobol_indices(output_name, algo)
        return results["total_order"]

    def plot_total_order_indices(self,output_name,plot_title="",algo="Jansen",save_dir=None):
        """Plot total-order Sobol indices for one output.

        Parameters
        ----------
        output_name : str
            Output variable name used for Sobol calculations.
        plot_title : str, optional
            Plot title. If omitted, a default title is generated from
            ``output_name``.
        algo : {"Jansen", "Saltelli", "Martinez", "MauntzKucherenko"}, optional
            OpenTURNS Sobol sensitivity algorithm to use.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        """
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
    """Post-process epistemic trade-study results.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self,tradestudy:Tradestudy):
        """Initialize an epistemic post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        self.tradestudy=tradestudy
        trade_name=self.tradestudy.trade_name
        self.post_storage=tradestudy.study._post_data[trade_name]

    def compute_bounds(self,var_name):
        """Compute minimum and maximum observed values for a variable.

        Parameters
        ----------
        var_name : str
            Variable name to analyze.

        Returns
        -------
        dict
            Dictionary with ``"minimum"`` and ``"maximum"`` entries. Each entry
            contains the observed value and the corresponding state dictionary.

        """
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
    """Post-process mixed aleatory-epistemic trade-study results.

    This class provides p-box, p-level bound, and horsetail utilities for mixed
    analyses where each epistemic state contains a group of aleatory samples.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize a mixed post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        self.tradestudy = tradestudy
        trade_name=self.tradestudy.trade_name
        self.post_storage=tradestudy.study._post_data[trade_name]

    def _generate_output_var_data(self,var_name):
        """Group output values by epistemic state and compute empirical CDFs.

        Parameters
        ----------
        var_name : str
            Output variable name to analyze.

        Returns
        -------
        state_to_cdf : dict
            Mapping from epistemic state tuples to SciPy empirical CDF objects.
        quantiles_data : ndarray
            Matrix of empirical CDF quantiles. Columns correspond to epistemic
            states.
        probabilities_data : ndarray
            Matrix of empirical CDF probabilities. Columns correspond to
            epistemic states.

        Raises
        ------
        ValueError
            If epistemic states do not all contain the same number of aleatory
            samples.

        """
        trade_name=self.tradestudy.trade_name
        trade_base=self.tradestudy.study.control.trade_studies[trade_name]
        out_vals= np.array(self.tradestudy.get_variable(var_name))
        #special case for IVP problems with ad hoc state generation
        epistemic_vars=list(trade_base.epistemic_var_to_interval.keys())
        epistemic_state_to_inds={}
        for i,result in enumerate(self.tradestudy._results):
            epistemic_state=tuple(result.state[epistemic_var] for epistemic_var in epistemic_vars)
            if epistemic_state not in epistemic_state_to_inds:
                epistemic_state_to_inds[epistemic_state]=[i]
            else:
                epistemic_state_to_inds[epistemic_state].append(i)

        num_cols=len(epistemic_state_to_inds.keys())
        group_sizes = [len(inds) for inds in epistemic_state_to_inds.values()]
        if len(set(group_sizes)) != 1:
            raise ValueError("Inconsistent number of aleatory samples per epistemic state.")
        num_rows = group_sizes[0]
        quantiles_data=np.zeros((num_rows,num_cols))
        probabilities_data=np.zeros((num_rows,num_cols))
        state_to_cdf={}
        for i,(epistemic_state,inds) in enumerate(epistemic_state_to_inds.items()):
            vals=out_vals[inds]
            ecdf=stats.ecdf(vals)
            state_to_cdf[epistemic_state]=ecdf
            quantiles_data[:,i]=ecdf.cdf.quantiles
            probabilities_data[:,i]=ecdf.cdf.probabilities

        return state_to_cdf,quantiles_data,probabilities_data


    def plot_pbox(self,var_name,x_axis_title=None,plot_title="",save_dir=None):
        """Plot a probability box for one output variable.

        Parameters
        ----------
        var_name : str
            Output variable name to analyze.
        x_axis_title : str, optional
            Label for the x-axis. If omitted, ``var_name`` is used.
        plot_title : str, optional
            Plot title. If omitted, a default title is generated.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        """

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
        """Return p-box bounds at one probability level.

        Parameters
        ----------
        var_name : str
            Output variable name to analyze.
        p_level : float
            Probability level in the interval [0, 1].

        Returns
        -------
        min_quantile : float
            Lower p-box quantile bound at ``p_level``.
        max_quantile : float
            Upper p-box quantile bound at ``p_level``.

        Raises
        ------
        ValueError
            If ``p_level`` is outside [0, 1].

        """
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
        """Plot horsetail empirical CDF curves for one output variable.

        Parameters
        ----------
        var_name : str
            Output variable name to analyze.
        x_axis_title : str, optional
            Label for the x-axis. If omitted, ``var_name`` is used.
        plot_title : str, optional
            Plot title. If omitted, a default title is generated.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        """
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


class MixedRegressionPCEPost(BasicMixedPost):
    """Post-process mixed regression PCE trade-study results.

    This class builds polynomial-chaos surrogate models from completed
    :class:`uam4x.UQ.MixedRegressionPCE` trade-study results.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    Notes
    -----
    Epistemic intervals are represented as uniform distributions when fitting
    the mixed surrogate.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize a mixed regression PCE post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        super().__init__(tradestudy)

    def _get_mixed_var_to_dist(self):
        """Build a distribution mapping for all mixed variables.

        Returns
        -------
        dict
            Mapping from epistemic and aleatory variable names to distribution
            objects. Epistemic intervals are converted to uniform
            distributions.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        mixed_var_to_dist = {}
        for name, interval in ts.epistemic_var_to_interval.items():
            mixed_var_to_dist[name] = UniformDistribution(interval[0], interval[1])
        for name, distribution in ts.aleatory_var_to_dist.items():
            mixed_var_to_dist[name] = distribution
        return mixed_var_to_dist

    def generate_surrogate(self, output_name):
        """Build and return a mixed polynomial-chaos surrogate.

        Parameters
        ----------
        output_name : str
            Output variable name used to fit the surrogate.

        Returns
        -------
        MixedSurrogate
            Surrogate model fitted to ``output_name``.

        Notes
        -----
        Log-normal input variables are transformed to their underlying normal
        variables before fitting the OpenTURNS functional chaos model. When
        the workflow was configured with a nonzero holdout fraction, the
        fitted surrogate stores the held-out inputs and outputs for later
        validation metrics. For Monte Carlo training samples this acts like a
        standard random train/test split. For structured space-filling designs
        such as Latin hypercube or Sobol sampling, the resulting holdout
        metrics can be optimistic because both subsets come from the same
        original design rather than independent validation samples.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        num_samples = len(self.tradestudy._results)
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

        outputs=np.array(self.tradestudy.get_variable(output_name))[:, None]
        X_test=None
        y_test=None
        # Handle hold out case
        if ts.hold_out_fraction > 0.0:
            X_train,X_test,y_train,y_test = train_test_split(abscissas,
                                                        outputs,
                                                        test_size=ts.hold_out_fraction,
                                                        random_state=ts.hold_out_seed)

            abscissas = ot.Sample(X_train)
            outputs = ot.Sample(y_train)
        else:
            abscissas = ot.Sample(abscissas)
            outputs = ot.Sample(outputs)

        # Joint distribution based on independent input aleatory dists
        joint = ot.JointDistribution(joint_list)

        # Enumeration function
        enum = ot.LinearEnumerateFunction(len(all_var_to_dist))

        # Basis built from distributions (Normal-->Hermite, Uniform-->Legendre)
        product_basis = ot.OrthogonalProductPolynomialFactory(poly_factories)

        # size of the basis is equal to C(degree of poly+num of aleatory, degree of poly)
        poly_order = ts.poly_order
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

        epistemic_var_to_interval=ts.epistemic_var_to_interval
        aleatory_var_to_dist=ts.aleatory_var_to_dist

        return MixedSurrogate(all_var_to_dist, res,epistemic_var_to_interval,
                              aleatory_var_to_dist,self.post_storage,X_test=X_test,y_test=y_test)

class MixedSurrogate:
    """Represent a mixed aleatory-epistemic polynomial-chaos surrogate.

    The surrogate wraps an OpenTURNS functional-chaos result with both
    epistemic and aleatory input variables. It provides prediction, sampling,
    p-box and horsetail plots, symbolic metamodels, Sobol indices, and
    holdout-based validation metrics when available. Those metrics are most
    directly interpretable when the parent regression workflow used Monte
    Carlo sampling; they may be optimistic for space-filling samplers such as
    Latin hypercube or Sobol. The metric methods also accept external
    physical-space validation inputs and outputs.

    Parameters
    ----------
    all_var_to_dist : dict
        Mapping from all mixed variable names to distribution objects.
    result : openturns.FunctionalChaosResult
        OpenTURNS functional-chaos result object.
    epistemic_var_to_interval : dict
        Mapping from epistemic variable names to interval bounds.
    aleatory_var_to_dist : dict
        Mapping from aleatory variable names to distribution objects.
    post_storage : dict
        Post-processing storage dictionary associated with the trade study.

    """

    _META_STATE_NAME="__IVP__STATE__META"
    def __init__(self, all_var_to_dist, result,epistemic_var_to_interval,
                 aleatory_var_to_dist,post_storage,X_test=None,y_test=None):
        # NEED TO ADD A VARIABLE NAME TO DISTRIBUTION FOR SPECIAL TRANSFORMS
        # FROM FAKE SPACE BACK TO PHYSICAL INTERNALLY
        """Initialize a mixed surrogate.

        Parameters
        ----------
        all_var_to_dist : dict
            Mapping from all mixed variable names to distribution objects.
        result : openturns.FunctionalChaosResult
            OpenTURNS functional-chaos result object.
        epistemic_var_to_interval : dict
            Mapping from epistemic variable names to interval bounds.
        aleatory_var_to_dist : dict
            Mapping from aleatory variable names to distribution objects.
        post_storage : dict
            Post-processing storage dictionary associated with the trade study.

        """
        self.epistemic_var_to_interval=epistemic_var_to_interval
        self.aleatory_var_to_dist=aleatory_var_to_dist
        self.all_var_to_dist = all_var_to_dist
        self._ot_res = result
        self.new_sample_points = None
        self.new_sample_values = None
        self._Chaos_SI=ot.FunctionalChaosSobolIndices(self._ot_res)
        self.post_storage=post_storage
        if X_test is not None:
            self.X_test=X_test
        if y_test is not None:
            self.y_test=y_test

    def get_surrogate_mse(self,X=None,y=None):
        """Return the mean-squared error for the surrogate.

        Parameters
        ----------
        X : array-like or None, optional
            Validation input points in physical mixed-variable space. When
            omitted, the stored internal holdout split is used. When provided,
            ``X`` must be a 2D array-like object with rows as validation
            samples and columns ordered to match ``self.all_var_to_dist``.
        y : array-like or None, optional
            Validation outputs corresponding to ``X``. Must be provided
            together with ``X``.

        Returns
        -------
        float
            Mean-squared error evaluated on either the stored holdout split or
            the provided external validation data.

        Raises
        ------
        ValueError
            If no holdout split was configured and no external validation data
            were provided, if only one of ``X`` and ``y`` was provided, or if
            ``X`` is not a 2D array-like object, or if ``X`` does not contain
            one column per mixed variable in surrogate order.

        """
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Both X and y must be provided together.")
        if X is None or y is None:
            if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
                raise ValueError("No hold out was specified for this surrogate, cannot compute MSE.")
            inputs=ot.PointCollection(self.X_test)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(self.y_test, evaluated_vals)
        else:
            #assumes provided order of X matches the provided order of mixed variables in the surrogate
            X_new=np.copy(X)
            if X_new.ndim != 2:
                raise ValueError("X must be a 2D array-like object with rows as samples and columns as variables.")
            if X_new.shape[1] != len(self.all_var_to_dist):
                raise ValueError("X must have one column per mixed variable in surrogate order.")
            y_new=np.asarray(y)
            if y_new.shape[0] != X_new.shape[0]:
                raise ValueError("y must contain one validation output per row of X.")
            for i, (name,dist) in enumerate(self.all_var_to_dist.items()):
                if isinstance(dist, LogNormalDistribution):
                    # transform X~Lognormal(mu,sigma)-->Z~Norm(mu,sigma)
                    if np.any(X_new[:, i] - dist.gamma <= 0.0):
                        raise ValueError(
                            f"LogNormal variable '{name}' contains values at or below its "
                            f"shift gamma={dist.gamma}; cannot apply log transform."
                        )
                    z = np.log(X_new[:, i] - dist.gamma)
                    X_new[:, i] = z
            inputs=ot.PointCollection(X_new)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(y_new, evaluated_vals)
        return mse

    def get_surrogate_r2(self,X=None,y=None):
        """Return the coefficient of determination for the surrogate.

        Parameters
        ----------
        X : array-like or None, optional
            Validation input points in physical mixed-variable space. When
            omitted, the stored internal holdout split is used. When provided,
            ``X`` must be a 2D array-like object with rows as validation
            samples and columns ordered to match ``self.all_var_to_dist``.
        y : array-like or None, optional
            Validation outputs corresponding to ``X``. Must be provided
            together with ``X``.

        Returns
        -------
        float
            R-squared evaluated on either the stored holdout split or the
            provided external validation data.

        Raises
        ------
        ValueError
            If no holdout split was configured and no external validation data
            were provided, if only one of ``X`` and ``y`` was provided, or if
            ``X`` is not a 2D array-like object, or if ``X`` does not contain
            one column per mixed variable in surrogate order.

        """
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Both X and y must be provided together.")
        if X is None or y is None:
            if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
                raise ValueError("No hold out was specified for this surrogate, cannot compute R2.")
            inputs=ot.PointCollection(self.X_test)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            r2=r2_score(self.y_test, evaluated_vals)
        else:
            #assumes provided order of X matches the provided order of mixed variables in the surrogate
            X_new=np.copy(X)
            if X_new.ndim != 2:
                raise ValueError("X must be a 2D array-like object with rows as samples and columns as variables.")
            if X_new.shape[1] != len(self.all_var_to_dist):
                raise ValueError("X must have one column per mixed variable in surrogate order.")
            y_new=np.asarray(y)
            if y_new.shape[0] != X_new.shape[0]:
                raise ValueError("y must contain one validation output per row of X.")
            for i, (name,dist) in enumerate(self.all_var_to_dist.items()):
                if isinstance(dist, LogNormalDistribution):
                    # transform X~Lognormal(mu,sigma)-->Z~Norm(mu,sigma)
                    if np.any(X_new[:, i] - dist.gamma <= 0.0):
                        raise ValueError(
                            f"LogNormal variable '{name}' contains values at or below its "
                            f"shift gamma={dist.gamma}; cannot apply log transform."
                        )
                    z = np.log(X_new[:, i] - dist.gamma)
                    X_new[:, i] = z
            inputs=ot.PointCollection(X_new)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            r2=r2_score(y_new, evaluated_vals)
        return r2

    def get_surrogate_rel_mse(self,X=None,y=None):
        """Return MSE normalized by the validation-output standard deviation.

        Parameters
        ----------
        X : array-like or None, optional
            Validation input points in physical mixed-variable space. When
            omitted, the stored internal holdout split is used. When provided,
            ``X`` must be a 2D array-like object with rows as validation
            samples and columns ordered to match ``self.all_var_to_dist``.
        y : array-like or None, optional
            Validation outputs corresponding to ``X``. Must be provided
            together with ``X``.

        Returns
        -------
        float
            Mean-squared error divided by the standard deviation of the stored
            holdout outputs or the provided external validation outputs.

        Raises
        ------
        ValueError
            If no holdout split was configured and no external validation data
            were provided, if only one of ``X`` and ``y`` was provided, or if
            ``X`` is not a 2D array-like object, if ``X`` does not contain one
            column per mixed variable in surrogate order, or if the validation
            outputs have zero standard deviation.

        """
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Both X and y must be provided together.")
        if X is None or y is None:
            if not hasattr(self, "X_test") or not hasattr(self, "y_test"):
                raise ValueError("No hold out was specified for this surrogate, cannot compute relative MSE.")
            inputs=ot.PointCollection(self.X_test)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(self.y_test, evaluated_vals)
            std=np.std(self.y_test)
        else:
            #assumes provided order of X matches the provided order of mixed variables in the surrogate
            X_new=np.copy(X)
            if X_new.ndim != 2:
                raise ValueError("X must be a 2D array-like object with rows as samples and columns as variables.")
            if X_new.shape[1] != len(self.all_var_to_dist):
                raise ValueError("X must have one column per mixed variable in surrogate order.")
            y_new=np.asarray(y)
            if y_new.shape[0] != X_new.shape[0]:
                raise ValueError("y must contain one validation output per row of X.")
            for i, (name,dist) in enumerate(self.all_var_to_dist.items()):
                if isinstance(dist, LogNormalDistribution):
                    # transform X~Lognormal(mu,sigma)-->Z~Norm(mu,sigma)
                    if np.any(X_new[:, i] - dist.gamma <= 0.0):
                        raise ValueError(
                            f"LogNormal variable '{name}' contains values at or below its "
                            f"shift gamma={dist.gamma}; cannot apply log transform."
                        )
                    z = np.log(X_new[:, i] - dist.gamma)
                    X_new[:, i] = z
            inputs=ot.PointCollection(X_new)
            evaluated_vals=np.array(self._ot_res.getMetaModel()(inputs))
            mse=mean_squared_error(y_new, evaluated_vals)
            std=np.std(y_new)
        if std == 0:
            raise ValueError("Standard deviation of validation outputs is zero, cannot compute relative MSE.")
        rel_mse=mse/std
        return rel_mse

    def get_symbolic_metamodel(self,model_type='physical'):
        """Return a symbolic expression for the surrogate metamodel.

        Parameters
        ----------
        model_type : {"physical", "probabilistic"}, optional
            Coordinate system for the symbolic expression. ``"physical"``
            expresses the model in physical input variables. ``"probabilistic"``
            returns the OpenTURNS probability-space expression.

        Returns
        -------
        model : sympy expression
            Symbolic surrogate expression.
        all_var_to_sympy_map : dict
            Mapping from mixed variable names to SymPy symbols.

        Raises
        ------
        ValueError
            If ``model_type`` is not ``"physical"`` or ``"probabilistic"``.

        """
        if model_type=='physical':
            return self._get_physical_symbolic_metamodel()
        elif model_type in ('probabilistic', 'probabalistic'):
            return self._get_probability_symbolic_metamodel()
        else:
            raise ValueError(
                "model_type must be either 'physical' or 'probabilistic'."
            )

    def _get_probability_symbolic_metamodel(self):
        """Build the symbolic metamodel in probability space.

        Returns
        -------
        model : sympy expression
            Symbolic expression in OpenTURNS basis variables.
        all_var_to_sympy_map : dict
            Mapping from mixed variable names to SymPy symbols.

        """
        result=self._ot_res
        result=self._ot_res
        coeffs=np.array(result.getCoefficients()).reshape(-1)
        indices=result.getIndices()
        basis=result.getOrthogonalBasis()

        #both epistemic and aleatory
        all_var_to_sympy_map={}
        for i,name in enumerate(self.all_var_to_dist.keys()):
            all_var_to_sympy_map[name]=sp.symbols(f'x{i}')

        model=0
        for coeff,ind in zip(coeffs,indices):
            term=str(basis.build(ind))
            term=sp.sympify(term)
            term=coeff*term
            model+=term

        return model, all_var_to_sympy_map

    def _get_physical_symbolic_metamodel(self):
        """Build the symbolic metamodel in physical input variables.

        Returns
        -------
        model : sympy expression
            Symbolic expression in physical mixed variables.
        all_var_to_sympy_map : dict
            Mapping from mixed variable names to SymPy symbols.

        Raises
        ------
        ValueError
            If an input distribution type cannot be transformed to the
            OpenTURNS basis variable.

        """
        result=self._ot_res
        coeffs=np.array(result.getCoefficients()).reshape(-1)
        indices=result.getIndices()
        basis=result.getOrthogonalBasis()

        #both epistemic and aleatory
        all_var_to_sympy_map={}
        physical_to_standard_map={}
        for i,name in enumerate(self.all_var_to_dist.keys()):
            physical_to_standard_map[sp.symbols(f'x{i}')]=sp.symbols(f'z{i}')
            all_var_to_sympy_map[name]=sp.symbols(f'x{i}')

        standard_to_physical_subs = {}
        for i, (name, distribution) in enumerate(self.all_var_to_dist.items()):
            physical_symbol = all_var_to_sympy_map[name]
            standard_symbol = sp.symbols(f"z{i}")

            if isinstance(distribution, UniformDistribution):

                #mapping from X~Unif(a,b)-->Z~Unif(-1,1)
                midpoint=(distribution.low+distribution.high)/2
                half_range=(distribution.high-distribution.low)/2
                transformed_symbol=(physical_symbol-midpoint)/half_range
            elif isinstance(distribution, NormalDistribution):
                #mapping from X~Norm(mu,sigma)-->Z~Norm(0,1)
                transformed_symbol=(physical_symbol-distribution.mean)/distribution.std
            elif isinstance(distribution, LogNormalDistribution):
                #mapping from X~Lognormal(mu,sigma)-->Z~Norm(0,1)
                #Map X~Lognormal(mu,sigma)-->Y~Norm(mu,sigma)
                transformed_symbol=(sp.ln(physical_symbol-distribution.gamma)-distribution.mu_log)/distribution.sigma_log
            else:
                raise ValueError(f'Invalid distribution')

            standard_to_physical_subs[standard_symbol] = transformed_symbol

        model=0
        for coeff,ind in zip(coeffs,indices):
            term=str(basis.build(ind))
            term=sp.sympify(term)
            term=term.subs(physical_to_standard_map)
            term=term.subs(standard_to_physical_subs)
            term=coeff*term
            model+=term
        return model, all_var_to_sympy_map


    def predict(self, name_to_value_dict):
        # Note, name_to_value_dict should be PHYSICAL inputs, not
        # probabilistic based inputs
        """Evaluate the surrogate at one dictionary of input variable values.

        Parameters
        ----------
        name_to_value_dict : dict
            Mapping from mixed variable names to physical input values.

        Returns
        -------
        openturns.Point
            Surrogate prediction at the provided input point.

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
        """Generate nested mixed surrogate samples.

        Parameters
        ----------
        num_epistemic_samples : int
            Number of outer-loop epistemic samples to generate.
        num_aleatory_samples : int
            Number of inner-loop aleatory samples to generate for each
            epistemic sample.
        method : {"MC", "LHS", "Sobol", "Halton"}, optional
            Sampling method used for both epistemic and aleatory samples.

        Returns
        -------
        sample_points : ndarray
            Sampled input points in physical variable space.
        sample_values : ndarray
            Surrogate output values evaluated at ``sample_points``.

        Raises
        ------
        ValueError
            If ``method`` is not a supported sampling method.

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
        """Aggregate sampled surrogate outputs by epistemic sample.

        Returns
        -------
        meta_to_cdf : dict
            Mapping from internal epistemic sample identifiers to empirical CDF
            objects.
        quantiles_data : ndarray
            Matrix of empirical CDF quantiles. Columns correspond to epistemic
            samples.
        probabilities_data : ndarray
            Matrix of empirical CDF probabilities. Columns correspond to
            epistemic samples.

        """
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
        """Plot horsetail empirical CDF curves for surrogate samples.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis. If omitted, ``"Output value"`` is used.
        plot_title : str, optional
            Plot title. If omitted, a default title is used.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If surrogate samples have not been generated with :meth:`sample`.

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
        """Plot a probability box for surrogate samples.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis. If omitted, ``"Output value"`` is used.
        plot_title : str, optional
            Plot title. If omitted, a default title is used.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        """

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
        """Return surrogate p-box bounds at one probability level.

        Parameters
        ----------
        p_level : float
            Probability level in the interval [0, 1].

        Returns
        -------
        min_quantile : float
            Lower p-box quantile bound at ``p_level``.
        max_quantile : float
            Upper p-box quantile bound at ``p_level``.

        Raises
        ------
        ValueError
            If ``p_level`` is outside [0, 1].

        """
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
        """Return the first-order Sobol index for one mixed variable.

        Parameters
        ----------
        var_name : str
            Mixed variable name.

        Returns
        -------
        float
            First-order Sobol sensitivity index.

        """
        for i,name in enumerate(self.all_var_to_dist.keys()):
            if var_name==name:
                return self._Chaos_SI.getSobolIndex(i)

    def get_total_order_sobol(self,var_name):
        """Return the total-order Sobol index for one mixed variable.

        Parameters
        ----------
        var_name : str
            Mixed variable name.

        Returns
        -------
        float
            Total-order Sobol sensitivity index.

        """
        for i,name in enumerate(self.all_var_to_dist.keys()):
            if var_name==name:
                return self._Chaos_SI.getSobolTotalIndex(i)

    def get_interaction_sobol(self,var_names:list):
        """Return the grouped Sobol index for a set of mixed variables.

        Parameters
        ----------
        var_names : list of str
            Mixed variable names included in the interaction group.

        Returns
        -------
        float
            Grouped Sobol sensitivity index.

        """
        inds=[]
        for i,name in enumerate(self.all_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        return self._Chaos_SI.getSobolGroupedIndex(inds)

    def get_total_interaction_sobol(self,var_names:list):
        """Return the grouped total-order Sobol index for mixed variables.

        Parameters
        ----------
        var_names : list of str
            Mixed variable names included in the interaction group.

        Returns
        -------
        float
            Grouped total-order Sobol sensitivity index.

        """
        inds=[]
        for i,name in enumerate(self.all_var_to_dist.keys()):
            for var_name in var_names:
                if var_name==name:
                    inds.append(i)

        return self._Chaos_SI.getSobolGroupedTotalIndex(inds)

    def plot_total_order_sobol(self,plot_title="",save_dir=None):
        """Plot total-order Sobol indices for all mixed variables.

        Parameters
        ----------
        plot_title : str, optional
            Plot title. If omitted, a default title is used.
        save_dir : str or path-like, optional
            Directory where the plot image is saved. If omitted, the plot is not
            written to disk.

        Returns
        -------
        None

        """
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


#--------------------------Local one at a time sensitivity post process class
class LocalSensitivityPost:
    """Post-process local one-at-a-time sensitivity results.

    This class computes finite-difference sensitivities from trade-study
    results generated by a local sensitivity workflow.

    Parameters
    ----------
    tradestudy : Tradestudy
        Trade-study results container used for post-processing.

    """

    def __init__(self, tradestudy: Tradestudy):
        """Initialize a local sensitivity post-processing helper.

        Parameters
        ----------
        tradestudy : Tradestudy
            Trade-study results container used for post-processing.

        """
        self.tradestudy = tradestudy

    def compute_sensitivity(self, output_name, method="forward"):
        """Compute local sensitivities for one output variable.

        Parameters
        ----------
        output_name : str
            Output variable name to analyze.
        method : {"forward", "backward", "central"}, optional
            Finite-difference method used to compute sensitivities.

        Returns
        -------
        var_to_sens : dict
            Nested dictionary mapping ``output_name`` to local sensitivities for
            each perturbed input variable.
        max_dict : dict
            Dictionary containing the variable with the largest absolute
            sensitivity and that sensitivity magnitude.

        Raises
        ------
        ValueError
            If ``method`` is not ``"forward"``, ``"backward"``, or
            ``"central"``.

        """
        trade_name = self.tradestudy.trade_name
        ts = self.tradestudy.study.control.trade_studies[trade_name]
        perturbation_vars = list(ts.variable_to_sensitivity_data.keys())
        # use np.max and np.min to locate
        outs = np.array(self.tradestudy.get_variable(output_name))
        var_to_sens = {output_name: {}}
        max_sense = 0.0
        max_sense_var = None
        for var_name in perturbation_vars:
            vals = self.tradestudy.get_variable(var_name)
            base = vals[0]
            pert_up = np.max(vals)
            pert_down = np.min(vals)
            out_base = outs[0]
            out_up = outs[np.argmax(vals)]
            out_down = outs[np.argmin(vals)]
            if method == "forward":
                sensitivity = (out_up - out_base) / (pert_up - base)
            elif method == "backward":
                sensitivity = (out_base - out_down) / (base - pert_down)
            elif method == "central":
                sensitivity = (out_up - out_down) / (pert_up - pert_down)
            else:
                raise ValueError(
                    f"Method {method} not recognized. Use 'forward', 'backward', or 'central'."
                )

            if abs(sensitivity) > max_sense:
                max_sense = abs(sensitivity)
                max_sense_var = var_name

            var_to_sens[output_name][var_name] = sensitivity

        max_dict = {"variable": max_sense_var, "sensitivity": max_sense}

        return var_to_sens, max_dict
