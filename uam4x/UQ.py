from __future__ import annotations
from matplotlib.dates import SA
import numpy as np
import scipy.stats as stats
from numpy.random import default_rng
import openturns as ot
import matplotlib.pyplot as plt
from math import comb
import copy

# Global registry
workflow_name_to_class_registry = {}
sampler_name_to_class_registry = {}
distribution_name_to_class_registry = {}
post_workflow_name_to_class_registry = {}

# Orchestrator class, manages workflows and interfaces with uam4x


def register_workflow(Workflow):
    """Register a UQ workflow class in the global workflow registry.

    Parameters
    ----------
    Workflow : Any
        Workflow class to register.

    Returns
    -------
    Any
        The registered class (decorator passthrough).

    """
    workflow_name_to_class_registry[Workflow.__name__] = Workflow
    return Workflow


def register_sampler(Sampler):
    """Register a sampler class in the global sampler registry.

    Parameters
    ----------
    Sampler : Any
        Sampler class to register.

    Returns
    -------
    Any
        The registered class (decorator passthrough).

    """
    sampler_name_to_class_registry[Sampler.__name__] = Sampler
    return Sampler


def register_distribution(Distribution):
    """Register a distribution class in the global distribution registry.

    Parameters
    ----------
    Distribution : Any
        Distribution class to register.

    Returns
    -------
    Any
        The registered class (decorator passthrough).

    """
    distribution_name_to_class_registry[Distribution.__name__] = Distribution
    return Distribution


def register_post_workflow(PostWorkflow):
    """Register a post-processing workflow class in the global post registry.

    Parameters
    ----------
    PostWorkflow : Any
        Post-processing workflow class to register.

    Returns
    -------
    Any
        The registered class (decorator passthrough).

    """
    post_workflow_name_to_class_registry[PostWorkflow.__name__] = PostWorkflow
    return PostWorkflow


class UQAnalysis:
    # MIGHT CHANGE THIS TO ALEATORYANALYSIS,
    # THEN MAYBE HAVE A MIXED ANALYSIS
    # AND PURE EPISTEMIC ALL ESSENTIALLY AT PARAMETERSWEEP/OPTIMIZATION level
    """Represent the uqanalysis component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, UQ_workflow, seed=None):
        """Initialize object state and configuration.

        Parameters
        ----------
        UQ_workflow : Any
            Workflow object that generates uncertainty states.
        seed : Any, optional
            Random seed used for OpenTurns sample generation.

        """
        self.UQ_workflow = UQ_workflow
        self.seed = seed if seed is not None else 42
        self.aleatory_var_to_dist = {}
        self.post_spec = pre_to_post_workflow_registry[UQ_workflow.__class__.__name__]

    def _check_analysis(self):
        """Validate that workflow settings are consistent with requested analysis.

        Notes
        -----
        This method performs class-internal setup or validation logic.

        """
        if hasattr(self.UQ_workflow, "_check_analysis"):
            self.UQ_workflow._check_analysis(self)

    def add_aleatory_variable(self, var_name, PDFObject):
        """Associate one aleatory variable name with a probability distribution.

        Parameters
        ----------
        var_name : Any
            Variable name to analyze or evaluate.
        PDFObject : Any
            Distribution object associated with one aleatory variable.

        """
        self.aleatory_var_to_dist[var_name] = PDFObject

    def get_states(self):
        """Generate state dictionaries for all uncertainty samples in this workflow.

        Returns
        -------
        Any
            Requested value.

        """
        ot.RandomGenerator.SetSeed(self.seed)
        return self.UQ_workflow._get_states(self.aleatory_var_to_dist)

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "type": self.__class__.__name__,
            "seed": self.seed,
            "aleatory_variables": {
                name: dist._to_dict()
                for name, dist in self.aleatory_var_to_dist.items()
            },
            "analysis_workflow": self.UQ_workflow._to_dict(),
            "post_processing": self.post_spec.__name__ if self.post_spec else None,
        }

    @classmethod
    def _from_dict(cls, UQdict):

        # load in the workflow from dictionary
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        UQdict : Any
            Serialized uncertainty-analysis dictionary.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        workflow_dict = UQdict["analysis_workflow"]
        workflow = workflow_name_to_class_registry[workflow_dict["name"]]._from_dict(
            workflow_dict
        )
        # load in seed
        seed = UQdict["seed"]
        # instantiate class
        UQanalysis = cls(workflow, seed)
        # iterate through the aleatory variables to readd
        for name, distribution_data in UQdict["aleatory_variables"].items():
            dist = distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)
            UQanalysis.add_aleatory_variable(name, dist)
        if UQdict["post_processing"] is not None:
            UQanalysis.post_spec = post_workflow_name_to_class_registry[
                UQdict["post_processing"]
            ]
        return UQanalysis


@register_workflow
class IVPAnalysis:
    _META_STATE_NAME = "__IVP__STATE__META"

    def __init__(self,AleatorySamplingClass,EpistemicSamplingClass,
                 num_epistemic_samples,num_aleatory_samples,seed=None):
        self.AleatorySamplingClass=AleatorySamplingClass
        if EpistemicSamplingClass not in [LatinHypercubeSampler,
                                          SobolSampler,HaltonSampler]:
            raise ValueError(f'Sampler must be low discrepancy sampler')

        self.EpistemicSamplingClass=EpistemicSamplingClass
        self.aleatory_var_to_dist={}
        self.num_aleatory_samples=num_aleatory_samples
        self.epistemic_var_to_vals={}
        self.num_epistemic_samples=num_epistemic_samples
        self._epistemic_var_to_dist={}
        self.seed = seed if seed is not None else 42
        self.post_spec=IVPAnalysisPost

    def add_aleatory_variable(self,name,PDFObject):
        if name == self._META_STATE_NAME:
            raise ValueError(
                f"'{self._META_STATE_NAME}' is reserved for IVP internal metadata. "
                "Choose a different aleatory variable name."
            )
        self.aleatory_var_to_dist[name]=PDFObject

    def add_epistemic_variable(self,name,interval):
        if name == self._META_STATE_NAME:
            raise ValueError(
                f"'{self._META_STATE_NAME}' is reserved for IVP internal metadata. "
                "Choose a different epistemic variable name."
            )
        if not isinstance(interval,np.ndarray):
            raise ValueError("Epistemic interval must be a numpy array with shape (2,).")
        if interval.shape != (2,):
            raise ValueError("Epistemic interval must have shape (2,), i.e. [low, high].")
        if not np.issubdtype(interval.dtype, np.number):
            raise ValueError("Epistemic interval entries must be numeric.")
        if not np.all(np.isfinite(interval)):
            raise ValueError("Epistemic interval entries must be finite values.")
        if interval[0] > interval[1]:
            raise ValueError("Epistemic interval must satisfy low <= high.")
        self.epistemic_var_to_vals[name]=interval
        self._epistemic_var_to_dist[name]=UniformDistribution(0,1)


    def get_states(self):
        ot.RandomGenerator.SetSeed(self.seed)
        #first generate epistemic outerloop

        epistemic_samples, epistemic_var_name_to_col =self.EpistemicSamplingClass.sample(
            self._epistemic_var_to_dist,self.num_epistemic_samples
            )

        for name,col in epistemic_var_name_to_col.items():
            interval_min=self.epistemic_var_to_vals[name][0]
            interval_max=self.epistemic_var_to_vals[name][1]
            epistemic_samples[:,col]=epistemic_samples[:,col]*(interval_max-interval_min)+interval_min

        epistemic_with_meta=np.hstack((np.arange(epistemic_samples.shape[0])[:,None],epistemic_samples,))
        #then generate aleatory inner loop
        aleatory_samples, aleatory_var_name_to_col=self.AleatorySamplingClass.sample(
            self.aleatory_var_to_dist,self.num_aleatory_samples
        )

        all_vars_to_col={self._META_STATE_NAME:0}
        for name,col in epistemic_var_name_to_col.items():
            all_vars_to_col[name]=col+1

        for name,col in aleatory_var_name_to_col.items():
            #account for meta offset
            all_vars_to_col[name]=col+(len(epistemic_var_name_to_col)+1)

        epistemic_states=np.repeat(epistemic_with_meta,aleatory_samples.shape[0],axis=0)
        aleatory_states=np.tile(aleatory_samples, (epistemic_with_meta.shape[0], 1))
        full_states_arr=np.hstack((epistemic_states,aleatory_states))
        #then aggregate states
        all_states=[]
        for row in full_states_arr:
            state_dict={}
            for name,col in all_vars_to_col.items():
                state_dict[name]=row[col]
            all_states.append(state_dict)

        return all_states

    def _to_dict(self):

        return {
            "type":self.__class__.__name__,
            "seed":self.seed,
            "aleatory_sampling_class_name":self.AleatorySamplingClass.__name__,
            "epistemic_sampling_class_name":self.EpistemicSamplingClass.__name__,
            "num_aleatory_samples":self.num_aleatory_samples,
            "num_epistemic_samples":self.num_epistemic_samples,
            "aleatory_variables":{name:dist._to_dict()
                                  for name,dist in self.aleatory_var_to_dist.items()},

            "epistemic_variables":{name:interval
                                   for name,interval in self.epistemic_var_to_vals.items()},
        }

    @classmethod
    def _from_dict(cls,IVPdict):
        seed = IVPdict["seed"]
        aleatory_sampling_class=sampler_name_to_class_registry[IVPdict["aleatory_sampling_class_name"]]
        epistemic_sampling_class=sampler_name_to_class_registry[IVPdict["epistemic_sampling_class_name"]]
        num_epistemic_samples=IVPdict["num_epistemic_samples"]
        num_aleatory_samples=IVPdict["num_aleatory_samples"]
        ivp_analysis=cls(aleatory_sampling_class,epistemic_sampling_class,
                         num_epistemic_samples,num_aleatory_samples,seed)
        for name,distribution_data in IVPdict["aleatory_variables"].items():
            dist=distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)

            ivp_analysis.add_aleatory_variable(name,dist)

        for name,interval_data in IVPdict["epistemic_variables"].items():
            ivp_analysis.add_epistemic_variable(name,np.array(interval_data))

        return ivp_analysis


@register_workflow
class SecondOrderAnalysis:
    pass


@register_workflow
class BasicStats:
    """Represent the basicstats component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, num_samples, SamplingClass):
        """Initialize object state and configuration.

        Parameters
        ----------
        num_samples : Any
            Number of samples to generate.
        SamplingClass : Any
            Sampler class used to generate uncertainty samples.

        """
        self.num_samples = num_samples
        self.SamplingClass = SamplingClass

    def _get_states(self, aleatory_to_distribution):
        # states should be a list of dictionaries
        """Generate analysis state dictionaries from configured distributions.

        Parameters
        ----------
        aleatory_to_distribution : Any
            Mapping from aleatory variable names to distributions.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        sample_points, var_name_to_col = self.SamplingClass.sample(
            aleatory_to_distribution, self.num_samples
        )
        states = []
        for row in range(sample_points.shape[0]):
            local_state = {}
            for name, col in var_name_to_col.items():
                local_state[name] = sample_points[row, col]
            states.append(local_state)
        return states

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "name": self.__class__.__name__,
            "sampler": self.SamplingClass.__name__,
            "num_samples": self.num_samples,
        }

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        workflow_dict : Any
            Serialized workflow dictionary.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        return cls(num_samples, sampler)


@register_workflow
class RegressionPCE(BasicStats):
    """Represent the regressionpce component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, num_samples, SamplingClass, poly_order):
        """Initialize object state and configuration.

        Parameters
        ----------
        num_samples : Any
            Number of samples to generate.
        SamplingClass : Any
            Sampler class used to generate uncertainty samples.
        poly_order : Any
            Polynomial order used for chaos expansions.

        """
        super().__init__(num_samples, SamplingClass)
        self.poly_order = poly_order

    def _check_analysis(self, UQAnalysis):
        # check order and number of aleatory vs. num samples
        # from XIU Polynomial expansion, dimension of space is C(order+number of random vars,number of random vars)
        # should oversample 2-3x
        """Validate that workflow settings are consistent with requested analysis.

        Parameters
        ----------
        UQAnalysis : Any
            Input value for `UQAnalysis`.

        """
        num_aleatory = len(UQAnalysis.aleatory_var_to_dist)
        min_terms = comb(self.poly_order + num_aleatory, self.poly_order)
        if min_terms > self.num_samples:
            raise ValueError(
                f"Insufficient terms for polynomial chaos expansion of degree {self.poly_order} for {num_aleatory} random variables"
            )
        elif min_terms * 3 > self.num_samples:
            print("Warning")

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "name": self.__class__.__name__,
            "sampler": self.SamplingClass.__name__,
            "num_samples": self.num_samples,
            "polynomial_order": self.poly_order,
        }

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        workflow_dict : Any
            Serialized workflow dictionary.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        poly_order = workflow_dict["polynomial_order"]
        return cls(num_samples, sampler, poly_order)


@register_workflow
class GaussQuadraturePCE:
    """Represent the gaussquadraturepce component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, poly_order):
        """Initialize object state and configuration.

        Parameters
        ----------
        poly_order : Any
            Polynomial order used for chaos expansions.

        """
        self.poly_order = poly_order

    def _check_analysis(self, UQAnalysis):
        """Validate that workflow settings are consistent with requested analysis.

        Parameters
        ----------
        UQAnalysis : Any
            Input value for `UQAnalysis`.

        """
        num_aleatory = len(UQAnalysis.aleatory_var_to_dist)
        if num_aleatory > 5:
            raise ValueError(
                "Quadrature based PCE may not be efficient for more than 5 aleatory variables"
            )

    def _get_states(self, aleatory_to_distribution):
        """Generate analysis state dictionaries from configured distributions.

        Parameters
        ----------
        aleatory_to_distribution : Any
            Mapping from aleatory variable names to distributions.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        num_aleatory = len(aleatory_to_distribution)

        if num_aleatory == 1:
            ((name, dist),) = aleatory_to_distribution.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(aleatory_to_distribution.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name] = i
            dist = ot.JointDistribution(joint)

        # Experiment with joint distribution with k marginals
        # the points per var is poly deg+1, the total number of points is
        # (deg+1)^k for full tensor implementation (used here)
        points_per_var = [self.poly_order + 1] * num_aleatory
        experiment = ot.GaussProductExperiment(dist, points_per_var)
        nodes, weights = experiment.generateWithWeights()
        nodes, weights = np.array(nodes), np.array(weights)

        states = []
        for row in range(nodes.shape[0]):
            local_state = {"weight": weights[row]}
            for name, col in var_name_to_col.items():
                local_state[name] = nodes[row, col]
            states.append(local_state)

        return states

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {"name": self.__class__.__name__, "polynomial_order": self.poly_order}

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        workflow_dict : Any
            Serialized workflow dictionary.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        poly_order = workflow_dict["polynomial_order"]
        return cls(poly_order)


@register_workflow
class SobolSensitivity(BasicStats):
    """Represent the sobolsensitivity component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, num_samples, SamplingClass, computeSecondOrder: bool = False):
        """Initialize object state and configuration.

        Parameters
        ----------
        num_samples : Any
            Number of samples to generate.
        SamplingClass : Any
            Sampler class used to generate uncertainty samples.
        computeSecondOrder : bool, optional
            Whether second-order Sobol terms are generated.

        """
        super().__init__(num_samples, SamplingClass)
        self.computeSecondOrder = computeSecondOrder

    def _get_states(self, aleatory_to_distribution):
        """Generate analysis state dictionaries from configured distributions.

        Parameters
        ----------
        aleatory_to_distribution : Any
            Mapping from aleatory variable names to distributions.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        ot.ResourceMap.Set(
            "SobolIndicesExperiment-SamplingMethod",
            self.SamplingClass._OT_ResourceMapName,
        )
        if len(aleatory_to_distribution) == 1:
            ((name, dist),) = aleatory_to_distribution.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(aleatory_to_distribution.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name] = i
            dist = ot.JointDistribution(joint)

        experiment = ot.SobolIndicesExperiment(
            dist, self.num_samples, self.computeSecondOrder
        )
        results = np.array(experiment.generate())

        # check matrix size:
        # from OpenTurns documentation: if computeSecondOrder is false, number of rows is N(2+nx) where N is num
        # samples and nx is number of marginals (distributions/aleatory vars)
        # if computeSecondOrder is true and nx=2, same as above
        # elif computeSecondOrder and nx!=2, rows is N(2+2*nx)
        if (not self.computeSecondOrder) or len(aleatory_to_distribution) == 2:
            if results.shape[0] != self.num_samples * (
                2 + len(aleatory_to_distribution)
            ):
                raise ValueError(f"Number of rows is unexpected {results.shape[0]}")
        else:
            if results.shape[0] != self.num_samples * (
                2 + 2 * len(aleatory_to_distribution)
            ):
                raise ValueError(f"Number of rows is unexpected {results.shape[0]}")

        states = []
        for row in range(results.shape[0]):
            local_state = {}
            for name, col in var_name_to_col.items():
                local_state[name] = results[row, col]
            states.append(local_state)

        return states

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "name": self.__class__.__name__,
            "sampler": self.SamplingClass.__name__,
            "num_samples": self.num_samples,
            "second_order_sobol": self.computeSecondOrder,
        }

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        workflow_dict : Any
            Serialized workflow dictionary.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        second_order_sobol = workflow_dict["second_order_sobol"]

        sobol_workflow = cls(num_samples, sampler, second_order_sobol)

        return sobol_workflow


# Define post process classes:
@register_post_workflow
class BasicStatsPost:
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
            raise ValueError()

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
            raise ValueError()

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
            raise ValueError()
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
            raise ValueError
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
            raise ValueError

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
            raise ValueError()
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
            raise ValueError

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


class MixedBasicStatsPost(BasicStatsPost):
    """Represent the mixedbasicstatspost component in the UAM4X workflow.

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

    def plot_probability_box():
        """Placeholder for probability-box visualization for mixed uncertainty.

        Notes
        -----
        This method performs class-internal setup or validation logic.

        """
        pass

    def compute_confidence_intervals():
        """Placeholder for confidence-interval calculations.

        Notes
        -----
        This method performs class-internal setup or validation logic.

        """
        pass


@register_post_workflow
class RegressionPCEPost(BasicStatsPost):
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
                    raise ValueError()

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

        return Surrogate(aleatory_var_to_dist, res)


@register_post_workflow
class GaussQuadtraturePCEPost(BasicStatsPost):
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
                    raise ValueError()

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

        return Surrogate(aleatory_var_to_dist, res)


class Surrogate:
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
            raise ValueError()
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

    def plot_cdf(self, save_dir=None):
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
        ax.set_title(f"CDF of surrogate output", fontsize=16, fontweight="bold")
        ax.set_xlabel("Output", fontsize=14)
        ax.set_ylabel("Probability", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/CDF_surrogate_output.png", dpi=300, bbox_inches="tight"
            )

        return np.hstack((quantiles, probabilities))

    def plot_pdf(self, bins=20, save_dir=None):
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
        ax.set_title(f"PDF of surrogate output", fontsize=16, fontweight="bold")
        ax.set_xlabel("Output", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_dir is not None:
            plt.savefig(
                f"{save_dir}/PDF_surrogate_output.png", dpi=300, bbox_inches="tight"
            )

        return ax


@register_post_workflow
class SobolSensitivityPost(BasicStatsPost):
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

    def init(self, tradestudy):
        """Execute init.

        Parameters
        ----------
        tradestudy : Any
            Input value for `tradestudy`.

        """
        super().init(tradestudy)

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


@register_post_workflow
class IVPAnalysisPost:
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
            raise ValueError(f'Should be cleanly divisible here, some error')

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
            raise ValueError(f'p_level must be between 0 and 1')


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


# Define various sampling methods


class BaseSampler:
    """Define a base interface for basesampler.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    @staticmethod
    def sample():
        """Generate new surrogate samples using a selected sampling strategy.

        Notes
        -----
        This method performs class-internal setup or validation logic.

        """
        pass


@register_sampler
class MonteCarloSampler(BaseSampler):
    """Provide montecarlosampler behavior for uncertainty workflows.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    _OT_ResourceMapName = "MonteCarlo"

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Generate new surrogate samples using a selected sampling strategy.

        Parameters
        ----------
        aleatory_to_dist : Any
            Mapping from aleatory variable names to distributions.
        num_samples : Any
            Number of samples to generate.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        if len(aleatory_to_dist) == 1:
            ((name, dist),) = aleatory_to_dist.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name] = i
            dist = ot.JointDistribution(joint)
        experiment = ot.MonteCarloExperiment(dist, num_samples)
        samples = np.array(experiment.generate())

        return samples, var_name_to_col


@register_sampler
class LatinHypercubeSampler(BaseSampler):
    """Provide latinhypercubesampler behavior for uncertainty workflows.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    _OT_ResourceMapName = "LHS"

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        # first generate the evenly spaced values on [0,1]
        """Generate new surrogate samples using a selected sampling strategy.

        Parameters
        ----------
        aleatory_to_dist : Any
            Mapping from aleatory variable names to distributions.
        num_samples : Any
            Number of samples to generate.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        if len(aleatory_to_dist) == 1:
            ((name, dist),) = aleatory_to_dist.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name] = i
            dist = ot.JointDistribution(joint)
        experiment = ot.LHSExperiment(dist, num_samples)
        samples = np.array(experiment.generate())

        return samples, var_name_to_col


@register_sampler
class SobolSampler(BaseSampler):

    # Note this use here is for SobolExperiment, where the
    # Quasi monte carlo uses the Sobol sampler
    """Provide sobolsampler behavior for uncertainty workflows.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """
    _OT_ResourceMapName = "QMC"

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Generate new surrogate samples using a selected sampling strategy.

        Parameters
        ----------
        aleatory_to_dist : Any
            Mapping from aleatory variable names to distributions.
        num_samples : Any
            Number of samples to generate.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        if len(aleatory_to_dist) == 1:
            ((name, dist),) = aleatory_to_dist.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name] = i
            dist = ot.JointDistribution(joint)
        sobolseq = ot.SobolSequence(dist.getDimension())
        experiment = ot.LowDiscrepancyExperiment(sobolseq, dist, num_samples)
        samples = np.array(experiment.generate())

        return samples, var_name_to_col


@register_sampler
class HaltonSampler(BaseSampler):
    """Provide haltonsampler behavior for uncertainty workflows.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Generate new surrogate samples using a selected sampling strategy.

        Parameters
        ----------
        aleatory_to_dist : Any
            Mapping from aleatory variable names to distributions.
        num_samples : Any
            Number of samples to generate.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        if len(aleatory_to_dist) == 1:
            ((name, dist),) = aleatory_to_dist.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(aleatory_to_dist.items()):
                joint.append(dist._OT_Dist())
                var_name_to_col[name] = i
            dist = ot.JointDistribution(joint)
        haltonseq = ot.HaltonSequence(dist.getDimension())
        experiment = ot.LowDiscrepancyExperiment(haltonseq, dist, num_samples)
        samples = np.array(experiment.generate())

        return samples, var_name_to_col


# Define various continuous distribution classes
@register_distribution
class NormalDistribution:
    """Represent a normaldistribution with serialization and sampling support.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, mean, std):
        """Initialize object state and configuration.

        Parameters
        ----------
        mean : Any
            Mean value of the normal distribution.
        std : Any
            Standard deviation of the normal distribution.

        """
        self.mean = mean
        self.std = std

    def _OT_Dist(self):
        """Return the equivalent OpenTurns distribution object.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        return ot.Normal(self.mean, self.std)

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {"name": self.__class__.__name__, "mean": self.mean, "std": self.std}

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        data : Any
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        return cls(data["mean"], data["std"])


@register_distribution
class UniformDistribution:
    """Represent a uniformdistribution with serialization and sampling support.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, low, high):
        """Initialize object state and configuration.

        Parameters
        ----------
        low : Any
            Lower bound of the uniform distribution.
        high : Any
            Upper bound of the uniform distribution.

        """
        self.low = low
        self.high = high
        self.scale = self.high - self.low

    def _OT_Dist(self):
        """Return the equivalent OpenTurns distribution object.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        return ot.Uniform(self.low, self.high)

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {"name": self.__class__.__name__, "low": self.low, "high": self.high}

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        data : Any
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        return cls(data["low"], data["high"])


@register_distribution
class LogNormalDistribution:
    """Represent a lognormaldistribution with serialization and sampling support.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, mu_log, sigma_log, gamma=0.0):
        """Initialize object state and configuration.

        Parameters
        ----------
        mu_log : Any
            Log-space mean parameter for the log-normal distribution.
        sigma_log : Any
            Log-space standard deviation for the log-normal distribution.
        gamma : Any, optional
            Shift parameter for the log-normal distribution.

        """
        self.mu_log = mu_log
        self.sigma_log = sigma_log
        self.gamma = gamma

    def _OT_Dist(self):
        """Return the equivalent OpenTurns distribution object.

        Returns
        -------
        Any
            Return value produced by this method.

        """
        return ot.LogNormal(self.mu_log, self.sigma_log, self.gamma)

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "name": self.__class__.__name__,
            "mu": self.mu_log,
            "sigma": self.sigma_log,
            "gamma": self.gamma,
        }

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        data : Any
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        return cls(data["mu"], data["sigma"], data.get("gamma", 0.0))


# Pair all workflow classes with their post process classes
pre_to_post_workflow_registry = {
    "BasicStats": BasicStatsPost,
    "RegressionPCE": RegressionPCEPost,
    "SobolSensitivity": SobolSensitivityPost,
    "GaussQuadraturePCE": GaussQuadtraturePCEPost,
}
if __name__ == "__main__":

    # Test the LHS functionality:
    uni = UniformDistribution(0.0, 10.0)
    norm = NormalDistribution(50.0, 25.0)

    sample_alteatory = {"var1": uni, "var2": norm}
    sample_single = {"var": uni}
    n = 500
    x = LatinHypercubeSampler.sample(sample_alteatory, n)

    j = ot.JointDistribution([uni._OT_Dist(), norm._OT_Dist()])
    print("hello")

    # IVP mixed uncertainty debug case:
    # 2 epistemic interval variables + 2 aleatory random variables.
    ivp_debug = IVPAnalysis(
        AleatorySamplingClass=MonteCarloSampler,
        EpistemicSamplingClass=LatinHypercubeSampler,
        num_epistemic_samples=6,
        num_aleatory_samples=25,
        seed=42,
    )

    # Aleatory variables
    ivp_debug.add_aleatory_variable("wind_gust", NormalDistribution(8.0, 1.5))
    ivp_debug.add_aleatory_variable("payload_mass", UniformDistribution(120.0, 180.0))

    # Epistemic interval variables (must be np.ndarray with shape (2,))
    ivp_debug.add_epistemic_variable("drag_factor", np.array([0.92, 1.08]))
    ivp_debug.add_epistemic_variable("battery_eff", np.array([0.88, 0.96]))

    # get_states() is currently under development and is called here for debugging.
    ivp_states = ivp_debug.get_states()
    print("IVP debug get_states result type:", type(ivp_states))
