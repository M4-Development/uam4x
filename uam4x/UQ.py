from __future__ import annotations
import numpy as np
import scipy.stats as stats
import openturns as ot
import matplotlib.pyplot as plt
from math import comb
import copy
from uam4x.utils import log

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

#--------------------Aleatory Suite-------------------------------------
class AleatoryAnalysis:
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
        ot.RandomGenerator.Reset()
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
        return UQanalysis

@register_workflow
class BasicAleatory:
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
class RegressionPCE(BasicAleatory):
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
            log.warning(
                "Regression PCE is lightly sampled: "
                f"polynomial order={self.poly_order}, num_variables={num_aleatory}, "
                f"basis_size={min_terms}, num_samples={self.num_samples}. "
                "Consider 2-3x oversampling relative to the basis size."
            )

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
class SobolSensitivity(BasicAleatory):
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




#--------------------Epistemic Suite-------------------------------------
class EpistemicAnalysis:
    def __init__(self,UQ_workflow,seed=None):
        self.UQ_workflow=UQ_workflow
        self.seed=seed if seed is not None else 42
        self.epistemic_var_to_vals={}

    def add_epistemic_variable(self,name,interval):
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

    def get_states(self):
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)

        return self.UQ_workflow._get_states(self)

    def _to_dict(self):
        return {
            "type":self.__class__.__name__,
            "seed":self.seed,
            "epistemic_variables":{name:interval
                                    for name,interval in self.epistemic_var_to_vals.items()},
            "analysis_workflow":self.UQ_workflow._to_dict(),
        }

    @classmethod
    def _from_dict(cls,epistemic_dict):
        seed=epistemic_dict['seed']
        workflow_dict=epistemic_dict['analysis_workflow']
        workflow=workflow_name_to_class_registry[workflow_dict["name"]]._from_dict(
            workflow_dict
        )
        epistemic_analysis=cls(workflow,seed)
        for name,interval_data in epistemic_dict["epistemic_variables"].items():
            epistemic_analysis.add_epistemic_variable(name,np.array(interval_data))

        return epistemic_analysis

@register_workflow
class BasicEpistemic:
    def __init__(self,num_samples,SamplingClass):
        self.num_samples=num_samples
        self.SamplingClass=SamplingClass

    def _get_states(self,EpistemicAnalysisObj):

        epistemic_to_vals=EpistemicAnalysisObj.epistemic_var_to_vals

        epistemic_to_distribution={}
        for name,interval in epistemic_to_vals.items():
            low=interval[0]
            high=interval[1]
            epistemic_to_distribution[name]=UniformDistribution(low,high)

        epistemic_samples, epistemic_var_name_to_col =self.SamplingClass.sample(
            epistemic_to_distribution,self.num_samples
            )

        all_states=[]
        for row in epistemic_samples:
            local_state={}
            for name,col in epistemic_var_name_to_col.items():
                local_state[name]=row[col]
            all_states.append(local_state)

        return all_states

    def _to_dict(self):
        return {
            "name":self.__class__.__name__,
            "sampler": self.SamplingClass.__name__,
            "num_samples":self.num_samples
        }

    @classmethod
    def _from_dict(cls,workflow_dict):
        sampler=sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples=workflow_dict['num_samples']
        return cls(num_samples,sampler)

#--------------------Mixed Analysis Suite-------------------------------------
class MixedAnalysis:
    _META_STATE_NAME = "__IVP__STATE__META"

    def __init__(self,UQ_workflow,seed=None):

        if not isinstance(UQ_workflow,(BasicMixedIVP,BasicSOP,MixedRegressionPCE)):
            raise ValueError(
                "MixedAnalysis requires a mixed workflow instance of type "
                "BasicMixedIVP, BasicSOP, or MixedRegressionPCE."
            )
        self.UQ_workflow=UQ_workflow
        self.seed = seed if seed is not None else 42
        self.aleatory_var_to_dist={}
        self.epistemic_var_to_vals={}

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

    def get_states(self):
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        return self.UQ_workflow._get_states(self)


    def _to_dict(self):

        return {
            "type":self.__class__.__name__,
            "seed":self.seed,
            "aleatory_variables":{name:dist._to_dict()
                                  for name,dist in self.aleatory_var_to_dist.items()},

            "epistemic_variables":{name:interval
                                   for name,interval in self.epistemic_var_to_vals.items()},
            "analysis_workflow":self.UQ_workflow._to_dict(),
        }

    @classmethod
    def _from_dict(cls,IVPdict):
        seed = IVPdict["seed"]
        workflow_dict = IVPdict["analysis_workflow"]
        workflow = workflow_name_to_class_registry[workflow_dict["name"]]._from_dict(
            workflow_dict
        )
        ivp_analysis=cls(workflow,seed)
        for name,distribution_data in IVPdict["aleatory_variables"].items():
            dist=distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)

            ivp_analysis.add_aleatory_variable(name,dist)

        for name,interval_data in IVPdict["epistemic_variables"].items():
            ivp_analysis.add_epistemic_variable(name,np.array(interval_data))

        return ivp_analysis


@register_workflow
class BasicSOP:

    _META_STATE_NAME = "__IVP__STATE__META"

    def __init__(self,num_epistemic_samples,EpistemicSamplingClass,
               num_aleatory_samples,AleatorySamplingClass):
        self.num_epistemic_samples=num_epistemic_samples
        self.num_aleatory_samples=num_aleatory_samples
        self.EpistemicSamplingClass=EpistemicSamplingClass
        self.AleatorySamplingClass=AleatorySamplingClass

    def _get_states(self,IVPAnalysisObj):
        # SOPIVP uses epistemic entries as distribution-parameter uncertainty.
        epistemic_to_vals = IVPAnalysisObj.epistemic_var_to_vals
        aleatory_to_distribution = IVPAnalysisObj.aleatory_var_to_dist


        epistemic_to_distribution={}
        for name,interval in epistemic_to_vals.items():
            low=interval[0]
            high=interval[1]
            epistemic_to_distribution[name]=UniformDistribution(low,high)

        epistemic_samples, epistemic_var_name_to_col =self.EpistemicSamplingClass.sample(
            epistemic_to_distribution,self.num_epistemic_samples
            )

        epistemic_with_meta=np.hstack((np.arange(epistemic_samples.shape[0])[:,None],epistemic_samples,))
        #then generate aleatory inner loop
        all_aleatory_states=np.zeros((self.num_aleatory_samples*epistemic_samples.shape[0],
                                      len(aleatory_to_distribution)))
        row=0
        var_to_used_param={}
        for epistemic_var,col in epistemic_var_name_to_col.items():
            param=epistemic_var.split('.')[-1]
            aleatory='.'.join(epistemic_var.split('.')[:-1])
            if aleatory not in var_to_used_param:
                var_to_used_param[aleatory]={}

            var_to_used_param[aleatory][param]=col

        for i in range(self.num_epistemic_samples):
            temp_aleatory=copy.deepcopy(aleatory_to_distribution)

            #resolve the aleatory vars
            for name,interval in var_to_used_param.items():
                base_dist=temp_aleatory[name]
                base_class=base_dist.__class__
                dist_dict=base_dist._to_dict()
                for param_name,param_data in interval.items():
                    dist_dict[param_name]=epistemic_samples[i,param_data]
                aleatory_instance=base_class._from_dict(dist_dict)
                temp_aleatory[name]=aleatory_instance


            aleatory_samples, aleatory_var_name_to_col=self.AleatorySamplingClass.sample(
               temp_aleatory,self.num_aleatory_samples
            )
            all_aleatory_states[row:row+self.num_aleatory_samples,:]=aleatory_samples
            row+=self.num_aleatory_samples

        all_vars_to_col={self._META_STATE_NAME:0}
        for name,col in epistemic_var_name_to_col.items():
            all_vars_to_col[name]=col+1

        for name,col in aleatory_var_name_to_col.items():
            #account for meta offset
            all_vars_to_col[name]=col+(len(epistemic_var_name_to_col)+1)

        epistemic_states=np.repeat(epistemic_with_meta,
                                   self.num_aleatory_samples,
                                   axis=0)
        full_states_arr=np.hstack((epistemic_states,all_aleatory_states))
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
            "name":self.__class__.__name__,
            "epistemic_sampler":self.EpistemicSamplingClass.__name__,
            "aleatory_sampler":self.AleatorySamplingClass.__name__,
            "num_epistemic_samples":self.num_epistemic_samples,
            "num_aleatory_samples":self.num_aleatory_samples
        }

    @classmethod
    def _from_dict(cls,workflow_dict):
        e_sampler=sampler_name_to_class_registry[workflow_dict["epistemic_sampler"]]
        a_sampler=sampler_name_to_class_registry[workflow_dict["aleatory_sampler"]]
        ne_samps=workflow_dict["num_epistemic_samples"]
        na_samps=workflow_dict["num_aleatory_samples"]
        return cls(ne_samps,e_sampler,na_samps,a_sampler)


@register_workflow
class BasicMixedIVP:

    _META_STATE_NAME = "__IVP__STATE__META"

    def __init__(self,num_epistemic_samples,EpistemicSamplingClass,
               num_aleatory_samples,AleatorySamplingClass):
        self.num_epistemic_samples=num_epistemic_samples
        self.num_aleatory_samples=num_aleatory_samples
        self.EpistemicSamplingClass=EpistemicSamplingClass
        self.AleatorySamplingClass=AleatorySamplingClass

    def _get_states(self,IVPAnalysisObj):
        #generate the epistemic to distribution here
        aleatory_to_distribution=IVPAnalysisObj.aleatory_var_to_dist
        epistemic_to_vals=IVPAnalysisObj.epistemic_var_to_vals

        epistemic_to_distribution={}
        for name,interval in epistemic_to_vals.items():
            low=interval[0]
            high=interval[1]
            epistemic_to_distribution[name]=UniformDistribution(low,high)

        epistemic_samples, epistemic_var_name_to_col =self.EpistemicSamplingClass.sample(
            epistemic_to_distribution,self.num_epistemic_samples
            )

        epistemic_with_meta=np.hstack((np.arange(epistemic_samples.shape[0])[:,None],epistemic_samples,))
        #then generate aleatory inner loop
        all_aleatory_states=np.zeros((self.num_aleatory_samples*epistemic_samples.shape[0],
                                      len(aleatory_to_distribution)))
        row=0
        for i in range(self.num_epistemic_samples):
            aleatory_samples, aleatory_var_name_to_col=self.AleatorySamplingClass.sample(
               aleatory_to_distribution,self.num_aleatory_samples
            )
            all_aleatory_states[row:row+self.num_aleatory_samples,:]=aleatory_samples
            row+=self.num_aleatory_samples

        all_vars_to_col={self._META_STATE_NAME:0}
        for name,col in epistemic_var_name_to_col.items():
            all_vars_to_col[name]=col+1

        for name,col in aleatory_var_name_to_col.items():
            #account for meta offset
            all_vars_to_col[name]=col+(len(epistemic_var_name_to_col)+1)

        epistemic_states=np.repeat(epistemic_with_meta,
                                   self.num_aleatory_samples,
                                   axis=0)
        full_states_arr=np.hstack((epistemic_states,all_aleatory_states))
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
            "name":self.__class__.__name__,
            "epistemic_sampler":self.EpistemicSamplingClass.__name__,
            "aleatory_sampler":self.AleatorySamplingClass.__name__,
            "num_epistemic_samples":self.num_epistemic_samples,
            "num_aleatory_samples":self.num_aleatory_samples
        }

    @classmethod
    def _from_dict(cls,workflow_dict):
        e_sampler=sampler_name_to_class_registry[workflow_dict["epistemic_sampler"]]
        a_sampler=sampler_name_to_class_registry[workflow_dict["aleatory_sampler"]]
        ne_samps=workflow_dict["num_epistemic_samples"]
        na_samps=workflow_dict["num_aleatory_samples"]
        return cls(ne_samps,e_sampler,na_samps,a_sampler)


@register_workflow
class MixedRegressionPCE:
    def __init__(self,SamplingClass,num_samples,
                 poly_order):
        self.SamplingClass=SamplingClass
        self.num_samples=num_samples
        self.poly_order = poly_order

    def _get_states(self,IVPAnalysisObj):
        #generate the epistemic to distribution here
        aleatory_to_distribution=IVPAnalysisObj.aleatory_var_to_dist
        epistemic_to_vals=IVPAnalysisObj.epistemic_var_to_vals

        all_vars_to_dist={}
        for name,interval in epistemic_to_vals.items():
            low=interval[0]
            high=interval[1]
            all_vars_to_dist[name]=UniformDistribution(low,high)

        for name,dist in aleatory_to_distribution.items():
            all_vars_to_dist[name]=dist

        all_samples, all_vars_to_col = self.SamplingClass.sample(
            all_vars_to_dist,self.num_samples
            )

        #then aggregate states
        all_states=[]
        for row in all_samples:
            state_dict={}
            for name,col in all_vars_to_col.items():
                state_dict[name]=row[col]
            all_states.append(state_dict)

        return all_states

    def _check_analysis(self, SOPAnalysis):
        # check order and number of aleatory vs. num samples
        # from XIU Polynomial expansion, dimension of space is C(order+number of random vars,number of random vars)
        # should oversample 2-3x
        """Validate that workflow settings are consistent with requested analysis.

        Parameters
        ----------
        UQAnalysis : Any
            Input value for `UQAnalysis`.

        """
        num_vars = len(SOPAnalysis.aleatory_var_to_dist) + len(
            SOPAnalysis.epistemic_var_to_vals
        )
        min_terms = comb(self.poly_order + num_vars, self.poly_order)
        num_samples=self.num_samples
        if min_terms > num_samples:
            raise ValueError(
                f"Insufficient terms for polynomial chaos expansion of degree {self.poly_order} "
                f"for {num_vars} mixed variables"
            )
        elif min_terms * 3 > num_samples:
            log.warning(
                "Mixed regression PCE is lightly sampled: "
                f"polynomial order={self.poly_order}, num_variables={num_vars}, "
                f"basis_size={min_terms}, num_samples={num_samples}. "
                "Consider 2-3x oversampling relative to the basis size."
            )

    def _to_dict(self):
        return{
            "name":self.__class__.__name__,
            "sampler":self.SamplingClass.__name__,
            "num_samples":self.num_samples,
            "polynomial_order":self.poly_order
        }

    @classmethod
    def _from_dict(cls, workflow_dict):
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        poly_order = workflow_dict["polynomial_order"]
        return cls(sampler,num_samples, poly_order)




#--------------------Dempster-Shafer Analysis-------------------------------------
class DempsterShafer:
    pass
#--------------------Samplers-------------------------------------
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


#--------------------Distributions-------------------------------------
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
    ivp_debug = MixedAnalysis(
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
