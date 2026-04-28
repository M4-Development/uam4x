from __future__ import annotations
import numpy as np
import openturns as ot
from math import comb
import copy
from uam4x.utils import log, is_bool, is_integer
import pandas as pd
from math import ceil

def gen_csv_states(csv_path,col_order):
    """Load uncertainty states from a CSV file.

    Parameters
    ----------
    csv_path : str or path-like
        Path to a CSV file containing one column for each uncertainty variable.
    col_order : sequence of str
        Column names to select from the CSV file and the order in which they
        should appear in each returned state dictionary.

    Returns
    -------
    list of dict
        State dictionaries generated from the selected CSV columns. Each
        dictionary maps variable names to the corresponding row values.

    """
    df=pd.read_csv(csv_path)
    df=df[col_order]
    return df.to_dict(orient='records')


#--------------------Aleatory Suite-------------------------------------
class AleatoryBaseClass:
    def __init__(self):
        self.aleatory_var_to_dist={}

    def add_aleatory_variable(self,name,PDFObject):
        """Register one aleatory variable and its distribution.

        Parameters
        ----------
        name : str
            Name of the UAM4X state variable to sample.
        PDFObject : object
            Distribution object associated with ``name``. The object must
            provide ``_OT_Dist`` and ``_to_dict`` methods.

        Returns
        -------
        None

        """
        self.aleatory_var_to_dist[name]=PDFObject

class BasicAleatory(AleatoryBaseClass):
    """Generate aleatory states with a sampling class.

    This workflow samples each registered aleatory distribution and returns one
    UAM4X state dictionary per sample.

    Parameters
    ----------
    num_samples : int
        Number of aleatory samples to generate.
    SamplingClass : type
        Sampler class used to generate samples. The class must provide a
        ``sample`` method accepting a variable-to-distribution mapping and a
        sample count.


    """

    def __init__(self, num_samples, SamplingClass,seed=42):
        """Initialize a basic aleatory workflow.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        SamplingClass : type
            Sampler class used to generate samples.
        seed : int, optional
            OpenTURNS random seed used before sample generation. The default is
            42.

        """
        super().__init__()
        self.num_samples=num_samples
        self.SamplingClass=SamplingClass
        self.seed=seed

    def get_states(self):
        """Generate sampled state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries generated from the sampled aleatory variables.
            Each dictionary maps variable names to one sampled value.

        """
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        sample_points, var_name_to_col = self.SamplingClass.sample(
            self.aleatory_var_to_dist, self.num_samples
        )
        states = []
        for row in range(sample_points.shape[0]):
            local_state = {}
            for name, col in var_name_to_col.items():
                local_state[name] = sample_points[row, col]
            states.append(local_state)
        return states

    def _to_dict(self):
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata and the
            serialized aleatory-variable distributions.

        """
        return {
            "type": self.__class__.__name__,
            "sampler": self.SamplingClass.__name__,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "aleatory_variables":{
                name:dist._to_dict()
                for name,dist in self.aleatory_var_to_dist.items()
            }
        }

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct a basic aleatory workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        BasicAleatory
            Reconstructed basic aleatory workflow.

        """
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        seed=workflow_dict['seed']
        analysis=cls(num_samples, sampler,seed)
        for name, distribution_data in workflow_dict["aleatory_variables"].items():
            dist = distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)
            analysis.add_aleatory_variable(name, dist)
        return analysis


class RegressionPCE(BasicAleatory):
    """Generate samples for a regression polynomial chaos expansion.

    This workflow uses a sampling class to generate aleatory states for fitting
    a polynomial chaos expansion by regression.

    Parameters
    ----------
    num_samples : int
        Number of aleatory samples to generate.
    SamplingClass : type
        Sampler class used to generate samples.
    poly_order : int
        Total polynomial order of the polynomial chaos basis.

    Notes
    -----
    The workflow checks that the number of samples is at least the number of
    basis terms required for the requested polynomial order and number of
    aleatory variables. When a nonzero holdout fraction is used, the basis
    feasibility check is performed against the effective post-split training
    sample count. Split-based holdout validation is most defensible for
    Monte Carlo sampling; for space-filling samplers such as Latin hypercube
    and Sobol, an independent validation design is usually more rigorous.

    """

    def __init__(self, num_samples, SamplingClass,poly_order,
                 seed=42,hold_out_fraction=0.0,hold_out_seed=None):
        """Initialize a regression PCE workflow.

        Parameters
        ----------
        num_samples : int
            Number of aleatory samples to generate.
        SamplingClass : type
            Sampler class used to generate samples.
        poly_order : int
            Total polynomial order of the polynomial chaos basis.
        seed : int, optional
            Random seed used to generate the regression sample set.
        hold_out_fraction : float, optional
            Fraction of completed samples reserved for surrogate validation.
            The remaining samples are used to fit the polynomial-chaos model.
            This split-based validation is most statistically natural for
            Monte Carlo sampling. For structured space-filling designs such as
            Latin hypercube or Sobol sampling, splitting one generated design
            can give optimistic validation metrics because the training and
            holdout sets are not independent designs.
        hold_out_seed : int or None, optional
            Random seed used to split training and holdout samples when
            ``hold_out_fraction`` is nonzero. Defaults to ``68`` in that case.

        """
        super().__init__(num_samples, SamplingClass,seed)
        self.poly_order = poly_order
        if hold_out_fraction<0.0 or hold_out_fraction>1.0:
            raise ValueError(f'Hold out fraction must be between zero and one')
        self.hold_out_fraction=hold_out_fraction
        if not hold_out_fraction==0.0:
            self.hold_out_seed=hold_out_seed if hold_out_seed is not None else 68


    def _check_analysis(self):
        # check order and number of aleatory vs. num samples
        # from XIU Polynomial expansion, dimension of space is C(order+number of random vars,number of random vars)
        # should oversample 2-3x
        """Validate the regression PCE sample count.

        Raises
        ------
        ValueError
            If ``num_samples`` is smaller than the number of required
            polynomial basis terms.

        Notes
        -----
        The number of basis terms is computed as
        ``comb(poly_order + num_aleatory, poly_order)`` using the effective
        training-sample count after any holdout split. A warning is logged
        when the effective training-sample count is less than three times the
        number of required basis terms.

        """
        num_aleatory = len(self.aleatory_var_to_dist)
        min_terms = comb(self.poly_order + num_aleatory, self.poly_order)
        num_samples = ceil(self.num_samples * (1 - self.hold_out_fraction))
        if min_terms > num_samples:
            raise ValueError(
                f"Insufficient terms for polynomial chaos expansion of degree {self.poly_order} for {num_aleatory} random variables"
            )
        elif min_terms * 3 > num_samples:
            log.warning(
                "Regression PCE is lightly sampled: "
                f"polynomial order={self.poly_order}, num_variables={num_aleatory}, "
                f"basis_size={min_terms}, num_samples={num_samples}. "
                "Consider 2-3x oversampling relative to the basis size."
            )

    def _to_dict(self):
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata, the
            serialized aleatory-variable distributions, and the polynomial
            order.

        """
        data=super()._to_dict()
        data['polynomial_order']=self.poly_order
        data['hold_out_fraction']=self.hold_out_fraction
        if not self.hold_out_fraction==0.0:
            data['hold_out_seed']=self.hold_out_seed
        return data

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct a regression PCE workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        RegressionPCE
            Reconstructed regression PCE workflow.

        """
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        poly_order = workflow_dict["polynomial_order"]
        seed=workflow_dict['seed']
        hold_out_fraction=workflow_dict['hold_out_fraction']
        hold_out_seed=workflow_dict.get('hold_out_seed', None)
        analysis=cls(num_samples=num_samples,
                     SamplingClass=sampler,
                     seed=seed,
                     poly_order=poly_order,
                     hold_out_fraction=hold_out_fraction,
                     hold_out_seed=hold_out_seed)
        for name, distribution_data in workflow_dict["aleatory_variables"].items():
            dist = distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)
            analysis.add_aleatory_variable(name, dist)
        return analysis


class GaussQuadraturePCE(AleatoryBaseClass):
    """Generate quadrature states for a polynomial chaos expansion.

    This workflow builds a tensor-product Gauss quadrature rule from the
    registered aleatory distributions. Each generated state includes sampled
    variable values and a quadrature ``weight``.

    Parameters
    ----------
    poly_order : int
        Polynomial order of the polynomial chaos expansion. The quadrature rule
        uses ``poly_order + 1`` points per aleatory variable.

    Notes
    -----
    This workflow is intended for low-dimensional aleatory analyses. Validation
    raises an error when more than five aleatory variables are configured.
    """

    def __init__(self, poly_order):
        """Initialize a Gauss quadrature PCE workflow.

        Parameters
        ----------
        poly_order : int
            Polynomial order of the polynomial chaos expansion.

        """
        super().__init__()
        self.poly_order = poly_order

    def _check_analysis(self):
        """Validate the quadrature workflow dimensionality.

        Raises
        ------
        ValueError
            If more than five aleatory variables are configured.

        """
        num_aleatory = len(self.aleatory_var_to_dist)
        if num_aleatory > 5:
            raise ValueError(
                "Quadrature based PCE may not be efficient for more than 5 aleatory variables"
            )

    def get_states(self):
        """Generate weighted quadrature state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries generated from the quadrature nodes. Each
            dictionary maps variable names to quadrature node values and
            includes a ``weight`` entry.

        Notes
        -----
        For ``k`` aleatory variables, this workflow generates a full tensor grid
        with ``(poly_order + 1) ** k`` states.

        """
        num_aleatory = len(self.aleatory_var_to_dist)

        if num_aleatory == 1:
            ((name, dist),) = self.aleatory_var_to_dist.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(self.aleatory_var_to_dist.items()):
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
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata, the
            serialized aleatory-variable distributions, and the polynomial
            order.

        """
        return {"type": self.__class__.__name__,
                "aleatory_variables":{
                name:dist._to_dict()
                for name,dist in self.aleatory_var_to_dist.items()
                },
                "polynomial_order": self.poly_order}

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct a Gauss quadrature PCE workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        GaussQuadraturePCE
            Reconstructed Gauss quadrature PCE workflow.

        """
        poly_order = workflow_dict["polynomial_order"]
        analysis=cls(poly_order)
        for name, distribution_data in workflow_dict["aleatory_variables"].items():
            dist = distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)
            analysis.add_aleatory_variable(name, dist)
        return analysis


class SobolSensitivity(BasicAleatory):
    """Generate samples for Sobol sensitivity analysis.

    This workflow creates the structured sample design required to estimate
    Sobol sensitivity indices for the configured aleatory variables.

    Parameters
    ----------
    num_samples : int
        Base sample size used by the Sobol experiment.
    SamplingClass : type
        Sampler class used by OpenTURNS for the Sobol experiment.
    computeSecondOrder : bool, optional
        Whether to generate samples for second-order Sobol indices. The default
        is False.

    Notes
    -----
    The total number of generated states depends on the number of aleatory
    variables and whether second-order indices are requested.

    """

    def __init__(self, num_samples, SamplingClass,seed=42, computeSecondOrder: bool = False):
        """Initialize a Sobol sensitivity workflow.

        Parameters
        ----------
        num_samples : int
            Base sample size used by the Sobol experiment.
        SamplingClass : type
            Sampler class used by OpenTURNS for the Sobol experiment.
        computeSecondOrder : bool, optional
            Whether to generate samples for second-order Sobol indices. The
            default is False.

        """
        super().__init__(num_samples, SamplingClass,seed)
        self.computeSecondOrder = computeSecondOrder

    def get_states(self):
        """Generate Sobol experiment state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries generated from the Sobol experiment design. Each
            dictionary maps variable names to sampled values.

        Raises
        ------
        ValueError
            If OpenTURNS returns an unexpected number of experiment rows.

        Notes
        -----
        If second-order indices are not requested, the experiment generates
        ``num_samples * (2 + k)`` states for ``k`` aleatory variables. With
        second-order indices and ``k != 2``, it generates
        ``num_samples * (2 + 2 * k)`` states.

        """
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        ot.ResourceMap.Set(
            "SobolIndicesExperiment-SamplingMethod",
            self.SamplingClass._OT_ResourceMapName,
        )
        if len(self.aleatory_var_to_dist) == 1:
            ((name, dist),) = self.aleatory_var_to_dist.items()
            var_name_to_col = {name: 0}
            dist = dist._OT_Dist()
        else:
            joint = []
            var_name_to_col = {}
            for i, (name, dist) in enumerate(self.aleatory_var_to_dist.items()):
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
        if (not self.computeSecondOrder) or len(self.aleatory_var_to_dist) == 2:
            if results.shape[0] != self.num_samples * (
                2 + len(self.aleatory_var_to_dist)
            ):
                raise ValueError(f"Number of rows is unexpected {results.shape[0]}")
        else:
            if results.shape[0] != self.num_samples * (
                2 + 2 * len(self.aleatory_var_to_dist)
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
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata, the
            serialized aleatory-variable distributions, and the second-order
            Sobol setting.

        """
        data=super()._to_dict()
        data['computeSecondOrder']=self.computeSecondOrder
        return data

    @classmethod
    def _from_dict(cls, workflow_dict):
        """Reconstruct a Sobol sensitivity workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        SobolSensitivity
            Reconstructed Sobol sensitivity workflow.

        """
        sampler = sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples = workflow_dict["num_samples"]
        second_order_sobol = workflow_dict["computeSecondOrder"]
        seed=workflow_dict['seed']
        analysis = cls(num_samples, sampler,seed, second_order_sobol)
        for name, distribution_data in workflow_dict["aleatory_variables"].items():
            dist = distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)
            analysis.add_aleatory_variable(name, dist)

        return analysis




#--------------------Epistemic Suite-------------------------------------
class EpistemicBaseClass:
    def __init__(self):
      self.epistemic_var_to_interval={}

    def add_epistemic_variable(self,name,interval):
        """Add an epistemic variable to the analysis.
        Parameters
        ----------
        name : str
            Name of the UAM4X state variable. Nested state variables can be
            referenced with dotted paths.
        interval : ndarray
            One-dimensional NumPy array with shape ``(2,)`` containing the
            lower and upper bounds as ``[low, high]``.
        Raises
        ------
        ValueError
            If ``interval`` is not a NumPy array, does not have shape ``(2,)``,
            contains nonnumeric or nonfinite entries, or has ``low > high``.
        """
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

        self.epistemic_var_to_interval[name]=interval

class BasicEpistemic(EpistemicBaseClass):
    """Generate epistemic states with a sampling class.

    This workflow treats each epistemic interval as a uniform distribution over
    its bounds, samples those distributions, and returns one UAM4X state
    dictionary per sample.

    Parameters
    ----------
    num_samples : int
        Number of epistemic samples to generate.
    SamplingClass : type
        Sampler class used to generate samples.

    """

    def __init__(self,num_samples,SamplingClass,seed=42):
        """Initialize a basic epistemic workflow.

        Parameters
        ----------
        num_samples : int
            Number of epistemic samples to generate.
        SamplingClass : type
            Sampler class used to generate samples.

        """
        super().__init__()
        self.num_samples=num_samples
        self.SamplingClass=SamplingClass
        self.seed=seed

    def get_states(self):
        """Generate sampled epistemic state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries generated from sampled epistemic intervals. Each
            dictionary maps variable names to one sampled value.

        Notes
        -----
        Each epistemic interval is sampled as a uniform distribution over its
        lower and upper bounds.

        """
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        epistemic_to_vals=self.epistemic_var_to_interval

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
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata and the
            serialized epistemic intervals.

        """
        return {
            "type":self.__class__.__name__,
            "sampler": self.SamplingClass.__name__,
            "num_samples":self.num_samples,
            "seed":self.seed,
            "epistemic_variables":{name:interval
                                    for name,interval in self.epistemic_var_to_interval.items()}
        }

    @classmethod
    def _from_dict(cls,workflow_dict):
        """Reconstruct a basic epistemic workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        BasicEpistemic
            Reconstructed basic epistemic workflow.

        """
        sampler=sampler_name_to_class_registry[workflow_dict["sampler"]]
        num_samples=workflow_dict['num_samples']
        seed=workflow_dict['seed']
        epistemic_analysis=cls(num_samples,sampler,seed=seed)
        for name, interval in workflow_dict['epistemic_variables'].items():
            epistemic_analysis.add_epistemic_variable(name,np.array(interval))
        return epistemic_analysis

#--------------------Mixed Analysis Suite-------------------------------------
class MixedBaseClass:
    def __init__(self):
        self.aleatory_var_to_dist={}
        self.epistemic_var_to_interval={}

    def add_aleatory_variable(self,name,PDFObject):
        """Add an aleatory variable to the mixed analysis.

        Parameters
        ----------
        name : str
            Name of the UAM4X state variable to sample. Nested state variables
            can be referenced with dotted paths.
        PDFObject : object
            Distribution object associated with ``name``. The object must
            provide ``_OT_Dist`` and ``_to_dict`` methods.

        """
        self.aleatory_var_to_dist[name]=PDFObject

    def add_epistemic_variable(self,name,interval):
        """Add an epistemic variable to the mixed analysis.

        Parameters
        ----------
        name : str
            Name of the UAM4X state variable. Nested state variables can be
            referenced with dotted paths.
        interval : ndarray
            One-dimensional NumPy array with shape ``(2,)`` containing the
            lower and upper bounds as ``[low, high]``.

        Raises
        ------
        ValueError
            If ``interval`` is not a NumPy array, does not have shape ``(2,)``,
            contains nonnumeric or nonfinite entries, or has ``low > high``.

        """
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
        self.epistemic_var_to_interval[name]=interval

class BasicSOP(MixedBaseClass):
    """Generate nested samples for second-order probability analysis.

    This workflow samples epistemic variables in an outer loop and aleatory
    variables in an inner loop. Epistemic variables are interpreted as
    uncertain parameters of aleatory distributions.

    Parameters
    ----------
    num_epistemic_samples : int
        Number of outer-loop epistemic samples to generate.
    EpistemicSamplingClass : type
        Sampler class used to sample epistemic intervals.
    num_aleatory_samples : int
        Number of inner-loop aleatory samples to generate for each epistemic
        sample.
    AleatorySamplingClass : type
        Sampler class used to sample aleatory distributions.

    Notes
    -----
    Epistemic variable names are expected to identify distribution parameters
    using dotted names, where the final component is the parameter name and the
    preceding components identify the aleatory variable.

    """

    def __init__(self,num_epistemic_samples,EpistemicSamplingClass,
               num_aleatory_samples,AleatorySamplingClass,seed=42):
        """Initialize a second-order probability workflow.

        Parameters
        ----------
        num_epistemic_samples : int
            Number of outer-loop epistemic samples to generate.
        EpistemicSamplingClass : type
            Sampler class used to sample epistemic intervals.
        num_aleatory_samples : int
            Number of inner-loop aleatory samples to generate for each
            epistemic sample.
        AleatorySamplingClass : type
            Sampler class used to sample aleatory distributions.

        """
        super().__init__()
        self.num_epistemic_samples=num_epistemic_samples
        self.num_aleatory_samples=num_aleatory_samples
        self.EpistemicSamplingClass=EpistemicSamplingClass
        self.AleatorySamplingClass=AleatorySamplingClass
        self.seed=seed

    def get_states(self):
        """Generate nested epistemic-aleatory state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries for all epistemic and aleatory sample
            combinations.

        Notes
        -----
        For each epistemic sample, the corresponding aleatory distribution
        parameters are updated before generating the inner-loop aleatory
        samples. The total number of returned states is
        ``num_epistemic_samples * num_aleatory_samples``.

        """
        # SOPIVP uses epistemic entries as distribution-parameter uncertainty.
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        epistemic_to_vals = self.epistemic_var_to_interval
        aleatory_to_distribution = self.aleatory_var_to_dist

        epistemic_to_distribution={}
        for name,interval in epistemic_to_vals.items():
            low=interval[0]
            high=interval[1]
            epistemic_to_distribution[name]=UniformDistribution(low,high)

        epistemic_samples, epistemic_var_name_to_col =self.EpistemicSamplingClass.sample(
            epistemic_to_distribution,self.num_epistemic_samples
            )

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

        all_vars_to_col={}
        for name,col in epistemic_var_name_to_col.items():
            all_vars_to_col[name]=col

        for name,col in aleatory_var_name_to_col.items():
            all_vars_to_col[name]=col+(len(epistemic_var_name_to_col))

        epistemic_states=np.repeat(epistemic_samples,
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
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata, serialized
            aleatory distributions, serialized epistemic intervals, and sample
            counts.

        """
        return {
            "type":self.__class__.__name__,
            "epistemic_sampler":self.EpistemicSamplingClass.__name__,
            "aleatory_sampler":self.AleatorySamplingClass.__name__,
            "num_epistemic_samples":self.num_epistemic_samples,
            "num_aleatory_samples":self.num_aleatory_samples,
            "seed":self.seed,
            "aleatory_variables":{name:dist._to_dict()
                                  for name,dist in self.aleatory_var_to_dist.items()},

            "epistemic_variables":{name:interval
                                   for name,interval in self.epistemic_var_to_interval.items()}

        }

    @classmethod
    def _from_dict(cls,data):
        """Reconstruct a second-order probability workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        BasicSOP
            Reconstructed second-order probability workflow.

        """
        e_sampler=sampler_name_to_class_registry[data["epistemic_sampler"]]
        a_sampler=sampler_name_to_class_registry[data["aleatory_sampler"]]
        ne_samps=data["num_epistemic_samples"]
        na_samps=data["num_aleatory_samples"]
        seed=data['seed']
        mixed_analysis=cls(num_epistemic_samples=ne_samps,
                           EpistemicSamplingClass=e_sampler,
                           num_aleatory_samples=na_samps,
                           AleatorySamplingClass=a_sampler,
                           seed=seed)

        for name,distribution_data in data["aleatory_variables"].items():
            dist=distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)

            mixed_analysis.add_aleatory_variable(name,dist)

        for name,interval_data in data["epistemic_variables"].items():
            mixed_analysis.add_epistemic_variable(name,np.array(interval_data))

        return mixed_analysis



class BasicMixedIVP(MixedBaseClass):
    """Generate nested samples for mixed interval-probability analysis.

    This workflow samples epistemic variables in an outer loop and aleatory
    variables in an inner loop. Unlike :class:`BasicSOP`, epistemic samples are
    included directly in each state and do not modify aleatory distribution
    parameters.

    Parameters
    ----------
    num_epistemic_samples : int
        Number of outer-loop epistemic samples to generate.
    EpistemicSamplingClass : type
        Sampler class used to sample epistemic intervals.
    num_aleatory_samples : int
        Number of inner-loop aleatory samples to generate for each epistemic
        sample.
    AleatorySamplingClass : type
        Sampler class used to sample aleatory distributions.

    """

    def __init__(self,num_epistemic_samples,EpistemicSamplingClass,
               num_aleatory_samples,AleatorySamplingClass,seed=42):
        """Initialize a mixed interval-probability workflow.

        Parameters
        ----------
        num_epistemic_samples : int
            Number of outer-loop epistemic samples to generate.
        EpistemicSamplingClass : type
            Sampler class used to sample epistemic intervals.
        num_aleatory_samples : int
            Number of inner-loop aleatory samples to generate for each
            epistemic sample.
        AleatorySamplingClass : type
            Sampler class used to sample aleatory distributions.

        """
        super().__init__()
        self.num_epistemic_samples=num_epistemic_samples
        self.num_aleatory_samples=num_aleatory_samples
        self.EpistemicSamplingClass=EpistemicSamplingClass
        self.AleatorySamplingClass=AleatorySamplingClass
        self.seed=seed

    def get_states(self):
        """Generate nested mixed state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries for all epistemic and aleatory sample
            combinations.

        Notes
        -----
        The total number of returned states is
        ``num_epistemic_samples * num_aleatory_samples``.

        """
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        #generate the epistemic to distribution here
        aleatory_to_distribution=self.aleatory_var_to_dist
        epistemic_to_vals=self.epistemic_var_to_interval

        epistemic_to_distribution={}
        for name,interval in epistemic_to_vals.items():
            low=interval[0]
            high=interval[1]
            epistemic_to_distribution[name]=UniformDistribution(low,high)

        epistemic_samples, epistemic_var_name_to_col =self.EpistemicSamplingClass.sample(
            epistemic_to_distribution,self.num_epistemic_samples
            )

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

        all_vars_to_col={}
        for name,col in epistemic_var_name_to_col.items():
            all_vars_to_col[name]=col

        for name,col in aleatory_var_name_to_col.items():
            all_vars_to_col[name]=col+(len(epistemic_var_name_to_col))

        epistemic_states=np.repeat(epistemic_samples,
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
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata, serialized
            aleatory distributions, serialized epistemic intervals, and sample
            counts.

        """
        return {
            "type":self.__class__.__name__,
            "epistemic_sampler":self.EpistemicSamplingClass.__name__,
            "aleatory_sampler":self.AleatorySamplingClass.__name__,
            "num_epistemic_samples":self.num_epistemic_samples,
            "num_aleatory_samples":self.num_aleatory_samples,
            "seed":self.seed,
            "aleatory_variables":{name:dist._to_dict()
                                  for name,dist in self.aleatory_var_to_dist.items()},

            "epistemic_variables":{name:interval
                                   for name,interval in self.epistemic_var_to_interval.items()}
        }

    @classmethod
    def _from_dict(cls,data):
        """Reconstruct a mixed interval-probability workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        BasicMixedIVP
            Reconstructed mixed interval-probability workflow.

        """
        e_sampler=sampler_name_to_class_registry[data["epistemic_sampler"]]
        a_sampler=sampler_name_to_class_registry[data["aleatory_sampler"]]
        ne_samps=data["num_epistemic_samples"]
        na_samps=data["num_aleatory_samples"]
        seed=data["seed"]
        mixed_analysis=cls(num_epistemic_samples=ne_samps,
                           EpistemicSamplingClass=e_sampler,
                           num_aleatory_samples=na_samps,
                           AleatorySamplingClass=a_sampler,
                           seed=seed)

        for name,distribution_data in data["aleatory_variables"].items():
            dist=distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)

            mixed_analysis.add_aleatory_variable(name,dist)

        for name,interval_data in data["epistemic_variables"].items():
            mixed_analysis.add_epistemic_variable(name,np.array(interval_data))

        return mixed_analysis


class MixedRegressionPCE(MixedBaseClass):
    """Generate samples for a mixed regression polynomial chaos expansion.

    This workflow samples both epistemic intervals and aleatory distributions
    together as inputs for fitting a polynomial chaos expansion by regression.

    Parameters
    ----------
    SamplingClass : type
        Sampler class used to generate samples.
    num_samples : int
        Number of mixed samples to generate.
    poly_order : int
        Total polynomial order of the polynomial chaos basis.

    Notes
    -----
    Epistemic intervals are sampled as uniform distributions over their lower
    and upper bounds. The workflow checks that the number of samples is at
    least the number of basis terms required for the requested polynomial order
    and total number of mixed variables. When a nonzero holdout fraction is
    used, the basis feasibility check is performed against the effective
    post-split training sample count. Split-based holdout validation is most
    defensible for Monte Carlo sampling; for space-filling samplers such as
    Latin hypercube and Sobol, an independent validation design is usually
    more rigorous.

    """

    def __init__(self,SamplingClass,num_samples,
                 poly_order,seed=42,
                 hold_out_fraction=0.0,hold_out_seed=None):
        """Initialize a mixed regression PCE workflow.

        Parameters
        ----------
        SamplingClass : type
            Sampler class used to generate samples.
        num_samples : int
            Number of mixed samples to generate.
        poly_order : int
            Total polynomial order of the polynomial chaos basis.
        seed : int, optional
            Random seed used to generate the mixed sample set.
        hold_out_fraction : float, optional
            Fraction of completed samples reserved for surrogate validation.
            The remaining samples are used to fit the polynomial-chaos model.
            This split-based validation is most statistically natural for
            Monte Carlo sampling. For structured space-filling designs such as
            Latin hypercube or Sobol sampling, splitting one generated design
            can give optimistic validation metrics because the training and
            holdout sets are not independent designs.
        hold_out_seed : int or None, optional
            Random seed used to split training and holdout samples when
            ``hold_out_fraction`` is nonzero. Defaults to ``68`` in that case.

        """
        super().__init__()
        self.SamplingClass=SamplingClass
        self.num_samples=num_samples
        self.poly_order = poly_order
        self.seed=seed
        if hold_out_fraction<0.0 or hold_out_fraction>1.0:
            raise ValueError(f'Hold out fraction must be between zero and one')
        self.hold_out_fraction=hold_out_fraction
        if not hold_out_fraction==0.0:
            self.hold_out_seed=hold_out_seed if hold_out_seed is not None else 68

    def get_states(self):
        """Generate sampled mixed state dictionaries.

        Returns
        -------
        list of dict
            State dictionaries generated from sampled epistemic and aleatory
            variables. Each dictionary maps variable names to one sampled value.

        Notes
        -----
        Epistemic intervals are sampled as uniform distributions over their
        lower and upper bounds.

        """
        ot.RandomGenerator.Reset()
        ot.RandomGenerator.SetSeed(self.seed)
        #generate the epistemic to distribution here
        aleatory_to_distribution=self.aleatory_var_to_dist
        epistemic_to_vals=self.epistemic_var_to_interval

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

    def _check_analysis(self):
        """Validate the mixed regression PCE sample count.

        Raises
        ------
        ValueError
            If ``num_samples`` is smaller than the number of required
            polynomial basis terms.

        Notes
        -----
        The number of basis terms is computed as
        ``comb(poly_order + num_vars, poly_order)``, where ``num_vars`` is the
        total number of aleatory and epistemic variables, using the effective
        training-sample count after any holdout split. A warning is logged
        when the effective training-sample count is less than three times the
        number of required basis terms.

        """
        num_vars = len(self.aleatory_var_to_dist) + len(
            self.epistemic_var_to_interval
        )
        min_terms = comb(self.poly_order + num_vars, self.poly_order)
        num_samples=ceil(self.num_samples*(1-self.hold_out_fraction))
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
        """Serialize the workflow to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing workflow metadata, serialized
            aleatory distributions, serialized epistemic intervals, and the
            polynomial order.

        """
        data={
            "type":self.__class__.__name__,
            "sampler":self.SamplingClass.__name__,
            "num_samples":self.num_samples,
            'seed':self.seed,
            "aleatory_variables":{name:dist._to_dict()
                                  for name,dist in self.aleatory_var_to_dist.items()},

            "epistemic_variables":{name:interval
                                   for name,interval in self.epistemic_var_to_interval.items()},
            "polynomial_order":self.poly_order,
            "hold_out_fraction":self.hold_out_fraction,

        }
        if not self.hold_out_fraction==0.0:
            data["hold_out_seed"]=self.hold_out_seed
        return data

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct a mixed regression PCE workflow from a dictionary.

        Parameters
        ----------
        workflow_dict : dict
            Serialized workflow dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        MixedRegressionPCE
            Reconstructed mixed regression PCE workflow.

        """
        sampler = sampler_name_to_class_registry[data["sampler"]]
        num_samples = data["num_samples"]
        poly_order = data["polynomial_order"]
        seed=data['seed']
        hold_out_fraction=data['hold_out_fraction']
        hold_out_seed=data.get('hold_out_seed', None)
        mixed_analysis=cls(SamplingClass=sampler,
                           num_samples=num_samples,
                             poly_order=poly_order,
                             seed=seed,
                             hold_out_fraction=hold_out_fraction,
                             hold_out_seed=hold_out_seed)
        for name,distribution_data in data["aleatory_variables"].items():
            dist=distribution_name_to_class_registry[
                distribution_data["name"]
            ]._from_dict(distribution_data)

            mixed_analysis.add_aleatory_variable(name,dist)

        for name,interval_data in data["epistemic_variables"].items():
            mixed_analysis.add_epistemic_variable(name,np.array(interval_data))

        return mixed_analysis


#-------------------------------CSV Stuff
class CSVPoints:
    """Store the CSV file backing a point-based uncertainty workflow.

    Parameters
    ----------
    csv_file : str or path-like
        Path to the CSV file containing one column per uncertainty variable.

    """
    def __init__(self,csv_file):
        self.csv_file=csv_file
        self.poly_order=None
        self.computeSecondOrder=None

    def add_polynomial_order(self,order):
        if not is_integer(order):
            raise ValueError(
                "Polynomial order must be an integer value."
            )
        if order < 0:
            raise ValueError(
                "Polynomial order must be nonnegative."
            )
        self.poly_order=order

    def set_compute_second_order(self, computeSecondOrder: bool):
        if not is_bool(computeSecondOrder):
            raise ValueError(
                "computeSecondOrder must be a boolean value."
            )
        self.computeSecondOrder = computeSecondOrder

    def _to_dict(self):
        """Serialize the CSV-backed workflow metadata.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the workflow type and CSV
            file path.

        """
        data= {
            "type":self.__class__.__name__,
            "csv_path":self.csv_file,
        }
        if self.poly_order is not None:
            data["polynomial_order"]=self.poly_order
        if self.computeSecondOrder is not None:
            data["computeSecondOrder"]=self.computeSecondOrder
        return data


class EpistemicCSVPoints(CSVPoints,EpistemicBaseClass):
    """Load epistemic states directly from a CSV file."""
    def __init__(self,csv_file):
        EpistemicBaseClass.__init__(self)
        CSVPoints.__init__(self,csv_file)

    def get_states(self):
        """Load epistemic state dictionaries from the configured CSV file.

        Returns
        -------
        list of dict
            State dictionaries ordered by the registered epistemic-variable
            names.

        """
        col_order=list(self.epistemic_var_to_interval.keys())
        return gen_csv_states(csv_path=self.csv_file,col_order=col_order)

    def _to_dict(self):
        data=super()._to_dict()
        data['epistemic_variables']={name:interval
                                    for name,interval in self.epistemic_var_to_interval.items()}
        return data

    @classmethod
    def _from_dict(cls,data):
        csv_file=data['csv_path']
        csv_analysis=cls(csv_file)
        for name,interval in data['epistemic_variables'].items():
            csv_analysis.add_epistemic_variable(name,np.array(interval))
        if "polynomial_order" in data:
            csv_analysis.add_polynomial_order(data["polynomial_order"])
        if "computeSecondOrder" in data:
            csv_analysis.set_compute_second_order(data["computeSecondOrder"])
        return csv_analysis


class AleatoryCSVPoints(CSVPoints,AleatoryBaseClass):
    """Load aleatory states directly from a CSV file."""

    def __init__(self, csv_file):
        CSVPoints.__init__(self,csv_file)
        AleatoryBaseClass.__init__(self)

    def get_states(self):
        """Load aleatory state dictionaries from the configured CSV file.

        Returns
        -------
        list of dict
            State dictionaries ordered by the registered aleatory-variable
            names.

        """
        col_order=list(self.aleatory_var_to_dist.keys())
        return gen_csv_states(csv_path=self.csv_file,col_order=col_order)

    def _to_dict(self):
        data=super()._to_dict()
        data['aleatory_variables']={name:dist._to_dict()
                                    for name,dist in self.aleatory_var_to_dist.items()
                                    }
        return data

    @classmethod
    def _from_dict(cls,data):
        csv_file=data['csv_path']
        csv_analysis=cls(csv_file)
        for name, distribution_data in data['aleatory_variables'].items():
            distribution=distribution_name_to_class_registry[
                distribution_data['name']
            ]._from_dict(distribution_data)
            csv_analysis.add_aleatory_variable(name,distribution)
        if "polynomial_order" in data:
            csv_analysis.add_polynomial_order(data["polynomial_order"])
        if "computeSecondOrder" in data:
            csv_analysis.set_compute_second_order(data["computeSecondOrder"])
        return csv_analysis

class MixedCSVPoints(CSVPoints,MixedBaseClass):
    """Load mixed aleatory-epistemic states directly from a CSV file."""
    def __init__(self,csv_file):
        CSVPoints.__init__(self,csv_file)
        MixedBaseClass.__init__(self)

    def get_states(self):
        """Load mixed state dictionaries from the configured CSV file.

        Returns
        -------
        list of dict
            State dictionaries ordered by epistemic-variable names followed by
            aleatory-variable names.

        """
        col_order=list(list(self.epistemic_var_to_interval.keys())+list(self.aleatory_var_to_dist.keys()))
        return gen_csv_states(csv_path=self.csv_file,col_order=col_order)

    def _to_dict(self):
        data=super()._to_dict()
        data['aleatory_variables']={name:dist._to_dict()
                                    for name,dist in
                                    self.aleatory_var_to_dist.items()
                                    }
        data['epistemic_variables']={name:interval
                                    for name,interval in
                                    self.epistemic_var_to_interval.items()
                                    }
        return data

    @classmethod
    def _from_dict(cls,data):
        csv_file=data['csv_path']
        csv_analysis=cls(csv_file)
        for name, distribution_data in data['aleatory_variables'].items():
            distribution=distribution_name_to_class_registry[
                distribution_data['name']
            ]._from_dict(distribution_data)
            csv_analysis.add_aleatory_variable(name,distribution)

        for name,interval_data in data["epistemic_variables"].items():
            csv_analysis.add_epistemic_variable(name,np.array(interval_data))
        if "polynomial_order" in data:
            csv_analysis.add_polynomial_order(data["polynomial_order"])
        if "computeSecondOrder" in data:
            csv_analysis.set_compute_second_order(data["computeSecondOrder"])
        return csv_analysis


#--------------------Samplers-------------------------------------
class BaseSampler:
    """Define the sampler interface for uncertainty workflows.

    Sampler classes generate numeric samples from a mapping of variable names
    to distribution objects and return both the sample array and a
    variable-to-column mapping.

    """

    @staticmethod
    def sample():
        """Generate samples from configured distributions.

        Notes
        -----
        Subclasses implement this method.

        """
        pass


class MonteCarloSampler(BaseSampler):
    """Generate independent Monte Carlo samples."""

    _OT_ResourceMapName = "MonteCarlo"

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Sample distributions with an OpenTURNS Monte Carlo experiment.

        Parameters
        ----------
        aleatory_to_dist : dict
            Mapping from variable names to distribution objects.
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : ndarray
            Array of sampled values with one row per sample and one column per
            variable.
        var_name_to_col : dict
            Mapping from variable names to column indices in ``samples``.

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


class LatinHypercubeSampler(BaseSampler):
    """Generate Latin hypercube samples."""

    _OT_ResourceMapName = "LHS"

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Sample distributions with an OpenTURNS Latin hypercube experiment.

        Parameters
        ----------
        aleatory_to_dist : dict
            Mapping from variable names to distribution objects.
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : ndarray
            Array of sampled values with one row per sample and one column per
            variable.
        var_name_to_col : dict
            Mapping from variable names to column indices in ``samples``.

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


class SobolSampler(BaseSampler):

    # Note this use here is for SobolExperiment, where the
    # Quasi monte carlo uses the Sobol sampler
    """Generate low-discrepancy Sobol sequence samples."""
    _OT_ResourceMapName = "QMC"

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Sample distributions with an OpenTURNS Sobol sequence experiment.

        Parameters
        ----------
        aleatory_to_dist : dict
            Mapping from variable names to distribution objects.
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : ndarray
            Array of sampled values with one row per sample and one column per
            variable.
        var_name_to_col : dict
            Mapping from variable names to column indices in ``samples``.

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


class HaltonSampler(BaseSampler):
    """Generate low-discrepancy Halton sequence samples."""

    @staticmethod
    def sample(aleatory_to_dist, num_samples):
        """Sample distributions with an OpenTURNS Halton sequence experiment.

        Parameters
        ----------
        aleatory_to_dist : dict
            Mapping from variable names to distribution objects.
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : ndarray
            Array of sampled values with one row per sample and one column per
            variable.
        var_name_to_col : dict
            Mapping from variable names to column indices in ``samples``.

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
class NormalDistribution:
    """Represent a normal probability distribution.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.

    """

    def __init__(self, mean, std):
        """Initialize a normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the normal distribution.
        std : float
            Standard deviation of the normal distribution.

        """
        self.mean = mean
        self.std = std

    def _OT_Dist(self):
        """Create the equivalent OpenTURNS distribution.

        Returns
        -------
        openturns.Normal
            OpenTURNS normal distribution with the configured mean and standard
            deviation.

        """
        return ot.Normal(self.mean, self.std)

    def _to_dict(self):
        """Serialize the distribution to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the distribution name, mean,
            and standard deviation.

        """
        return {"name": self.__class__.__name__, "mean": self.mean, "std": self.std}

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct a normal distribution from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized normal distribution dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        NormalDistribution
            Reconstructed normal distribution.

        """
        return cls(data["mean"], data["std"])


class UniformDistribution:
    """Represent a uniform probability distribution.

    Parameters
    ----------
    low : float
        Lower bound of the uniform distribution.
    high : float
        Upper bound of the uniform distribution.

    """

    def __init__(self, low, high):
        """Initialize a uniform distribution.

        Parameters
        ----------
        low : float
            Lower bound of the uniform distribution.
        high : float
            Upper bound of the uniform distribution.

        """
        self.low = low
        self.high = high

    def _OT_Dist(self):
        """Create the equivalent OpenTURNS distribution.

        Returns
        -------
        openturns.Uniform
            OpenTURNS uniform distribution with the configured lower and upper
            bounds.

        """
        return ot.Uniform(self.low, self.high)

    def _to_dict(self):
        """Serialize the distribution to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the distribution name and
            bounds.

        """
        return {"name": self.__class__.__name__, "low": self.low, "high": self.high}

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct a uniform distribution from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized uniform distribution dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        UniformDistribution
            Reconstructed uniform distribution.

        """
        return cls(data["low"], data["high"])


class LogNormalDistribution:
    """Represent a shifted log-normal probability distribution.

    Parameters
    ----------
    mu_log : float
        Mean parameter of the underlying normal distribution in log space.
    sigma_log : float
        Standard deviation parameter of the underlying normal distribution in
        log space.
    gamma : float, optional
        Shift parameter of the log-normal distribution. The default is 0.0.

    """

    def __init__(self, mu_log, sigma_log, gamma=0.0):
        """Initialize a shifted log-normal distribution.

        Parameters
        ----------
        mu_log : float
            Mean parameter of the underlying normal distribution in log space.
        sigma_log : float
            Standard deviation parameter of the underlying normal distribution
            in log space.
        gamma : float, optional
            Shift parameter of the log-normal distribution. The default is 0.0.

        """
        self.mu_log = mu_log
        self.sigma_log = sigma_log
        self.gamma = gamma

    def _OT_Dist(self):
        """Create the equivalent OpenTURNS distribution.

        Returns
        -------
        openturns.LogNormal
            OpenTURNS log-normal distribution with the configured log-space
            parameters and shift.

        """
        return ot.LogNormal(self.mu_log, self.sigma_log, self.gamma)

    def _to_dict(self):
        """Serialize the distribution to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the distribution name,
            log-space parameters, and shift.

        """
        return {
            "name": self.__class__.__name__,
            "mu": self.mu_log,
            "sigma": self.sigma_log,
            "gamma": self.gamma,
        }

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct a log-normal distribution from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized log-normal distribution dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        LogNormalDistribution
            Reconstructed log-normal distribution.

        """
        return cls(data["mu"], data["sigma"], data.get("gamma", 0.0))

# Global registry
workflow_name_to_class_registry = {BasicAleatory.__name__: BasicAleatory,
                                   RegressionPCE.__name__: RegressionPCE,
                                   GaussQuadraturePCE.__name__: GaussQuadraturePCE,
                                   SobolSensitivity.__name__: SobolSensitivity,
                                   BasicEpistemic.__name__: BasicEpistemic,
                                   BasicMixedIVP.__name__: BasicMixedIVP,
                                   BasicSOP.__name__: BasicSOP,
                                   MixedRegressionPCE.__name__: MixedRegressionPCE,
                                   AleatoryCSVPoints.__name__:AleatoryCSVPoints,
                                   EpistemicCSVPoints.__name__:EpistemicCSVPoints,
                                   MixedCSVPoints.__name__:MixedCSVPoints
                                   }
sampler_name_to_class_registry = {LatinHypercubeSampler.__name__: LatinHypercubeSampler,
                                  SobolSampler.__name__: SobolSampler,
                                  HaltonSampler.__name__: HaltonSampler,
                                  MonteCarloSampler.__name__: MonteCarloSampler}
distribution_name_to_class_registry = {NormalDistribution.__name__: NormalDistribution,
                                      UniformDistribution.__name__: UniformDistribution,
                                      LogNormalDistribution.__name__: LogNormalDistribution}

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
    ivp_debug = BasicMixedIVP(
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
