import copy
from uam4x.utils import hash_state, convert_numpy_to_python
import matplotlib.pyplot as plt
import numpy as np
from uam4x.utils import split_dotted_path_and_indices


class Results:
    """Store analysis result sets for a study.

    A ``Results`` object maps analysis names to :class:`Tradestudy` result
    containers and provides access to all stored study results.

    Parameters
    ----------
    study : Study, optional
        Parent study associated with these results.

    """

    def __init__(self, study=None):
        """Initialize a container for study results.

        Parameters
        ----------
        study : Study, optional
            Parent study associated with these results.

        """
        self.tradestudies = {}
        self.study = study

    def get_tradestudy(self, name):
        """Return one stored analysis result container by name.

        Parameters
        ----------
        name : str
            Analysis name.

        Returns
        -------
        Tradestudy or None
            Stored result container for ``name``, or None if it is not present.

        """
        if name in self.tradestudies:
            return self.tradestudies[name]
        else:
            return None

    def add_tradestudy(self, name, Trade):
        """Attach an analysis result container to this results object.

        Parameters
        ----------
        name : str
            Analysis name.
        Trade : Tradestudy
            Result container to attach.

        Returns
        -------
        None

        Notes
        -----
        The attached :class:`Tradestudy` receives a back-reference to this
        :class:`Results` object.

        """
        Trade._results_owner = self
        self.tradestudies[name] = Trade

    def has_results(self):
        """Return whether any analysis results are stored.

        Returns
        -------
        bool
            True if at least one analysis result container is stored, otherwise
            False.

        """
        if self.tradestudies:
            return True
        else:
            return False

    def get_all_tradestudies(self):
        """Return names of all stored analysis result containers.

        Returns
        -------
        list of str
            Analysis names with stored results.

        """
        tradestudies = list(self.tradestudies.keys())
        return tradestudies

    def _to_dict(self):
        """Serialize the results object to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary mapping analysis names to serialized
            :class:`Tradestudy` data.

        """
        tradename_to_tradestudy = {}
        for name, tradestudy in self.tradestudies.items():
            tradename_to_tradestudy[name] = tradestudy._to_dict()
        return tradename_to_tradestudy

    @classmethod
    def _from_dict(cls, tradename_to_results, study):
        """Reconstruct a results object from a dictionary.

        Parameters
        ----------
        tradename_to_results : dict
            Mapping from analysis names to serialized :class:`Tradestudy` data.
        study : Study
            Parent study associated with the reconstructed results.

        Returns
        -------
        Results
            Reconstructed results object with attached study reference.

        """
        res = cls(study)
        for trade_name, results in tradename_to_results.items():
            ts = Tradestudy._from_dict(trade_name, results)
            res.add_tradestudy(trade_name, ts)
        return res


class Tradestudy:
    """Store all result points for one named analysis.

    A ``Tradestudy`` groups the individual :class:`Result` objects produced by
    one analysis and provides access to variables that change across those
    result points.


    """

    def __init__(self, trade_name):
        """Initialize storage for one analysis result set.

        Parameters
        ----------
        trade_name : str
            Name of the analysis associated with these results.

        """
        self._results = []
        self._hash_to_result = {}
        self.trade_name = trade_name
        self._results_owner = None
        self._results_owner = None


    @property
    def study(self):
        """Return the parent study associated with these results.

        Returns
        -------
        Study or None
            Parent study resolved through the owning :class:`Results` object, or
            None if this result set is not attached.

        """
        if self._results_owner is None:
            return None
        else:
            return self._results_owner.study


    def get_changing_variables(self):
        """Return variables that can vary across result points.

        Returns
        -------
        list of str
            Variable names that may change across the analysis, including design
            variables, tracked outputs, and sweep parameters when applicable.

        Raises
        ------
        ValueError
            If the associated analysis name is not found in the parent study
            control.

        """
        if self.trade_name in self.study.control.trade_studies:
            trade = self.study.control.trade_studies[self.trade_name]
        elif self.trade_name == "Default State":
            trade = None
        else:
            raise ValueError(
                f'Trade study "{self.trade_name}" not found in study control.'
            )

        # get Design variable names if applicable
        get_desvar_names = getattr(trade, "get_desvar_names", None)
        des_vars = get_desvar_names() if callable(get_desvar_names) else []

        # get parameter variables if applicable
        get_param_names = getattr(trade, "get_all_param_names", None)
        param_names = get_param_names() if callable(get_param_names) else []

        tracked = (
            []
            if self.study.solution.tracked_outputs is None
            else self.study.solution.tracked_outputs
        )

        return des_vars + tracked + list(param_names)

    def set_study_reference(self, study):
        """Set the parent study reference used for variable resolution.

        Parameters
        ----------
        study : Study
            Parent study used to resolve vehicle and analysis metadata.

        Returns
        -------
        None

        """
        self.study = study

    def get_variable(self, variable_name):
        """Return one variable across all stored result points.

        Parameters
        ----------
        variable_name : str
            Variable name to retrieve. This may refer to a tracked output, design
            variable, sweep variable, uncertainty variable, sensitivity variable,
            or a fixed vehicle parameter. Indexed access using bracket notation is
            supported for array-valued variables.

        Returns
        -------
        list
            Values of ``variable_name`` across stored result points. For fixed
            vehicle parameters, the returned list contains one value.

        Raises
        ------
        ValueError
            If this result set is not attached to a :class:`Results` object, if no
            stored results are available, if the parent study reference is missing,
            or if ``variable_name`` cannot be resolved.

        Notes
        -----
        Missing tracked outputs or design-variable outputs are returned as
        ``numpy.nan`` for the corresponding result points.

        """
        if not self._results_owner:
            raise ValueError(
                "Tradestudy has no parent Results. Tradestudy must be attached to a Results object"
            )
        if not self._results:
            raise ValueError("No results available in Results object.")
        if self.study is None:
            raise ValueError("Study reference is not set in Results object.")

        values = []
        # handle tradestudy desvars more robustly
        if self.trade_name in self.study.control.trade_studies:
            trade = self.study.control.trade_studies[self.trade_name]
        elif self.trade_name == "Default State":
            trade = None
        else:
            raise ValueError(
                f'Trade study "{self.trade_name}" not found in study control.'
            )

        # get Design variable names if applicable
        get_desvar_names = getattr(trade, "get_desvar_names", None)
        des_vars = get_desvar_names() if callable(get_desvar_names) else []

        # get parameter variables if applicable
        get_param_names = getattr(trade, "get_all_param_names", None)
        param_names = get_param_names() if callable(get_param_names) else []

        # get UQ variable names if applicable
        aleatory_vars_to_pdf = getattr(trade, "aleatory_var_to_dist", None) or {}
        epistemic_vars_to_intervals=getattr(trade,"epistemic_var_to_vals",None) or {}
        # get sensitivity variable names if applicable
        sensitivity_vars_to_values = (
            getattr(trade, "variable_to_sensitivity_data", None) or {}
        )
        if not self._results:
            print(f"No results for tradestudy {self.trade_name}")
            return None
        # first check if variable name is in tracked outputs

        if variable_name in self.study.solution.tracked_outputs:
            for result in self._results:
                if variable_name in result.outputs:
                    values.append(result.outputs[variable_name])
                else:
                    values.append(np.nan)
                    # log as missing.
            return values

        inds = None
        if "[" in variable_name:
            # Functionality exists that checks for brackets in: step name (plugins)
            # vehicles names, component names, mode names, and parameter names. which [ is not allowed
            # are all not allowed.
            # This handles and has checks for the brackets.
            variable_name, inds = split_dotted_path_and_indices(variable_name)
        # check if the variable is a design variable
        if des_vars and variable_name in des_vars:
            for result in self._results:
                if variable_name in result.outputs:
                    if inds is not None:
                        values.append(result.outputs[variable_name][inds])
                    else:
                        values.append(result.outputs[variable_name])
                else:
                    values.append(np.nan)
                    # log as missing.
            return values
        # check if the variable is a value in a parameter sweep
        # check if the variable was part of the UQ analysis:
        # check if the variable was part of the sensitivity analysis:
        elif (
            variable_name in param_names
            or variable_name in aleatory_vars_to_pdf
            or variable_name in sensitivity_vars_to_values
            or variable_name in epistemic_vars_to_intervals
        ):
            for result in self._results:
                if inds is not None:
                    values.append(result.state[variable_name][inds])
                else:
                    values.append(result.state[variable_name])
                    # log as missing.
            return values
        # then check if the variable name is in the vehicle parameters
        else:
            try:
                if inds is not None:
                    value = self.study.get_value(variable_name)[inds]
                    values.append(value)
                else:
                    value = self.study.get_value(variable_name)
                    values.append(value)
                return values
            except:
                raise ValueError(
                    f"Variable name {variable_name} not found in tracked outputs, trade study sweep variables, or vehicle parameters."
                )

    def _to_dict(self):
        """Serialize the analysis result set to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing serialized result points.

        """
        return {
                "points": [result._to_dict() for result in self._results],
            }

    @classmethod
    def _from_dict(cls, trade_name, results_dict):
        """Reconstruct an analysis result set from a dictionary.

        Parameters
        ----------
        trade_name : str
            Name of the analysis associated with the serialized results.
        results_dict : dict
            Serialized result-set dictionary, typically produced by :meth:`_to_dict`.

        Returns
        -------
        Tradestudy
            Reconstructed result set with restored :class:`Result` points.

        """
        results = cls(trade_name=trade_name)
        for result in results_dict["points"]:
            r = Result._from_dict(result)
            results._results.append(r)
            results._hash_to_result[r.state_hash] = r
        return results


class Result:
    """Store one evaluation result for a single analysis state.

    A ``Result`` contains the input state, tracked outputs, optional template
    checksums, and optional convergence status for one executed analysis point.

    """

    def __init__(self, state):
        """Initialize one stored evaluation result.

        Parameters
        ----------
        state : dict
            Input state dictionary for a single model evaluation.

        Notes
        -----
        The input state is deep-copied and normalized for serialization before the
        result hash is computed.

        """
        self.state = convert_numpy_to_python(copy.deepcopy(state))
        self.state_hash = hash_state(self.state)
        self.template_hashes = {}
        self.outputs = {}
        self.convergence_status = None

    def _add_convergence_status(self, convergence_status):
        """Store optimization convergence status.

        Parameters
        ----------
        convergence_status : bool
            Convergence flag for the executed analysis point.

        Returns
        -------
        None

        """
        self.convergence_status = convergence_status

    def _add_result(self, result_name, result_value):
        """Store one tracked output value.

        Parameters
        ----------
        result_name : str
            Output variable name.
        result_value : object
            Output value for ``result_name``.

        Returns
        -------
        None

        """
        self.outputs[result_name] = result_value

    def _add_template_checksum(self, file_path, checksum):
        """Store one template checksum for provenance tracking.

        Parameters
        ----------
        file_path : str or path-like
            Template or artifact path tracked for this result.
        checksum : str
            Checksum associated with ``file_path``.

        Returns
        -------
        None

        """
        self.template_hashes[file_path] = checksum

    def equal_states(self, result_object):
        """Return whether two results share the same input state.

        Parameters
        ----------
        result_object : Result
            Result object to compare against.

        Returns
        -------
        bool
            True if both results have the same state hash, otherwise False.

        """
        return self.state_hash == result_object.state_hash

    def _to_dict(self):
        """Serialize the result to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the input state, outputs, and
            optional template checksums and convergence status.

        """
        d = {
            "input_state": self.state,
            "outputs": self.outputs,
        }
        if self.template_hashes:
            d["template_checksums"] = self.template_hashes

        if self.convergence_status is not None:
            d["converged"] = self.convergence_status

        return d

    @classmethod
    def _from_dict(cls, results_dict):
        """Reconstruct a result from a dictionary.

        Parameters
        ----------
        results_dict : dict
            Serialized result dictionary, typically produced by :meth:`_to_dict`.

        Returns
        -------
        Result
            Reconstructed result object.

        Notes
        -----
        List-valued state and output entries are converted back to NumPy arrays
        during reconstruction.

        """

        #lists are always arrays in our code
        input_state=copy.deepcopy(results_dict["input_state"])
        for category,data in results_dict['input_state'].items():
            if isinstance(data,list):
                input_state[category]=np.array(data)
            elif isinstance(data,dict):
                for var_name,var_value in data.items():
                    if isinstance(var_value,list):
                        input_state[category][var_name]=np.array(var_value)



        result = cls(input_state)

        output_state=copy.deepcopy(results_dict.get("outputs", {}))
        for var_name,var_value in output_state.items():
            if isinstance(var_value,list):
                output_state[var_name]=np.array(var_value)
            else:
                output_state[var_name]=var_value

        result.outputs = output_state

        result.template_hashes = copy.deepcopy(
            results_dict.get("template_checksums", {})
        )

        if "converged" in results_dict:
            result._add_convergence_status(results_dict["converged"])

        return result
