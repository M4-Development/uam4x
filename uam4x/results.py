import copy
from uam4x.utils import hash_state, convert_numpy_to_python
import matplotlib.pyplot as plt
import numpy as np
from uam4x.utils import split_dotted_path_and_indices


class Results:
    """Represent the results component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, study=None):
        """Initialize a container for trade-study result sets.

        Parameters
        ----------
        study : Any, optional
            Input value for `study`.

        """
        self.tradestudies = {}
        self.study = study

    def get_tradestudy(self, name):
        """Return tradestudy.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.

        Returns
        -------
        Any
            Requested value.

        """
        if name in self.tradestudies:
            return self.tradestudies[name]
        else:
            return None

    def add_tradestudy(self, name, Trade):
        """Attach a trade-study results container to this results object.

        Parameters
        ----------
        name : Any
            Identifier used by this method.
        Trade : Any
            Input value for `Trade`.

        """
        Trade._results_owner = self
        self.tradestudies[name] = Trade

    def has_results(self):
        """Return `True` when any trade-study results are stored.

        Returns
        -------
        Any
            Status flag for the requested check.

        """
        if self.tradestudies:
            return True
        else:
            return False

    def get_all_tradestudies(self):
        """Return all tradestudies.

        Returns
        -------
        Any
            Requested value.

        """
        tradestudies = list(self.tradestudies.keys())
        return tradestudies

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        tradename_to_tradestudy = {}
        for name, tradestudy in self.tradestudies.items():
            tradename_to_tradestudy[name] = tradestudy._to_dict()
        return tradename_to_tradestudy

    @classmethod
    def _from_dict(cls, tradename_to_results, study):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        tradename_to_results : Any
            Mapping from trade-study names to serialized result data.
        study : Any
            Input value for `study`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        res = cls(study)
        for trade_name, results in tradename_to_results.items():
            ts = Tradestudy._from_dict(trade_name, results)
            res.add_tradestudy(trade_name, ts)
        return res


class Tradestudy:
    """Represent the tradestudy component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, trade_name):
        """Initialize storage for all result points of one trade study.

        Parameters
        ----------
        trade_name : Any
            Trade-study name under `Study.control.trade_studies`.

        """
        self._results = []
        self._hash_to_result = {}
        self.trade_name = trade_name
        self._results_owner = None
        self._results_owner = None


    @property
    def study(self):
        """Return the parent `Study` object through the owning `Results` container.

        Returns
        -------
        Any
            Requested value.

        """
        if self._results_owner is None:
            return None
        else:
            return self._results_owner.study


    def get_changing_variables(self):
        """Return variables that change across trade-study evaluations.

        Returns
        -------
        Any
            Requested value.

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
        study : Any
            Input value for `study`.

        """
        self.study = study

    def get_variable(self, variable_name):
        """Return one variable across all stored result points in the trade study.

        Parameters
        ----------
        variable_name : Any
            Identifier used by this method.

        Returns
        -------
        Any
            Requested value.

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
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
                "points": [result._to_dict() for result in self._results],
            }

    @classmethod
    def _from_dict(cls, trade_name, results_dict):
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        trade_name : Any
            Trade-study name under `Study.control.trade_studies`.
        results_dict : Any
            Serialized dictionary payload for one result or tradestudy.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        results = cls(trade_name=trade_name)
        for result in results_dict["points"]:
            r = Result._from_dict(result)
            results._results.append(r)
            results._hash_to_result[r.state_hash] = r
        return results


class Result:
    """Represent the result component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, state):
        """Initialize one stored evaluation result for a single input state.

        Parameters
        ----------
        state : Any
            Input state dictionary for a single model evaluation.

        """
        self.state = convert_numpy_to_python(copy.deepcopy(state))
        self.state_hash = hash_state(self.state)
        self.template_hashes = {}
        self.outputs = {}
        self.convergence_status = None

    def _add_convergence_status(self, convergence_status):
        """Add convergence status.

        Parameters
        ----------
        convergence_status : Any
            Collection consumed by this method.

        """
        self.convergence_status = convergence_status

    def _add_result(self, result_name, result_value):
        """Add result.

        Parameters
        ----------
        result_name : Any
            Identifier used by this operation.
        result_value : Any
            Input value for `result_value`.

        """
        self.outputs[result_name] = convert_numpy_to_python(result_value)

    def _add_template_checksum(self, file_path, checksum):
        """Store one template checksum associated with this result point.

        Parameters
        ----------
        file_path : Any
            Template or artifact path tracked for provenance.
        checksum : Any
            Checksum value associated with a file/template artifact.

        """
        self.template_hashes[file_path] = checksum

    def equal_states(self, result_object):
        """Return whether this result and another result share the same state hash.

        Parameters
        ----------
        result_object : Any
            Input value for `result_object`.

        Returns
        -------
        Any
            Status flag for the requested check.

        """
        return self.state_hash == result_object.state_hash

    def _to_dict(self):
        """Serialize this object to a JSON-compatible dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

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
        """Reconstruct this object from serialized dictionary data.

        Parameters
        ----------
        results_dict : Any
            Serialized dictionary payload for one result or tradestudy.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        result = cls(results_dict["input_state"])
        result.outputs = copy.deepcopy(results_dict.get("outputs", {}))
        result.template_hashes = copy.deepcopy(
            results_dict.get("template_checksums", {})
        )

        if "converged" in results_dict:
            result._add_convergence_status(results_dict["converged"])

        return result
