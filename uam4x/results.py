from __future__ import annotations
from operator import attrgetter
import copy
import warnings


import numpy as np

from uam4x.vehicle import VehicleManager,Vehicle
from uam4x.utils import hash_dict, convert_numpy_to_python,split_dotted_path_and_indices
from uam4x.control import trade_type_to_class

class Results:
    """Store analysis result sets for a study.

    A ``Results`` object maps solution names to :class:`SolutionResults`
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
        self.solutions = {}
        self.study = study

    def get_solution_result(self, name):
        """Return one stored solution result container by name.

        Parameters
        ----------
        name : str
            Solution name.

        Returns
        -------
        SolutionResults or None
            Stored result container for ``name``, or ``None`` if it is not
            present.

        """
        if name in self.solutions:
            return self.solutions[name]
        else:
            return None

    def get_tradestudy(self, solution_name, trade_name)->TradestudyResults:
        """Return a trade-study result from a named solution.

        Parameters
        ----------
        solution_name : str
            Name of the solution that owns the result.
        trade_name : str
            Name of the trade study within that solution's results.

        Returns
        -------
        TradestudyResults or None
            Stored trade-study result, or ``None`` if the solution is not present.

        Raises
        ------
        ValueError
            If the solution exists but has no result for ``trade_name``.

        """
        solution_result = self.get_solution_result(solution_name)
        if solution_result is None:
            return None
        return solution_result.get_tradestudy(trade_name)

    def add_solution_result(self, solution_result):
        """Attach an analysis result container to this results object.

        Parameters
        ----------
        solution_result : SolutionResults
            Per-solution result container to attach.

        Returns
        -------
        None

        Notes
        -----
        The attached :class:`SolutionResults` receives the same parent study
        reference as this :class:`Results` object.

        """
        solution_result.study = self.study
        self.solutions[solution_result.name] = solution_result

    def has_solution_result(self):
        """Return whether any analysis results are stored.

        Returns
        -------
        bool
            True if at least one analysis result container is stored, otherwise
            False.

        """
        if self.solutions:
            return True
        else:
            return False

    def get_all_solution_results(self):
        """Return names of all stored solution result containers.

        Returns
        -------
        list of str
            Solution names with stored results.

        """
        solution_results = list(self.solutions.keys())
        return solution_results

    def _to_dict(self):
        """Serialize the results object to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary mapping solution names to serialized
            :class:`SolutionResults` data.

        """
        solution_name_to_solution_results = {}
        for name, solution_results in self.solutions.items():
            solution_name_to_solution_results[name] = solution_results._to_dict()
        return solution_name_to_solution_results

    @classmethod
    def _from_dict(cls, solution_name_to_solution_results, study):
        """Reconstruct a results object from a dictionary.

        Parameters
        ----------
        solution_name_to_solution_results : dict
            Mapping from solution names to serialized
            :class:`SolutionResults` data.
        study : Study
            Parent study associated with the reconstructed results.

        Returns
        -------
        Results
            Reconstructed results object with attached study reference.

        """
        res = cls(study)
        for solution_name, trade_name_to_trade_data in solution_name_to_solution_results.items():
            solution_result = SolutionResults._from_dict(
                                solution_name=solution_name,
                                study=study,
                                tradename_to_results=trade_name_to_trade_data)
            res.add_solution_result(solution_result)
        return res

class SolutionResults:
    def __init__(self,name='',study=None):
        """Initialize the results associated with one solution.

        Parameters
        ----------
        name : str, optional
            Name of the solution that produced these results.
        study : Study, optional
            Parent study used for current-model consistency checks.

        """
        self.name=name
        self.study=study
        self.tradestudies={}

    def get_tradestudy(self, name):
            """Return one stored analysis result container by name.

            Parameters
            ----------
            name : str
                Analysis name.

            Returns
            -------
            TradestudyResults
                Stored result container for ``name``.

            Raises
            ------
            ValueError
                If no result is stored under ``name`` for this solution.

            """
            if name in self.tradestudies:
                ts=self.tradestudies[name]
                if self.study is None:
                    raise ValueError(
                        f"Results for solution {self.name!r} are not attached "
                        "to a Study, so current-model consistency checks "
                        "cannot be performed."
                    )
                # Compare saved execution inputs with current model inputs.
                # A default-state run has no trade-study definition.
                if name != "Default State":
                    executed_trade_dict=ts._executed_tradestudy._to_dict()
                    current_trade=self.study.control.trade_studies.get(name)
                    if current_trade is None:
                        warnings.warn(
                            f"Results for solution {self.name!r}, trade {name!r} "
                            "refer to a trade study that is not present in the "
                            "current model. The saved results remain accessible.",
                            UserWarning,
                            stacklevel=2,
                        )
                        trade_dict=executed_trade_dict
                    else:
                        trade_dict=current_trade._to_dict()
                    if hash_dict(executed_trade_dict)!=hash_dict(trade_dict):
                        warnings.warn(
                            f"Results for solution {self.name!r}, trade {name!r} were run with a different trade-study definition. Rerun this combination before using current trade settings.",
                            UserWarning,
                            stacklevel=2,
                        )
                executed_vehicles={veh.name:veh._to_dict() for veh in ts._executed_vehicles}
                current_vehicles={veh.name:veh._to_dict() for veh in self.study.vehicles}
                for vehicle_name in sorted(executed_vehicles):
                    if vehicle_name not in current_vehicles:
                        reason="is missing from the current model"
                    elif hash_dict(executed_vehicles[vehicle_name])!=hash_dict(current_vehicles[vehicle_name]):
                        reason="has changed since these results were run"
                    else:
                        continue
                    warnings.warn(
                        f"Results for solution {self.name!r}, trade {name!r}: vehicle {vehicle_name!r} {reason}. Rerun this combination if it depends on that vehicle.",
                        UserWarning,
                        stacklevel=2,
                    )
                return ts
            raise ValueError(f'The tradestudy {name} is not a tradestudy associated with the solution {self.name}')

    def add_tradestudy(self, name, trade):
        """Attach an analysis result container to this results object.

        Parameters
        ----------
        name : str
            Analysis name.
        trade : TradestudyResults
            Result container to attach.

        Returns
        -------
        None

        Notes
        -----
        The attached :class:`TradestudyResults` receives a back-reference to this
        :class:`Results` object.

        """
        if name != trade.trade_name:
            raise ValueError(
                f"Trade-study key {name!r} does not match "
                f"trade name {trade.trade_name!r}"
            )
        if trade.solution_name != self.name:
            raise ValueError(
                f"Trade study belongs to solution {trade.solution_name!r}, "
                f"not {self.name!r}"
            )

        trade._results_owner = self
        self.tradestudies[name] = trade

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
            :class:`TradestudyResults` data.

        """
        tradename_to_tradestudy = {}
        for name, tradestudy in self.tradestudies.items():
            tradename_to_tradestudy[name] = tradestudy._to_dict()
        return tradename_to_tradestudy

    @classmethod
    def _from_dict(cls, solution_name,tradename_to_results, study):
        """Reconstruct a results object from a dictionary.

        Parameters
        ----------
        solution_name : str
            Name of the solution associated with the serialized results.
        tradename_to_results : dict
            Mapping from analysis names to serialized :class:`TradestudyResults` data.
        study : Study
            Parent study associated with the reconstructed results.

        Returns
        -------
        Results
            Reconstructed results object with attached study reference.

        """

        res = cls(name=solution_name,study=study)
        for trade_name, results in tradename_to_results.items():
            ts = TradestudyResults._from_dict(trade_name, results,solution_name)
            res.add_tradestudy(trade_name, ts)
        return res


class TradestudyResults:
    """Store all result points for one named analysis.

    A ``TradestudyResults`` groups the individual :class:`Result` objects produced by
    one analysis and provides access to variables that change across those
    result points.


    """

    def __init__(self, trade_name,solution_name,
                 executed_vehicles,
                 executed_tradestudy,
                 tracked_output_names):
        """Initialize storage for one analysis result set.

        Parameters
        ----------
        trade_name : str
            Name of the analysis associated with these results.
        solution_name : str
            Name of the solution that produced these results.
        executed_vehicles : VehicleManager
            Snapshot of the vehicles used for execution.
        executed_tradestudy : object or None
            Snapshot of the trade-study definition used for execution. This is
            ``None`` for a default-state run.
        tracked_output_names : sequence of str or None
            Output variable names captured for each result point.

        """
        self._results = []
        self._hash_to_result = {}
        self.trade_name = trade_name
        self._results_owner = None
        self.solution_name=solution_name
        self.tracked_output_names=tracked_output_names
        self._executed_vehicles=executed_vehicles
        self._executed_tradestudy=executed_tradestudy

    @property
    def executed_tradestudy(self):
        """Return the trade-study definition used to produce these results."""
        return self._executed_tradestudy

    @property
    def executed_vehicles(self):
        """Return the vehicle snapshot used to produce these results."""
        return self._executed_vehicles

    def get_case(self,case_id):
        """Return a result point by case identifier.

        Parameters
        ----------
        case_id : int
            Identifier assigned to the result point during execution.

        Returns
        -------
        Result
            Result point whose ``case_id`` matches ``case_id``.

        Raises
        ------
        ValueError
            If no stored result has the requested case identifier.

        """
        for result in self._results:
            if result.case_id==case_id:
                return result

        raise ValueError(f'no case found for case id {case_id}')

    @property
    def study(self):
        """Return the parent study associated with these results.

        Returns
        -------
        Study
            Parent study resolved through the owning result containers.

        Raises
        ------
        ValueError
            If these trade-study results are not attached to a study.

        """
        study = (
            None if self._results_owner is None
            else self._results_owner.study
        )
        if study is None:
            raise ValueError(
                f"Results for solution {self.solution_name!r}, trade "
                f"{self.trade_name!r} are not attached to a Study. Add the "
                "TradestudyResults through SolutionResults.add_tradestudy(), then "
                "attach that SolutionResults to a Results object with a Study."
            )
        return study


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

        trade = self._executed_tradestudy

        # get Design variable names if applicable
        get_desvar_names = getattr(trade, "get_desvar_names", None)
        des_vars = get_desvar_names() if callable(get_desvar_names) else []

        # get parameter variables if applicable
        get_param_names = getattr(trade, "get_all_param_names", None)
        param_names = get_param_names() if callable(get_param_names) else []


        tracked = (
            []
            if self.tracked_output_names is None
            else self.tracked_output_names
        )

        return des_vars + tracked + list(param_names)


    def get_variable(self, variable_name,*,as_column=False):
        """Return one variable across all stored result points.

        Parameters
        ----------
        variable_name : str
            Variable name to retrieve. This may refer to a tracked output, design
            variable, sweep variable, uncertainty variable, sensitivity variable,
            or a fixed vehicle parameter. Indexed access using bracket notation is
            supported for array-valued variables.
        as_column : bool, optional
            If True, insert a singleton axis after the result-point axis. For a
            scalar-valued variable this changes the returned shape from ``(n,)``
            to ``(n, 1)``. The default is False.

        Returns
        -------
        numpy.ndarray
            Values of ``variable_name`` across stored result points. By default,
            scalar-valued variables have shape ``(n,)`` and array-valued
            variables preserve their trailing dimensions. If ``as_column`` is
            True, a singleton axis is inserted after the first axis. For fixed
            vehicle parameters, the returned array contains one value along its
            first axis.

        Raises
        ------
        ValueError
            If no stored results are available or if ``variable_name`` cannot be
            resolved.

        Notes
        -----
        Missing tracked outputs or design-variable outputs are returned as
        ``numpy.nan`` for the corresponding result points.

        """
        if not self._results:
            raise ValueError("No results available in Results object.")

        values = []
        # handle tradestudy desvars more robustly
        trade=self._executed_tradestudy

        # get Design variable names if applicable
        get_desvar_names = getattr(trade, "get_desvar_names", None)
        des_vars = get_desvar_names() if callable(get_desvar_names) else []

        # get parameter variables if applicable
        get_param_names = getattr(trade, "get_all_param_names", None)
        param_names = get_param_names() if callable(get_param_names) else []

        # get UQ variable names if applicable
        aleatory_vars_to_pdf = getattr(trade, "aleatory_var_to_dist", None) or {}
        epistemic_vars_to_intervals=getattr(trade,"epistemic_var_to_interval",None) or {}
        # get sensitivity variable names if applicable
        sensitivity_vars_to_values = (
            getattr(trade, "variable_to_sensitivity_data", None) or {}
        )
        # first check if variable name is in tracked outputs
        tracked_output_names = self.tracked_output_names or []
        if variable_name in tracked_output_names:
            for result in self._results:
                if variable_name in result.outputs:
                    values.append(result.outputs[variable_name])
                else:
                    values.append(np.nan)
                    # log as missing.
            values = np.asarray(values)
            if as_column:
                values=values[:,None]
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
            values = np.asarray(values)
            if as_column:
                values=values[:,None]
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
            values = np.asarray(values)
            if as_column:
                values=values[:,None]
            return values
        # then check if the variable name is in the vehicle parameters
        else:
            try:
                name='.'.join(variable_name.split('.')[1:])
                if inds is not None:
                    value = attrgetter(name)(self._executed_vehicles)[inds]
                    values.append(value)
                else:
                    value = attrgetter(name)(self._executed_vehicles)
                    values.append(value)
                values = np.asarray(values)
                if as_column:
                    values=values[:,None]
                return values
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Variable name {variable_name} not found in tracked outputs, trade study sweep variables, or vehicle parameters."
                ) from exc

    def _to_dict(self):
        """Serialize the analysis result set to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing serialized result points.

        """
        return {
                "points": [result._to_dict() for result in self._results],
                'executed_vehicles':{vehicle.name:vehicle._to_dict()
                                     for vehicle in self._executed_vehicles},
                'executed_tradestudy':(
                    None if self._executed_tradestudy is None
                    else self._executed_tradestudy._to_dict()
                                       ),
                'tracked_output_names':self.tracked_output_names
            }

    @classmethod
    def _from_dict(cls, trade_name, results_dict,solution_name):
        """Reconstruct an analysis result set from a dictionary.

        Parameters
        ----------
        trade_name : str
            Name of the analysis associated with the serialized results.
        results_dict : dict
            Serialized result-set dictionary, typically produced by :meth:`_to_dict`.
        solution_name : str
            Name of the solution that produced the serialized results.

        Returns
        -------
        TradestudyResults
            Reconstructed result set with restored :class:`Result` points.

        """

        vehicles=VehicleManager()
        for name,data in results_dict['executed_vehicles'].items():
            vehicles.add_vehicle(Vehicle._from_dict(data))

        trade_dict=results_dict['executed_tradestudy']
        if not trade_dict:
            executed_trade = None
        else:
            trade_type_name = trade_dict.get('type')
            if trade_type_name not in trade_type_to_class:
                raise ValueError(
                    f"Unknown executed trade study type: {trade_type_name!r}"
                )
            analysis_type=trade_type_to_class[trade_type_name]
            executed_trade=analysis_type._from_dict(trade_dict)

        tracked_output_names=results_dict['tracked_output_names']

        results = cls(
            trade_name=trade_name,
            solution_name=solution_name,
            executed_vehicles=vehicles,
            executed_tradestudy=executed_trade,
            tracked_output_names=tracked_output_names)

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
        self.state = copy.deepcopy(state)
        self.state_hash = hash_dict(self.state)
        self.template_hashes = {}
        self.outputs = {}
        self.convergence_status = None
        self.case_id=0

    def _set_case_id(self,case_id):
        """Assign the execution case identifier for this result point.

        Parameters
        ----------
        case_id : int
            Identifier assigned by the execution engine.

        """
        self.case_id=case_id

    def get_var_val(self,name):
        """Return a state or output value by variable name.

        Parameters
        ----------
        name : str
            Variable name to retrieve.

        Returns
        -------
        object
            Value stored in the input state or tracked outputs.

        Raises
        ------
        ValueError
            If ``name`` is absent from both the input state and outputs.

        Notes
        -----
        Input-state values take precedence if the same name appears in both
        mappings.

        """
        if name in self.state:
            return self.state[name]
        elif name in self.outputs:
            return self.outputs[name]

        raise ValueError(f'Variable with name {name} not in result')

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
            "input_state": convert_numpy_to_python(self.state),
            "outputs": self.outputs,
        }
        if self.template_hashes:
            d["template_checksums"] = self.template_hashes

        if self.convergence_status is not None:
            d["converged"] = self.convergence_status


        d["case_id"]=self.case_id

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

        if "case_id" in results_dict:
            result._set_case_id(results_dict['case_id'])

        return result
