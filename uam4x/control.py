from __future__ import annotations
from typing import Any, Dict, List, Union
import copy
from uam4x.log import get_log
from uam4x.utils import log
import numpy as np
from uam4x.utils import (
    is_continuous_type_or_value,
    is_numpy_array,
    value_to_type_name,
    replace_empty_quotes_with_none,
)
from uam4x.UQ import BasicAleatory, RegressionPCE, GaussQuadraturePCE, SobolSensitivity, \
BasicEpistemic, BasicMixedIVP, BasicSOP, MixedRegressionPCE, \
AleatoryCSVPoints, EpistemicCSVPoints, MixedCSVPoints

from uam4x.local_sensitivity import LocalSensitivity

DEFAULT_TRADE_NAME = "Default State"


class Optimization:
    """Define the user-facing optimization setup for a UAM4X trade study.

    An ``Optimization`` object is the main optimization API exposed to UAM4X
    users. It stores design variables, constraints, objective metadata, and
    driver settings that are later translated into the underlying OpenMDAO
    optimization execution.

    Notes
    -----
    This class defines the optimization problem at the UAM4X interface level.
    Lower-level OpenMDAO optimization machinery is used internally during
    execution.

    """

    def __init__(self) -> None:
        """Initialize an optimization definition.

        Notes
        -----
        Construction initializes empty design-variable and constraint collections,
        no objective, default driver settings, and default optimizer options.

        """
        self.des_vars = {}
        self.constraints = {}
        self.objective_name = ""
        self._mdao_driver_options = {"optimizer": "SLSQP"}
        self._optimizer_options = {}
        self.driver = "ScipyOptimizeDriver"

    def _to_dict(self):
        """Serialize the optimization definition to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing design variables, constraints,
            objective name, driver settings, and optimizer options.

        """
        save_dict = {
            "type": self.__class__.__name__,
            "des_vars": self.des_vars,
            "constraints": self.constraints,
            "objective_name": self.objective_name,
            "driver_options": self._mdao_driver_options,
            "optimizer_options": self._optimizer_options,
            "driver": self.driver,
        }
        return save_dict

    def set_driver(self, name):
        """Set the OpenMDAO driver class name.

        Parameters
        ----------
        name : str
            Driver class name.

        Returns
        -------
        None

        Notes
        -----
        The currently supported driver names are ``"pyOptSparseDriver"`` and
        ``"ScipyOptimizeDriver"``.

        """
        self.driver = name

    def add_driver_option(self, name, value):
        """Add or overwrite an OpenMDAO driver option.

        Parameters
        ----------
        name : str
            Driver option name.
        value : object
            Driver option value.

        Returns
        -------
        None

        """
        self._mdao_driver_options[name] = value

    def add_optimizer_option(self, name, value):
        """Add or overwrite an optimizer-specific option.

        Parameters
        ----------
        name : str
            Optimizer option name.
        value : object
            Optimizer option value.

        Returns
        -------
        None

        """
        self._optimizer_options[name] = value

    def get_desvar_names(self):
        """Return design-variable names.

        Returns
        -------
        list of str
            Names of all registered design variables.

        """
        return list(self.des_vars.keys())

    @classmethod
    def _from_dict(cls, data: dict) -> "Optimization":
        """Reconstruct an optimization definition from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized optimization dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        Optimization
            Reconstructed optimization definition.

        Notes
        -----
        List-valued lower and upper bounds are converted back to NumPy arrays during
        reconstruction.

        """
        obj = cls()
        # if upper/lower for DV or constraints are lists, must be
        # desired to be arrays, so convert.
        data["constraints"] = replace_empty_quotes_with_none(data["constraints"])
        data["des_vars"] = replace_empty_quotes_with_none(data["des_vars"])
        for constraint, constraint_vals in data["constraints"].items():
            if isinstance(constraint_vals["lower"], list):
                constraint_vals["lower"] = np.array(constraint_vals["lower"])
            if isinstance(constraint_vals["upper"], list):
                constraint_vals["upper"] = np.array(constraint_vals["upper"])
        for des_var, des_var_vals in data["des_vars"].items():
            if isinstance(des_var_vals["lower"], list):
                des_var_vals["lower"] = np.array(des_var_vals["lower"])
            if isinstance(des_var_vals["upper"], list):
                des_var_vals["upper"] = np.array(des_var_vals["upper"])

        obj.des_vars = data["des_vars"]
        obj.constraints = data["constraints"]
        obj.objective_name = data["objective_name"]
        obj.driver = data["driver"]
        if data["driver_options"]:
            for name, value in data["driver_options"].items():
                obj.add_driver_option(name, value)
        if data["optimizer_options"]:
            for name, value in data["optimizer_options"].items():
                obj.add_optimizer_option(name, value)
        return obj

    def add_design_var(
        self,
        name: str,
        lower: np.ndarray | float | np.floating | None = None,
        upper: np.ndarray | float | np.floating | None = None,
        indices: list | None = None,
    ):
        """Add a design variable to the optimization definition.

        Parameters
        ----------
        name : str
            Design-variable name.
        lower : np.ndarray | float | np.floating | None, optional
            Lower bound for the design variable.
        upper : np.ndarray | float | np.floating | None, optional
            Upper bound for the design variable.
        indices : list | None, optional
            Indices of an array-valued design variable to expose as active design
            variables. If omitted, the full variable is used.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If bound types are inconsistent or if a bound is not continuous-valued.
        ValueError
            If array-shaped bounds have mismatched shapes or if any upper bound is
            less than the corresponding lower bound.

        """
        self.check_upper_lower(
            name, "design variable", lower=lower, upper=upper, indices=indices
        )
        self.des_vars[name] = {"lower": lower, "upper": upper, "indices": indices}

    def edit_design_var(
            self,
            name: str,
            dict_key:str,
            value: np.ndarray | float | np.floating |list | None = None
            ):
        """Edit an existing design variable's lower, upper, or indices value.
        Parameters
        ----------
        name : str
            The name of the design variable to edit.
        dict_key : str
            The field to update, must be one of 'lower', 'upper', or 'indices'.
        value : np.ndarray | float | np.floating | list | None, optional
            The new value to assign to the field."""
        current_desvar = self.des_vars[name]
        if dict_key not in current_desvar:
            raise KeyError
        bounds = {k: current_desvar[k] for k in ('lower', 'upper', 'indices')}
        bounds[dict_key] = value
        self.check_upper_lower(name, 'design variable', **bounds)
        self.des_vars[name][dict_key] = value

    def add_constraint(
        self,
        name: str,
        lower: np.ndarray | float | np.floating | None = None,
        upper: np.ndarray | float | np.floating | None = None,
        indices=None,
    ):
        """Add a constraint to the optimization definition.

        Parameters
        ----------
        name : str
            Constraint variable name.
        lower : np.ndarray | float | np.floating | None, optional
            Lower bound for the constraint.
        upper : np.ndarray | float | np.floating | None, optional
            Upper bound for the constraint.
        indices : list or ndarray, optional
            Indices of an array-valued constrained variable to apply as active
            constraints. If omitted, the full variable is constrained.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If bound types are inconsistent or if a bound is not continuous-valued.
        ValueError
            If both bounds are omitted, if array-shaped bounds have mismatched
            shapes, or if any upper bound is less than the corresponding lower
            bound.

        """
        self.check_upper_lower(
            name, const_or_dv="constraint", lower=lower, upper=upper, indices=indices
        )
        self.constraints[name] = {
            "lower": lower,
            "upper": upper,
            "indices": indices,
        }

    def edit_constraint(
            self,
            name: str,
            dict_key:str,
            value: np.ndarray | float | np.floating |list | None = None
            ):
        """Edit an existing constraints's lower, upper, or indices value.
        Parameters
        ----------
        name : str
            The name of the constraint to edit.
        dict_key : str
            The field to update, must be one of 'lower', 'upper', or 'indices'.
        value : np.ndarray | float | np.floating | list | None, optional
            The new value to assign to the field."""
        current_constraint = self.constraints[name]
        if dict_key not in current_constraint:
            raise KeyError
        bounds = {k: current_constraint[k] for k in ('lower', 'upper', 'indices')}
        bounds[dict_key] = value
        self.check_upper_lower(name, 'constraint', **bounds)
        self.constraints[name][dict_key] = value

    @staticmethod
    def check_upper_lower(
        name: str,
        const_or_dv,
        lower: np.ndarray | float | np.floating | None = None,
        upper: np.ndarray | float | np.floating | None = None,
        indices: list | np.ndarray | None = None,
    ):

        # At least one bound must be declared on a constraint

        """Validate lower and upper bound declarations.

        Parameters
        ----------
        name : str
            Design-variable or constraint name.
        const_or_dv : str
            Kind of quantity being validated, typically ``"constraint"`` or
            ``"design variable"``.
        lower : np.ndarray | float | np.floating | None, optional
            Lower bound to validate.
        upper : np.ndarray | float | np.floating | None, optional
            Upper bound to validate.
        indices : list | np.ndarray | None, optional
            Indices selecting active entries of an array-valued design variable or
            constraint. This argument is accepted for API consistency but is not
            validated by this method.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If bound values are not continuous-valued or if lower and upper bounds
            have different types.
        ValueError
            If a constraint omits both bounds, if array-shaped bounds have
            mismatched shapes, or if any upper bound is less than the corresponding
            lower bound.

        """

        # Check continuity within our paradigm (here user would not be able to declare constraints as lists, ints etc.)
        # without, relies on MDAO to consolidate mismatch e.g declared as array but some weird list is provided.
        if lower is not None and not is_continuous_type_or_value(lower):
            raise TypeError(
                f"The lower bound for {name} is not continuos.\n"
                f"Only floats and floating point arrays are accepted."
            )
        if upper is not None and not is_continuous_type_or_value(upper):
            raise TypeError(
                f"The upper bound for {name} is not continuos.\n"
                f"Only floats and floating point arrays are accepted."
            )

        elif lower is not None and upper is not None:
            if type(lower) != type(upper):
                raise TypeError(
                    f"Upper and lower bounds must be of the same type. The {const_or_dv}:{name}\n"
                    f"has a lower bound of type {value_to_type_name(lower)} and an upper bound of type {value_to_type_name(upper)}\n"
                )

            # Note, if one is an np.ndarray the other must be as well
            if is_numpy_array(lower) and is_numpy_array(upper):
                if lower.shape != upper.shape:
                    raise ValueError(
                        f"{const_or_dv} shapes must match.The {const_or_dv} {name}\n"
                        f"has lower shape: {lower.shape} and upper shape {upper.shape}."
                    )
                if np.any(upper - lower < 0):
                    raise ValueError(
                        f"{const_or_dv} upper bounds must be greater than or equal to the lower bounds.\n"
                        f"The following {const_or_dv} {name} has lower={lower} and upper={upper}."
                    )
            else:
                if upper - lower < 0:
                    raise ValueError(
                        f"{const_or_dv} upper bounds must be greater than or equal to the lower bounds.\n"
                        f"The following {const_or_dv} {name} has lower={lower} and upper={upper}."
                    )

    def add_objective(self, name: str):
        """Set the optimization objective variable.

        Parameters
        ----------
        name : str
            Objective variable name.

        Returns
        -------
        None

        """
        self.objective_name = name


class ParameterSweep:
    """Define a user-facing parameter sweep for a UAM4X trade study.

    A ``ParameterSweep`` maps parameter names to equal-length lists of values
    and generates one study state per list position. A sweep may chain to
    another :class:`ParameterSweep` or to an :class:`Optimization`, allowing
    nested sweeps and sweep-plus-optimization workflows.

    Notes
    -----
    When a child sweep is present, generated states form the Cartesian product
    of the parent and child sweep states.

    """

    log = get_log(log, object_name="parameter sweep")

    def __init__(
        self,
        param_name_to_value_list: Dict[str, Any],
        child: Union["ParameterSweep", Optimization] = None,
    ) -> None:
        """Initialize a parameter sweep definition.

        Parameters
        ----------
        param_name_to_value_list : Dict[str, Any]
            Mapping from parameter names to lists of sweep values.
        child : Union['ParameterSweep', Optimization], optional
            Child sweep or optimization definition chained below this sweep.

        Raises
        ------
        TypeError
            If any mapped sweep values are not provided as lists.
        ValueError
            If parameter value lists do not all have the same length.

        Notes
        -----
        The sweep length is inferred from the first parameter value list and all
        other lists must match it.

        """
        self.child = child
        self.length: int = None
        self.param_name_to_value_list = param_name_to_value_list
        self.log = ParameterSweep.log
        self.log.debug("initialize")
        for name, value_list in self.param_name_to_value_list.items():
            if not isinstance(value_list, list):
                self.log.error(
                    f"Values for parameter '{name}' must be a list, got {type(value_list).__name__}"
                )
                raise TypeError(
                    f"Values for parameter '{name}' must be a list, got {type(value_list).__name__}"
                )
            if self.length is None:
                self.length = len(value_list)
            elif self.length != len(value_list):
                self.log.error(
                    f"All parameter value lists must have the same length, but '{name}' has length {len(value_list)} while others have length {self.length}"
                )
                raise ValueError(
                    f"All parameter value lists must have the same length, but '{name}' has length {len(value_list)} while others have length {self.length}"
                )

    def add_child(self, child: Union["ParameterSweep", Optimization]) -> None:
        """Set the child sweep or optimization definition.

        Parameters
        ----------
        child : Union['ParameterSweep', Optimization]
            Child object chained below this sweep.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``child`` is not a :class:`ParameterSweep` or :class:`Optimization`.

        """
        self.log.debug("set child")
        if not isinstance(child, (Optimization, ParameterSweep)):
            self.log.error("Child must be ParameterSweep or Optimization")
            raise TypeError("Child must be ParameterSweep or Optimization")
        self.child = child

    def get_optim_child(self):
        """Return the optimization child at the end of the sweep chain.

        Returns
        -------
        Optimization or None
            Terminal optimization object if one exists, otherwise None.

        """
        child = self.child
        if child is None:
            return None
        if isinstance(child, ParameterSweep):
            return child.get_optim_child()
        if isinstance(child, Optimization):
            return child

    def get_desvar_names(self):
        # assumes optimization cannot have child
        """Return design-variable names from the terminal optimization child.

        Returns
        -------
        list of str
            Design-variable names from the child optimization, or an empty list if
            no optimization child is present.

        """
        child = self.child
        if self.child is None:
            return []
        else:
            return child.get_desvar_names()

    def get_all_param_names(self):
        # assumes optimization cannot have child
        """Return all parameter names across the sweep chain.

        Returns
        -------
        set of str
            Union of parameter names defined by this sweep and any child sweeps.

        """
        child = self.child
        params = set(self.param_name_to_value_list.keys())
        if isinstance(self.child, ParameterSweep):
            return params.union(child.get_all_param_names())
        else:
            return params

    def get_states(self) -> List[Dict[str, Any]]:
        """Generate trade-study states for the sweep chain.

        Returns
        -------
        list of dict
            Generated state dictionaries. Each state contains one selected value
            for every sweep parameter in the chain. If a child optimization is
            present, its serialized optimization definition is embedded in each
            state and ``"optimization"`` is set to True.

        Notes
        -----
        Nested sweeps generate the Cartesian product of parent and child sweep
        states.

        """
        states = []
        self.log.debug("get states")
        for i in range(self.length):
            state = {}
            for name, values in self.param_name_to_value_list.items():
                state[name] = values[i]
            if isinstance(self.child, Optimization):
                state["optimization"] = True
                state.update(self.child._to_dict())
                states.append(state)
            elif isinstance(self.child, ParameterSweep):
                # in cases where there is a child parameter sweep,
                # want a cartesian product of states (for every in parent, create a state
                # for for every in child).
                child_states = self.child.get_states()
                states_to_add = []
                for child_state in child_states:
                    sample_state = copy.deepcopy(state)
                    sample_state.update(child_state)
                    states_to_add.append(sample_state)
                states.extend(states_to_add)
            else:
                state["optimization"] = False
                states.append(state)
        return states

    def _to_dict(self):
        """Serialize the parameter sweep to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the sweep values and optional
            child definition.

        """
        self.log.debug("write to dict")
        return {
            "type": "ParameterSweep",
            "param_name_to_value_list": self.param_name_to_value_list,
            "child": self.child._to_dict() if self.child else "",
        }

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct a parameter sweep from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized parameter sweep dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        ParameterSweep
            Reconstructed parameter sweep with any child sweep or optimization
            restored.

        """
        cls.log.debug("parameter sweep from dict")
        child_dict = data["child"]
        if not child_dict:
            child = None
        elif child_dict["type"] == "ParameterSweep":
            child = ParameterSweep._from_dict(child_dict)
        elif child_dict["type"] == "Optimization":
            child = Optimization._from_dict(child_dict)
        return ParameterSweep(data["param_name_to_value_list"], child=child)


class Control:
    """Define study metadata and trade-study configurations for UAM4X.

    A ``Control`` object is the top-level user-facing container for study
    metadata and named trade studies. It stores descriptive metadata, repository
    context, variant information, and the collection of configured trade-study
    workflows.

    Notes
    -----
    Supported trade-study objects currently include parameter sweeps,
    optimizations, uncertainty analyses, and local sensitivity studies.

    """

    log = get_log(log, object_name="control")

    def __init__(
        self,
        title: str = "Default Title",
        author: str = "Default Author",
        description: str = "Default Description",
        repo_root: str = None,
    ) -> None:
        """Initialize a control definition.

        Parameters
        ----------
        title : str, optional
            Study title.
        author : str, optional
            Study author.
        description : str, optional
            Study description.
        repo_root : str, optional
            Root directory of the study repository. This value is stored in control
            metadata written to study JSON files. It may be set directly or populated
            automatically when a study is generated through :class:`Variants` with an
            assigned repository root.

        Notes
        -----
        Construction initializes an empty trade-study registry and clears the
        variant name.

        """
        self.title = title
        self.author = author
        self.description = description
        self.repo_root = repo_root if repo_root is not None else ""
        self.trade_studies = {}
        self.log = Control.log
        self.variant_name = ""

    def _set_repo_root(self, repo_root: str) -> None:
        """Set the study repository root directory.

        Parameters
        ----------
        repo_root : str
            Root directory of the study repository. This value is stored in control
            metadata written to study JSON files. It may be assigned directly or
            populated automatically during variant generation.

        Returns
        -------
        None

        """
        self.repo_root = repo_root

    def _set_variant_name(self, variant_name: str) -> None:
        """Set the current variant name.

        Parameters
        ----------
        variant_name : str
            Variant name associated with this control definition.

        Returns
        -------
        None

        """
        self.variant_name = variant_name

    def add_metadata(
        self,
        title: str = "Default Title",
        author: str = "Default Author",
        description: str = "Default Description",
    ) -> None:
        """Update study metadata.

        Parameters
        ----------
        title : str, optional
            Study title.
        author : str, optional
            Study author.
        description : str, optional
            Study description.

        Returns
        -------
        None

        """
        self.title = title
        self.author = author
        self.description = description

    def add_tradestudy(
        self, trade_name: str, trade: Union[ParameterSweep, Optimization,
                                            BasicAleatory,RegressionPCE,
                                            GaussQuadraturePCE,SobolSensitivity,
                                            BasicEpistemic, BasicMixedIVP,
                                            BasicSOP,MixedRegressionPCE,
                                            AleatoryCSVPoints,EpistemicCSVPoints,
                                            MixedCSVPoints],
    ) -> None:
        """Register a named trade study.

        Parameters
        ----------
        trade_name : str
            Trade-study name.
        trade : ParameterSweep or Optimization or LocalSensitivity or UQ workflow
            Trade-study definition to register. Supported UQ workflows include
            aleatory, epistemic, mixed, and CSV-backed workflow classes.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``trade_name`` is ``"Default State"``, which is reserved.

        """
        if trade_name == DEFAULT_TRADE_NAME:
            self.log.error(
                'Trade study name cannot be "Default State", as this is reserved.'
            )
            raise ValueError(
                'Trade study name cannot be "Default State", as this is reserved.'
            )
        self.log.debug(f"add trade: {trade_name}")
        self.trade_studies[trade_name] = trade

    def _to_dict(self) -> dict:
        """Serialize the control definition to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing metadata, repository context,
            variant name, and serialized trade studies.

        """
        self.log.debug("write to dict")
        return {
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "repo_root": self.repo_root,
            "variant_name": self.variant_name,
            "trade_studies": {k: v._to_dict() for k, v in self.trade_studies.items()},
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Control":
        """Reconstruct a control definition from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized control dictionary, typically produced by :meth:`_to_dict`.

        Returns
        -------
        Control
            Reconstructed control definition with restored metadata, variant name,
            and trade studies.

        Raises
        ------
        ValueError
            If a serialized trade-study type is not recognized.

        """
        cls.log.debug("control from dict")
        control = cls(
            title=data["title"],
            author=data["author"],
            description=data["description"],
            repo_root=data.get("repo_root", None),
        )

        control._set_variant_name(data.get("variant_name", ""))

        for k, v in data["trade_studies"].items():
            analysis_type=v["type"]
            if analysis_type not in trade_type_to_class:
                raise ValueError(f"Unknown trade study type: {analysis_type}")
            trade=trade_type_to_class[analysis_type]._from_dict(v)
            control.add_tradestudy(k, trade)

        return control

trade_type_to_class = {
    ParameterSweep.__name__: ParameterSweep,
    Optimization.__name__: Optimization,
    LocalSensitivity.__name__:LocalSensitivity,
    BasicAleatory.__name__:BasicAleatory,
    RegressionPCE.__name__:RegressionPCE,
    GaussQuadraturePCE.__name__:GaussQuadraturePCE,
    SobolSensitivity.__name__:SobolSensitivity,
    BasicEpistemic.__name__:BasicEpistemic,
    BasicMixedIVP.__name__: BasicMixedIVP,
    BasicSOP.__name__:BasicSOP,
    MixedRegressionPCE.__name__:MixedRegressionPCE,
    AleatoryCSVPoints.__name__:AleatoryCSVPoints,
    EpistemicCSVPoints.__name__:EpistemicCSVPoints,
    MixedCSVPoints.__name__:MixedCSVPoints
}
if __name__ == "__main__":
    opt = Optimization()
    opt.add_design_var(name="wing_span", lower=10.0, upper=15.0)
    opt.add_design_var(name="wing_area", lower=20.0, upper=30.0)
    opt.add_constraint("drag", lower=0.1, upper=0.5)
    opt.add_objective("weight")

    param_sweep = ParameterSweep(
        param_name_to_value_list={
            "aspect_ratio": [10.0, 12.5, 15.0],
        },
        child=opt,
    )
    control = Control(
        title="Vehicle Control", author="Tom", description="Control for vehicle design"
    )
    control.add_tradestudy("aspect_ratio_parameter_sweep", param_sweep)

    # Print all trade studies and their details
    print("\nTrade studies in control:")
    for name, trade in control.trade_studies.items():
        print(f"Trade name: {name}")
        print(f"Type: {type(trade).__name__}")
        if isinstance(trade, ParameterSweep):
            print("Parameter sweep states:")
            for state in trade.get_states():
                print(state)
        print()

    # Add and print a direct optimization trade
    control.add_tradestudy("direct_optimization", opt)
    print("Direct optimization trade as dict:")
    print(vars(opt))
