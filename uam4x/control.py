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
from uam4x.UQ import UQAnalysis,IVPAnalysis
from uam4x.local_sensitivity import LocalSensitivity

DEFAULT_TRADE_NAME = "Default State"


class Optimization:
    """Represent the optimization component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self) -> None:
        """Initialize object state and default configuration.

        Notes
        -----
        This method performs internal setup or validation logic.

        """
        self.des_vars = {}
        self.constraints = {}
        self.objective_name = ""
        self._mdao_driver_options = {"optimizer": "SLSQP"}
        self._optimizer_options = {}
        self.optimizer = "ScipyOptimizeDriver"

    def _to_dict(self):
        """Execute to dict.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        save_dict = {
            "type": self.__class__.__name__,
            "des_vars": self.des_vars,
            "constraints": self.constraints,
            "objective_name": self.objective_name,
            "driver_options": self._mdao_driver_options,
            "optimizer_options": self._optimizer_options,
            "optimizer": self.optimizer,
        }
        return save_dict

    def set_optimizer(self, name):
        """Set optimizer.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.

        """
        self.optimizer = name

    def add_driver_option(self, name, value):
        """Add driver option.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.
        value : Any
            Input value for `value`.

        """
        self._mdao_driver_options[name] = value

    def add_optimizer_option(self, name, value):
        """Add optimizer option.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.
        value : Any
            Input value for `value`.

        """
        self._optimizer_options[name] = value

    def get_desvar_names(self):
        """Return desvar names.

        Returns
        -------
        Any
            Requested value.

        """
        return list(self.des_vars.keys())

    @classmethod
    def _from_dict(cls, data: dict) -> "Optimization":
        """Execute from dict.

        Parameters
        ----------
        data : dict
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

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
        obj.optimizer = data["optimizer"]
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
        """Add design var.

        Parameters
        ----------
        name : str
            Identifier used by this operation.
        lower : np.ndarray | float | np.floating | None, optional
            Input value for `lower`.
        upper : np.ndarray | float | np.floating | None, optional
            Input value for `upper`.
        indices : list | None, optional
            Input value for `indices`.

        """
        self.check_upper_lower(
            name, "design variable", lower=lower, upper=upper, indices=indices
        )
        self.des_vars[name] = {"lower": lower, "upper": upper, "indices": indices}

    def add_constraint(
        self,
        name: str,
        lower: np.ndarray | float | np.floating | None = None,
        upper: np.ndarray | float | np.floating | None = None,
        indices=None,
    ):
        """Add constraint.

        Parameters
        ----------
        name : str
            Identifier used by this operation.
        lower : np.ndarray | float | np.floating | None, optional
            Input value for `lower`.
        upper : np.ndarray | float | np.floating | None, optional
            Input value for `upper`.
        indices : Any, optional
            Input value for `indices`.

        """
        self.check_upper_lower(
            name, const_or_dv="constraint", lower=lower, upper=upper, indices=indices
        )
        self.constraints[name] = {
            "lower": lower,
            "upper": upper,
            "indices": indices,
        }

    @staticmethod
    def check_upper_lower(
        name: str,
        const_or_dv,
        lower: np.ndarray | float | np.floating | None = None,
        upper: np.ndarray | float | np.floating | None = None,
        indices: list | np.ndarray | None = None,
    ):

        # At least one bound must be declared on a constraint

        """Validate upper lower.

        Parameters
        ----------
        name : str
            Identifier used by this operation.
        const_or_dv : Any
            Input value for `const_or_dv`.
        lower : np.ndarray | float | np.floating | None, optional
            Input value for `lower`.
        upper : np.ndarray | float | np.floating | None, optional
            Input value for `upper`.
        indices : list | np.ndarray | None, optional
            Input value for `indices`.

        """
        if lower is None and upper is None and const_or_dv == "constraint":
            raise ValueError(
                f"At least one constraint must be declared. The constraint {name}\n"
                f"does not have at least a lower or upper bound declared."
            )

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
        """Add objective.

        Parameters
        ----------
        name : str
            Identifier used by this operation.

        """
        self.objective_name = name


class ParameterSweep:
    """Represent the parametersweep component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    log = get_log(log, object_name="parameter sweep")

    def __init__(
        self,
        param_name_to_value_list: Dict[str, Any],
        child: Union["ParameterSweep", Optimization] = None,
    ) -> None:
        """Initialize object state and default configuration.

        Parameters
        ----------
        param_name_to_value_list : Dict[str, Any]
            Input value for `param_name_to_value_list`.
        child : Union['ParameterSweep', Optimization], optional
            Input value for `child`.

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

    def set_child(self, child: Union["ParameterSweep", Optimization]) -> None:
        """Set child.

        Parameters
        ----------
        child : Union['ParameterSweep', Optimization]
            Input value for `child`.

        """
        self.log.debug("set child")
        if not isinstance(child, (Optimization, ParameterSweep)):
            self.log.error("Child must be ParameterSweep or Optimization")
            raise TypeError("Child must be ParameterSweep or Optimization")
        self.child = child

    def get_optim_child(self):
        """Return optim child.

        Returns
        -------
        Any
            Requested value.

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
        """Return desvar names.

        Returns
        -------
        Any
            Requested value.

        """
        child = self.child
        if self.child is None:
            return []
        else:
            return child.get_desvar_names()

    def get_all_param_names(self):
        # assumes optimization cannot have child
        """Return all param names.

        Returns
        -------
        Any
            Requested value.

        """
        child = self.child
        params = set(self.param_name_to_value_list.keys())
        if isinstance(self.child, ParameterSweep):
            return params.union(child.get_all_param_names())
        else:
            return params

    def get_states(self) -> List[Dict[str, Any]]:
        """Return states.

        Returns
        -------
        Any
            Requested value.

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
        """Execute to dict.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        self.log.debug("write to dict")
        return {
            "type": "ParameterSweep",
            "param_name_to_value_list": self.param_name_to_value_list,
            "child": self.child._to_dict() if self.child else "",
        }

    @classmethod
    def _from_dict(cls, data):
        """Execute from dict.

        Parameters
        ----------
        data : Any
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

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
    """Represent the control component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    log = get_log(log, object_name="control")

    def __init__(
        self,
        title: str = "Default Title",
        author: str = "Default Author",
        description: str = "Default Description",
        repo_root: str = None,
    ) -> None:
        """Initialize object state and default configuration.

        Parameters
        ----------
        title : str, optional
            Input value for `title`.
        author : str, optional
            Input value for `author`.
        description : str, optional
            Input value for `description`.
        repo_root : str, optional
            Input value for `repo_root`.

        """
        self.title = title
        self.author = author
        self.description = description
        self.repo_root = repo_root if repo_root is not None else ""
        self.trade_studies = {}
        self.log = Control.log
        self.variant_name = ""

    def _set_repo_root(self, repo_root: str) -> None:
        """Set the Git repository root directory for the study object.

        Parameters
        ----------
        repo_root : str
            Absolute path to the root of the Git repository used
            for study data, variant generation, and version-controlled outputs.

        """
        self.repo_root = repo_root

    def _set_variant_name(self, variant_name: str) -> None:
        """Set variant name.

        Parameters
        ----------
        variant_name : str
            Identifier used by this operation.

        """
        self.variant_name = variant_name

    def add_metadata(
        self,
        title: str = "Default Title",
        author: str = "Default Author",
        description: str = "Default Description",
    ) -> None:
        """Add metadata.

        Parameters
        ----------
        title : str, optional
            Input value for `title`.
        author : str, optional
            Input value for `author`.
        description : str, optional
            Input value for `description`.

        """
        self.title = title
        self.author = author
        self.description = description

    def add_tradestudy(
        self, trade_name: str, trade: Union[ParameterSweep, Optimization, UQAnalysis,IVPAnalysis],
    ) -> None:
        """Add tradestudy.

        Parameters
        ----------
        trade_name : str
            Identifier used by this operation.
        trade : Union[ParameterSweep, Optimization, UQAnalysis]
            Input value for `trade`.

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
        """Execute to dict.

        Returns
        -------
        Any
            JSON-serializable representation.

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
        """Execute from dict.

        Parameters
        ----------
        data : dict
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

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
            if v["type"] == "ParameterSweep":
                trade = ParameterSweep._from_dict(v)
            elif v["type"] == "Optimization":
                trade = Optimization._from_dict(v)
            elif v["type"] == "UQAnalysis":
                trade = UQAnalysis._from_dict(v)
            elif v["type"] == "LocalSensitivity":
                trade = LocalSensitivity._from_dict(v)
            elif v["type"]=="IVPAnalysis":
                trade=IVPAnalysis._from_dict(v)
            else:
                raise ValueError(f"Unknown trade study type: {v['type']}")
            control.add_tradestudy(k, trade)

        return control


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
