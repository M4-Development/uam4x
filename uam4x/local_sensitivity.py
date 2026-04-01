from __future__ import annotations
import copy
import numpy as np
from uam4x.results import Tradestudy


class LocalSensitivity:
    """Represent the localsensitivity component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, step_size=1e-6):
        """Initialize object state and default configuration.

        Parameters
        ----------
        step_size : Any, optional
            Input value for `step_size`.

        """
        self.step_size = step_size
        self.variable_to_sensitivity_data = {}

    def add_perturbation_var(self, name, evaluation_point, step_size=None):
        """Add perturbation var.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.
        evaluation_point : Any
            Input value for `evaluation_point`.
        step_size : Any, optional
            Input value for `step_size`.

        """
        data = {
            "evaluation_point": evaluation_point,
            "step_size": step_size if step_size is not None else self.step_size,
        }
        self.variable_to_sensitivity_data[name] = data

    def get_states(self):
        """Return states.

        Returns
        -------
        Any
            Requested value.

        """
        baseline_state = {
            name: data["evaluation_point"]
            for name, data in self.variable_to_sensitivity_data.items()
        }
        baseline_state["optimization"] = False
        states = [baseline_state]
        b_states = []
        f_states = []

        for name, sensitivity_data in self.variable_to_sensitivity_data.items():
            local_b_state = copy.deepcopy(baseline_state)
            local_f_state = copy.deepcopy(baseline_state)
            local_b_state[name] = (
                sensitivity_data["evaluation_point"] - sensitivity_data["step_size"]
            )
            local_f_state[name] = (
                sensitivity_data["evaluation_point"] + sensitivity_data["step_size"]
            )

            b_states.append(local_b_state)
            f_states.append(local_f_state)

        states += b_states + f_states

        return states

    def _to_dict(self):
        """Execute to dict.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "type": self.__class__.__name__,
            "global_step_size": self.step_size,
            "sensitivity_parameters": {
                name: data for name, data in self.variable_to_sensitivity_data.items()
            },
        }

    @classmethod
    def _from_dict(cls, sensitivity_dict):
        """Execute from dict.

        Parameters
        ----------
        sensitivity_dict : Any
            Input value for `sensitivity_dict`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        step_size = sensitivity_dict["global_step_size"]
        variable_to_sensitivity_data = sensitivity_dict["sensitivity_parameters"]
        sensitivity_obj = cls(step_size)

        for name, data in variable_to_sensitivity_data.items():
            sensitivity_obj.variable_to_sensitivity_data[name] = data

        return sensitivity_obj



