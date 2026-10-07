from __future__ import annotations
import copy
import numpy as np


class LocalSensitivity:
    """Define the user-facing API for local one-at-a-time sensitivity analysis.

    A ``LocalSensitivity`` object specifies evaluation points and perturbation
    step sizes for selected variables. It generates baseline and perturbed
    states that can be used to estimate local sensitivities of study outputs.

    Parameters
    ----------
    step_size : float, optional
        Default perturbation step size used for variables that do not define
        their own step size.

    Notes
    -----
    Generated states include one baseline state plus one backward and one
    forward perturbed state for each registered variable.

    """

    def __init__(self, step_size=1e-6):
        """Initialize a local sensitivity study definition.

        Parameters
        ----------
        step_size : float, optional
            Default perturbation step size used for variables that do not define
            their own step size.

        """
        self.step_size = step_size
        self.variable_to_sensitivity_data = {}

    def add_perturbation_var(self, name, evaluation_point, step_size=None):
        """Register a variable for local sensitivity analysis.

        Parameters
        ----------
        name : str
            Variable name.
        evaluation_point : float
            Baseline value at which local sensitivity is evaluated.
        step_size : float, optional
            Variable-specific perturbation step size. If omitted, the global
            ``step_size`` defined on the analysis is used.

        Returns
        -------
        None

        """
        data = {
            "evaluation_point": evaluation_point,
            "step_size": step_size if step_size is not None else self.step_size,
        }
        self.variable_to_sensitivity_data[name] = data

    def get_states(self):
        """Generate baseline and perturbed states.

        Returns
        -------
        list of dict
            State dictionaries for the local sensitivity study. The first state is
            the baseline evaluation point with ``"optimization"`` set to False.
            For each registered variable, one backward-perturbed and one forward-
            perturbed state are added.

        Notes
        -----
        States are returned in this order: baseline, all backward perturbations,
        then all forward perturbations.

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
        """Serialize the local sensitivity definition to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the global step size and
            registered sensitivity variables.

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
        """Reconstruct a local sensitivity definition from a dictionary.

        Parameters
        ----------
        sensitivity_dict : dict
            Serialized local sensitivity dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        LocalSensitivity
            Reconstructed local sensitivity definition.

        """
        step_size = sensitivity_dict["global_step_size"]
        variable_to_sensitivity_data = sensitivity_dict["sensitivity_parameters"]
        sensitivity_obj = cls(step_size)

        for name, data in variable_to_sensitivity_data.items():
            sensitivity_obj.variable_to_sensitivity_data[name] = data

        return sensitivity_obj

