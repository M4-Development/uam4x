from __future__ import annotations
from typing import Callable
import os
import json
import importlib.util
from uam4x.utils import (
    check_and_get_config,
    check_numpy_shape,
    is_scalar_type_or_value,
    is_numpy_array,
    normalize_shape,
    is_main_process,
    registry_writes_enabled,
    is_continuous_type_or_value,
    split_dotted_path_and_indices,
)
from collections.abc import Sequence
from numbers import Real
from itertools import product
from uam4x.log import get_log
from uam4x.utils import log, canonicalize_path
import numpy as np
import inspect
import hashlib
import copy

_plugin_registry = {}


class Group:
    """Represent a solution group containing steps, links, and subgroups.

    A ``Group`` organizes plugin steps and nested groups, stores variable links
    between them, and provides helper methods for serialization, plugin
    generation, link validation, and execution ordering.

    Notes
    -----
    ``Solution`` extends ``Group`` to represent the top-level workflow.

    """

    log = get_log(log, object_name="group")

    def __init__(self):
        """Initialize an empty solution group."""
        self.log = Group.log
        self.log.debug(f"initialize")
        self.links = []
        self.steps = {}
        self._execution_steps = {}
        self.cycle_solver = ""
        self._approx_totals = {"compute": False}
        self._is_cycle = False
        self.group_name = ""
        self.cycle_declared_step_order = []

    def set_cycle_solver(self, solver_name):
        """Set the solver used for cyclic group execution.

        Parameters
        ----------
        solver_name : str
            Solver identifier.

        Returns
        -------
        None

        """
        self.cycle_solver = solver_name

    def approx_totals(self, method="fd", step=10**-6, form="forward"):
        """Configure approximate total-derivative settings for the group.

        Parameters
        ----------
        method : str, optional
            Approximation method, such as finite difference.
        step : float, optional
            Step size used by the approximation method.
        form : str, optional
            Approximation form, such as ``"forward"``.

        Returns
        -------
        None

        """
        self._approx_totals = {
            "compute": True,
            "method": method,
            "step": step,
            "form": form,
        }

    def add_link(self, independent: str, dependent: str):
        """Add a variable link between two names in the group.

        Parameters
        ----------
        independent : str
            Source variable name.
        dependent : str
            Destination variable name.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``dependent`` includes bracketed indices. Only independent
            variables may use indexed access.

        Notes
        -----
        Duplicate links are ignored.

        """
        self.log.debug(
            f"add link. independent link: {independent}, dependent link: {dependent}"
        )
        # need slicing here dependent cannot
        if "[" in dependent:
            raise ValueError(
                rf"Only independent variables can have indices specified. The independent variable {dependent}\n"
                rf"has slices indices specified."
            )
        link = [independent, dependent]
        if link not in self.links:
            self.links.append(link)

    def remove_link(self, independent: str, dependent: str):
        """Remove a variable link from the group.

        Parameters
        ----------
        independent : str
            Source variable name.
        dependent : str
            Destination variable name.

        Returns
        -------
        None

        Notes
        -----
        If the link is not present, no action is taken.

        """
        self.log.debug(
            f"remove link. independent link: {independent}, dependent link: {dependent}"
        )
        link = [independent, dependent]
        if link in self.links:
            self.links.remove(link)

    def get_step_names(self):
        """Return step names in the group.

        Returns
        -------
        list of str
            Step names in insertion order.

        """
        self.log.debug(f"get step names")
        return [step_name for step_name in self.steps]

    def _make_step_info(self, step, plugin_options, plugin_hash=None):
        """Normalize step metadata for a plugin or subgroup.

        Parameters
        ----------
        step : type[Plugin] or Group
            Plugin class or subgroup object to normalize.
        plugin_options : dict
            Plugin options associated with the step.
        plugin_hash : str, optional
            Precomputed plugin source hash. If omitted and ``step`` is a plugin
            class, the hash is computed from the plugin source code.

        Returns
        -------
        dict or Group
            Normalized plugin metadata dictionary for plugin steps, or the
            original subgroup object for nested groups.

        Raises
        ------
        TypeError
            If ``step`` is neither a plugin class nor a :class:`Group`
            instance.

        """
        if isinstance(step, type) and issubclass(step, Plugin):

            if plugin_hash is None:
                src = inspect.getsource(step)
                h = hashlib.sha256()
                h.update(src.encode("utf-8"))
                plugin_hash = h.hexdigest()

            return {
                "plugin": step,
                "plugin_options": plugin_options,
                "plugin_hash": plugin_hash,
            }

        elif isinstance(step, Group):
            return step

        else:
            raise TypeError(f"Only plugin classes and group objects are allowed")

    def add_step(self, step, step_name=None, plugin_options: dict = None):
        """Add a plugin step or subgroup to the group.

        Parameters
        ----------
        step : type[Plugin] or Group
            Plugin class or subgroup object to add.
        step_name : str, optional
            Name used to register the step. If omitted, a default name of the
            form ``"stepN"`` is assigned.
        plugin_options : dict, optional
            Plugin options associated with the step.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``step`` is a class that is not a subclass of :class:`Plugin`.
        ValueError
            If ``step_name`` contains ``"["`` or ``"]"``.

        Notes
        -----
        If ``step`` is a subgroup, its ``group_name`` attribute is set to
        ``step_name``.

        """
        if plugin_options is None:
            plugin_options = {}
        self.log.debug(f"add step: {step_name}")
        if isinstance(step, type) and not issubclass(step, Plugin):
            self.log.error("add_step: plugin_class should be a callable (class).")
            raise TypeError("plugin_class should be a callable (class).")
        if step_name is not None and ("[" in step_name or "]" in step_name):
            self.log.error(f'steps cannot have the character "[" or "]" in them')
            raise ValueError(f'steps cannot have the character "[" or "]" in them')
        # plugin_name should be a string, not a class
        if step_name is None:
            step_name = "step" + str(len(self.steps) + 1)
        if isinstance(step, Group):
            step.group_name = step_name

        step_info = self._make_step_info(step, plugin_options)
        self.steps[step_name] = step_info

    def remove_step(self, step_name):
        """Remove a step from the group.

        Parameters
        ----------
        step_name : str
            Step name.

        Returns
        -------
        None

        Notes
        -----
        If ``step_name`` is not present, no action is taken.

        """
        self.log.debug(f"remove step: {step_name}")
        if step_name in self.steps:
            self.steps.pop(step_name)

    def edit_step(self, step_name, plugin_options: dict = None):
        """Update plugin options for an existing plugin step.

        Parameters
        ----------
        step_name : str
            Step name.
        plugin_options : dict, optional
            Replacement or merged plugin options.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``step_name`` refers to a subgroup instead of a plugin step.
        KeyError
            If ``step_name`` is not present in the group.

        """
        self.log.debug(f"edit step: {step_name}")
        if step_name in self.steps:
            step = self.steps[step_name]
            if isinstance(step, Group):
                raise ValueError(
                    f"Only plugins are editable at the Group level, sub groups"
                    "must be edited in their respective groups"
                )
            if plugin_options is not None:
                current_options = step["plugin_options"]
                for option_name, info in plugin_options.items():
                    current_options[option_name] = info
            else:
                step["plugin_options"] = {}
        else:
            raise KeyError(f"The step {step_name} is not a step")

    def add_plugin_option(self, step_name: str, option_name: str, option_value):
        """Add or overwrite one plugin option for a step.

        Parameters
        ----------
        step_name : str
            Step name.
        option_name : str
            Plugin option name.
        option_value : object
            Plugin option value.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``step_name`` refers to a subgroup instead of a plugin step.
        KeyError
            If ``step_name`` is not present in the group.

        """
        self.log.debug(f"add plugin option {option_name} to step {step_name}")
        if step_name in self.steps:
            step = self.steps[step_name]
            if isinstance(step, Group):
                raise ValueError(
                    f"Only plugins are editable at the Group level, sub groups"
                    "must be edited in their respective groups"
                )
            if option_name in step["plugin_options"]:
                print(
                    f'{option_name} already a plugin option. Overwriting value: {step["plugin_options"][option_name]}'
                )
            step["plugin_options"][option_name] = option_value
        else:
            raise KeyError(f"The step {step_name} is not a step")

    def remove_plugin_option(self, step_name: str, option_name: str):
        """Remove one plugin option from a step.

        Parameters
        ----------
        step_name : str
            Step name.
        option_name : str
            Plugin option name to remove.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``step_name`` refers to a subgroup instead of a plugin step.
        KeyError
            If ``step_name`` is not present in the group.

        """
        self.log.debug(f"remove plugin option {option_name} to step {step_name}")
        if step_name in self.steps:
            step = self.steps[step_name]
            if isinstance(step, Group):
                raise ValueError(
                    f"Only plugins are editable at the Group level, sub groups"
                    "must be edited in their respective groups"
                )
            step["plugin_options"].pop(option_name)
        else:
            raise KeyError(f"The step {step_name} is not a step")

    def _to_dict(self) -> dict:
        """Serialize the group to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing serialized steps, links,
            cycle solver configuration, approximate-total settings, group
            name, and declared cycle step order.

        Notes
        -----
        Plugin steps are serialized by plugin class name and plugin metadata.
        Nested groups are serialized recursively.

        """
        self.log.debug(f"write to dict")
        steps_serialized = {}
        for step_name, step_info in self.steps.items():
            if isinstance(step_info, dict):
                plugin_class = step_info["plugin"]
                plugin_name = plugin_class.__name__
                steps_serialized[step_name] = {
                    "plugin": plugin_name,
                    "plugin_options": step_info["plugin_options"],
                    "plugin_hash": step_info["plugin_hash"],
                    "specification": "plugin",
                }
            else:
                # if its a group instead
                steps_serialized[step_name] = step_info._to_dict()
                steps_serialized[step_name]["specification"] = "group"

        return {
            "steps": steps_serialized,
            "links": self.links,
            "cycle_solver": self.cycle_solver,
            "approx_totals": self._approx_totals,
            "group_name": self.group_name,
            "cycle_declared_step_order": self.cycle_declared_step_order,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Solution":
        """Reconstruct a group from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized group dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        Group
            Reconstructed group with restored steps, links, and cycle
            metadata.

        """
        group = cls()
        cls.log.debug("solution from dict")
        # Reconstruct steps, mapping plugin name to class if possible
        for step_name, step_data in data["steps"].items():
            if step_data["specification"] == "plugin":
                plugin_name = step_data["plugin"]
                step = get_plugin_class(plugin_name)
                plugin_options = step_data["plugin_options"]
                plugin_hash = step_data["plugin_hash"]
            elif step_data["specification"] == "group":
                step = Group._from_dict(step_data)
                plugin_options = {}
                plugin_hash = None

            step_info = group._make_step_info(step, plugin_options, plugin_hash)
            group.steps[step_name] = step_info

        group.links = data["links"]
        group.cycle_solver = data["cycle_solver"]
        group._approx_totals = data["approx_totals"]
        group.group_name = data["group_name"]
        group.cycle_declared_step_order = data["cycle_declared_step_order"]
        return group

    def generate_plugins(self):
        """Instantiate plugin objects for the group.

        Returns
        -------
        dict
            Nested dictionary mirroring the group structure. Plugin steps are
            replaced with instantiated :class:`Plugin` objects, and subgroup
            entries contain deeper dictionaries of instantiated plugins.

        """
        plugins = {}
        for step_name, data in self.steps.items():
            if isinstance(data, dict):
                plugin = data["plugin"]()
                plugin.setup_plugin(data["plugin_options"])
                plugins[step_name] = plugin
            elif isinstance(data, Group):
                group_plugs = data.generate_plugins()
                plugins[step_name] = group_plugs

        return plugins

    def _recurse_generated_plugins(self, plugins, plugin_dict):
        """Flatten a nested generated-plugin tree into a single dictionary.

        Parameters
        ----------
        plugins : dict
            Nested dictionary produced by :meth:`generate_plugins`.
        plugin_dict : dict
            Output dictionary populated in place with ``{step_name: plugin}``
            entries for leaf plugin objects.

        Returns
        -------
        None

        Notes
        -----
        Keys added to ``plugin_dict`` are leaf step names, not fully qualified
        group paths.

        """

        for step_name, data in plugins.items():
            if isinstance(data, Plugin):
                plugin_dict[step_name] = data
            elif isinstance(data, dict):
                self._recurse_generated_plugins(data, plugin_dict)

    def _get_all_plugin_names_to_data(self, head_dict, group_name):
        """Collect fully qualified plugin metadata for this group tree.

        Parameters
        ----------
        head_dict : dict
            Output dictionary populated in place with fully qualified plugin
            names mapped to plugin metadata.
        group_name : str
            Dotted prefix used for this group in the recursive traversal.

        Returns
        -------
        None

        Notes
        -----
        Fully qualified names are built as ``"<group_name>.<step_name>"``.

        """
        for name, data in self.steps.items():
            if isinstance(data, dict):
                head_dict[f"{group_name}.{name}"] = {
                    "plugin": data["plugin"],
                    "plugin_hash": data["plugin_hash"],
                    "plugin_options": data["plugin_options"],
                }
            elif isinstance(data, Group):
                data._get_all_plugin_names_to_data(head_dict, f"{group_name}.{name}")

    def sort_steps(self):
        """Compute execution order for steps in the group.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a cycle is detected and no cycle solver or declared cycle step
            order is available, or if the declared cycle step order length does
            not match the number of steps.

        Notes
        -----
        Step ordering is derived from declared links using a topological sort.
        If a cycle is detected, the group switches to cycle mode and uses
        ``cycle_declared_step_order``.

        """
        self._execution_steps = {}
        # generate a list of dependent links
        for name, data in self.steps.items():
            if isinstance(data, Group):
                data.sort_steps()
        # may want to handle the case where there is a single plugin/group
        # within a group (either a raise or do nothing?)
        reduced_links = []
        counter = 0
        step_to_location = {}
        location_to_step = {}
        steps = set()
        for independent, dependent in self.links:
            independent = independent.split(".")[0]
            dependent = dependent.split(".")[0]
            steps.add(independent)
            steps.add(dependent)
            pair = [independent, dependent]
            if pair not in reduced_links:
                reduced_links.append(pair)
            if independent not in step_to_location:
                step_to_location[independent] = counter
                location_to_step[counter] = independent
                counter += 1
            if dependent not in step_to_location:
                step_to_location[dependent] = counter
                location_to_step[counter] = dependent
                counter += 1

        adj = [[] for _ in range(len(steps))]
        in_degree = [0] * len(steps)
        for independent, dependent in reduced_links:
            adj[step_to_location[independent]].append(step_to_location[dependent])
            in_degree[step_to_location[dependent]] += 1

        sources = []
        for loc in step_to_location.values():
            if in_degree[loc] == 0:
                sources.append(loc)

        sorted_steps = []
        while sources:
            source = sources.pop()
            sorted_steps.append(source)
            for check_node in adj[source]:
                in_degree[check_node] -= 1
                if in_degree[check_node] == 0:
                    sources.append(check_node)

        if len(sorted_steps) != len(steps):
            print(
                f"Cycle detected in group {self.group_name}, using declared step order in that group"
            )
            self._execution_steps = copy.deepcopy(self.steps)
            self._is_cycle = True
            if not self.cycle_solver:
                raise ValueError(f"Cycle solver must be selected if cycle is used")
            if not self.cycle_declared_step_order:
                raise ValueError(f"Groups with cycles must have user provided order")
            if not len(self.cycle_declared_step_order) == len(self.steps.keys()):
                raise ValueError(
                    f"The step order must have the same length as the steps"
                )
            for name in self.cycle_declared_step_order:
                self._execution_steps[name] = self.steps[name]
        else:
            for name, data in self.steps.items():
                if name not in list(step_to_location.keys()):
                    self._execution_steps[name] = data
            for step_order in sorted_steps:
                step = location_to_step[step_order]
                if step != "vehicles":
                    self._execution_steps[step] = self.steps[step]

    def _check_links_to_plugins(self, plugins):
        """Validate that linked variables are declared by plugin steps.

        Parameters
        ----------
        plugins : dict
            Nested plugin-instance mapping produced by
            :meth:`generate_plugins`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a linked plugin input or output name was not declared by the
            plugin.

        Notes
        -----
        Validation is performed recursively for nested groups.

        """
        for step_name in self._execution_steps:
            # handle plugin case
            if isinstance(self._execution_steps[step_name], dict):
                input_names, output_names = self.get_plugin_input_and_output_names(
                    step_name
                )
                # check coherance of declared I/O from plugin and from branching script
                for input_name in input_names:
                    name = ".".join(input_name.split(".")[1:])
                    if name not in plugins[step_name].plugin_input_names:
                        self.log.error(
                            f'The input {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                            f"Check your plugin's setup() step for I/O declaration"
                        )
                        raise ValueError(
                            f'The input {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                            f"Check your plugin's setup() step for I/O declaration"
                        )
                for output_name in output_names:
                    name = ".".join(output_name.split(".")[1:])
                    if "[" in name:
                        # any time where indexes are specified in the link, this is required to be
                        # an index specification, nothing to do with the variable name.
                        name, _ = split_dotted_path_and_indices(name)
                    if name not in plugins[step_name].plugin_output_names:
                        self.log.error(
                            f'The output {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                            f"Check your plugin's setup() step for I/O declaration"
                        )
                        raise ValueError(
                            f'The output {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                            f"Check your plugin's setup() step for I/O declaration"
                        )

            elif isinstance(self._execution_steps[step_name], Group):
                self._execution_steps[step_name]._check_links_to_plugins(
                    plugins[step_name]
                )

    def get_plugin_input_and_output_names(self, step_name):
        """Return linked input and output names for one step.

        Parameters
        ----------
        step_name : str
            Step name.

        Returns
        -------
        input_names : set of str
            Linked variable names entering ``step_name``.
        output_names : set of str
            Linked variable names produced by ``step_name``.

        Notes
        -----
        Bracketed indices on independent variable links are removed before the
        names are reported.

        """
        input_names = []
        output_names = []
        for independent, dependent in self.links:
            # remove indices from index based links
            if "[" in independent:
                independent, _ = split_dotted_path_and_indices(independent)
            if dependent.split(".")[0] == step_name:
                input_names.append(dependent)
            if independent.split(".")[0] == step_name:
                output_names.append(independent)
        return set(input_names), set(output_names)

    def get_all_groups(self, name_to_group, parent_name):
        """Collect nested groups in the current group tree.

        Parameters
        ----------
        name_to_group : dict
            Output dictionary populated in place with fully qualified group
            names mapped to :class:`Group` objects.
        parent_name : str
            Dotted prefix used for this group in the recursive traversal.

        Returns
        -------
        dict
            Updated ``name_to_group`` dictionary.

        """
        for name, step in self.steps.items():
            if isinstance(step, Group):
                full_name = parent_name + "." + name
                name_to_group[full_name] = step
                step.get_all_groups(name_to_group, full_name)
        return name_to_group

    def get_var_type(self, name, plugins):
        """Return declared metadata type for a variable name.

        Parameters
        ----------
        name : str
            Fully qualified variable name.
        plugins : dict
            Plugin-instance mapping used to resolve the variable.

        Returns
        -------
        type
            Declared Python type for the variable.

        """
        return self.get_var_data(name, plugins)["type"]

    def get_var_value(self, name, plugins):
        """Return declared default value for a variable name.

        Parameters
        ----------
        name : str
            Fully qualified variable name.
        plugins : dict
            Plugin-instance mapping used to resolve the variable.

        Returns
        -------
        object
            Declared variable value.

        """
        return self.get_var_data(name, plugins)["value"]

    def get_var_shape(self, name, plugins):
        """Return declared shape metadata for a variable name.

        Parameters
        ----------
        name : str
            Fully qualified variable name.
        plugins : dict
            Plugin-instance mapping used to resolve the variable.

        Returns
        -------
        tuple or None
            Declared variable shape metadata.

        """
        return self.get_var_data(name, plugins)["shape"]

    def get_var_data(self, name, plugins):
        """Return declared metadata for a variable name.

        Parameters
        ----------
        name : str
            Fully qualified variable name.
        plugins : dict
            Plugin-instance mapping used to resolve the variable.

        Returns
        -------
        dict
            Variable metadata dictionary containing at least ``"type"``,
            ``"value"``, and ``"shape"``.

        Raises
        ------
        KeyError
            If the variable name cannot be resolved from the plugin/group tree.

        """
        split_name = name.split(".")
        step_name = split_name[0]
        rest_name = ".".join(split_name[1:])
        step = plugins[step_name]
        if isinstance(step, Plugin):
            if rest_name in step.plugin_input_names:
                var_type = step.plugin_input_names[rest_name]["type"]
                var_value = step.plugin_input_names[rest_name]["value"]
                var_shape = step.plugin_input_names[rest_name]["shape"]

            elif rest_name in step.plugin_output_names:
                var_type = step.plugin_output_names[rest_name]["type"]
                var_value = step.plugin_output_names[rest_name]["value"]
                var_shape = step.plugin_output_names[rest_name]["shape"]
            else:
                raise KeyError(
                    f"Provided path is not an input/output to the plugin {step_name}"
                )
            var_data = {"value": var_value, "type": var_type, "shape": var_shape}
            return var_data
        elif isinstance(step, dict):
            local_plugins = plugins[step_name]
            return self.get_var_data(rest_name, local_plugins)

    def _get_global_execution_order_plugins(
        self, sol_name, plugins, global_name_to_plugin
    ):
        """Collect plugins in global execution order.

        Parameters
        ----------
        sol_name : str
            Dotted prefix for this group in the global solution tree.
        plugins : dict
            Nested plugin-instance mapping produced by
            :meth:`generate_plugins`.
        global_name_to_plugin : dict
            Output dictionary populated in place with fully qualified step
            names mapped to plugin instances.

        Returns
        -------
        None

        Notes
        -----
        Fully qualified names are built as ``"<sol_name>.<step_name>"``.

        """
        for name, step in self._execution_steps.items():
            if isinstance(step, Group):
                full_name = sol_name + "." + name
                local_plugs = plugins[name]
                step._get_global_execution_order_plugins(
                    full_name, local_plugs, global_name_to_plugin
                )
            else:
                global_name_to_plugin[sol_name + "." + name] = plugins[name]

    def _build_links(self, base_name):
        """Build fully qualified links for the group tree.

        Parameters
        ----------
        base_name : str
            Dotted prefix for this group in the global solution tree.

        Returns
        -------
        list of list[str]
            Fully qualified ``[output_name, input_name]`` link pairs for this
            group and all nested groups.

        Notes
        -----
        Output links beginning with ``"vehicles"`` are treated as already
        global and are not prefixed with ``base_name``.

        """
        global_links = []
        for output, input in self.links:
            if output.split(".")[0] != "vehicles":
                output = base_name + "." + output
            input = base_name + "." + input
            global_links.append([output, input])
        for name, step in self._execution_steps.items():
            if isinstance(step, Group):
                new_base_name = base_name + "." + name
                global_links += step._build_links(new_base_name)
        return global_links

    def declare_cycle_step_order(self, names):
        """Declare explicit step order for cyclic group execution.

        Parameters
        ----------
        names : sequence of str
            Step names in the order they should be executed when the group
            contains a cycle.

        Returns
        -------
        None

        """
        self.cycle_declared_step_order = names

    def get_all_steps(self):
        """Get all steps in dotted paths"""
        paths = []
        for step, value in self.steps.items():
            if isinstance(value, Group):
                paths.extend([f"{step}.{path}" for path in value.get_all_steps()])
            else:
                paths.append(step)
        return paths

    def get_all_steps_dict(self, prefix=""):
        result = {}
        current_key = prefix or "solution"

        # Use the simpler function
        result[current_key] = self.get_all_steps()

        for step, value in self.steps.items():
            if isinstance(value, Group):
                full_path = f"{prefix}.{step}" if prefix else step
                result.update(value.get_all_steps_dict(full_path))

        return result

class Solution(Group):
    """Represent the top-level executable solution workflow.

    ``Solution`` extends :class:`Group` with tracked outputs and top-level
    helpers for plugin discovery, group traversal, link validation, vehicle
    parameter usage, and global execution-order construction.

    """

    log = get_log(log, object_name="solution")

    def __init__(self) -> None:
        """Initialize an empty top-level solution workflow."""
        super().__init__()
        self.tracked_outputs = []
        self.log = Solution.log
        self.log.debug(f"initialize")
        self.step_name = "solution"

    def add_tracked_output(self, output: str):
        """Add a tracked output to the solution.

        Parameters
        ----------
        output : str
            Fully qualified output variable name to track.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If ``output`` is already being tracked.

        """
        self.log.debug(f"add tracked output: {output}")
        if output in self.tracked_outputs:
            self.log.error(f"output already being tracked: {output}")
            raise RuntimeError(f"output already being tracked: {output}")
        self.tracked_outputs.append(output)

    def remove_tracked_output(self, output: str):
        """Remove a tracked output from the solution.

        Parameters
        ----------
        output : str
            Fully qualified output variable name to stop tracking.

        Returns
        -------
        None

        Notes
        -----
        If ``output`` is not currently tracked, no action is taken.

        """
        self.log.debug(f"remove tracked output: {output}")
        if output in self.tracked_outputs:
            self.tracked_outputs.remove(output)

    def _to_dict(self) -> dict:
        # Serialize steps, converting plugin class to its name
        """Serialize the solution to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the serialized group data and
            tracked outputs.

        """
        self.log.debug(f"write to dict")
        steps_serialized = {}
        for step_name, step_info in self.steps.items():
            if isinstance(step_info, dict):
                plugin_class = step_info["plugin"]
                plugin_name = plugin_class.__name__
                steps_serialized[step_name] = {
                    "plugin": plugin_name,
                    "plugin_options": step_info["plugin_options"],
                    "plugin_hash": step_info["plugin_hash"],
                    "specification": "plugin",
                }
            else:
                # if its a group instead
                steps_serialized[step_name] = step_info._to_dict()
                steps_serialized[step_name]["specification"] = "group"

        return {
            "steps": steps_serialized,
            "links": self.links,
            "tracked_outputs": self.tracked_outputs,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Solution":
        """Reconstruct a solution from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized solution dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        Solution
            Reconstructed solution with restored steps, links, and tracked
            outputs.

        """
        sol = cls()
        cls.log.debug("solution from dict")
        # Reconstruct steps, mapping plugin name to class if possible
        for step_name, step_data in data["steps"].items():
            if step_data["specification"] == "plugin":
                plugin_name = step_data["plugin"]
                step = get_plugin_class(plugin_name)
                plugin_options = step_data["plugin_options"]
                plugin_hash = step_data["plugin_hash"]
            elif step_data["specification"] == "group":

                step = Group._from_dict(step_data)
                plugin_options = {}
                plugin_hash = None

            step_info = sol._make_step_info(step, plugin_options, plugin_hash)
            sol.steps[step_name] = step_info

        sol.tracked_outputs = data["tracked_outputs"]
        sol.links = data["links"]
        return sol

    def get_all_plugin_names_to_hashes(self):
        """Return plugin hashes for all steps in the solution.

        Returns
        -------
        dict
            Mapping from fully qualified plugin step names to plugin source
            hashes.

        """
        plugins_names_to_hashes = {}
        plugins_name_to_data = self.get_all_plugin_names_to_data()
        for name, data in plugins_name_to_data.items():
            plugins_names_to_hashes[name] = data["plugin_hash"]
        return plugins_names_to_hashes

    def get_all_plugin_names_to_classes(self):
        """Return plugin classes for all steps in the solution.

        Returns
        -------
        dict
            Mapping from fully qualified plugin step names to plugin classes.

        """
        plugins_names_to_classes = {}
        plugins_name_to_data = self.get_all_plugin_names_to_data()
        for name, data in plugins_name_to_data.items():
            plugins_names_to_classes[name] = data["plugin"]
        return plugins_names_to_classes

    def get_all_steps_of_type(self, plugin_type):
        """Get all plugins of a certain type"""
        steps_to_type = self.get_all_plugin_names_to_classes()
        plugins = []
        for name, plugin in steps_to_type.items():
            if plugin == plugin_type:
                plugins.append(name)

        return plugins

    def get_all_plugin_names_to_options(self):
        """Return plugin options for all steps in the solution.

        Returns
        -------
        dict
            Mapping from fully qualified plugin step names to plugin option
            dictionaries.

        """
        plugins_names_to_options = {}
        plugins_name_to_data = self.get_all_plugin_names_to_data()
        for name, data in plugins_name_to_data.items():
            plugins_names_to_options[name] = data["plugin_options"]
        return plugins_names_to_options

    def get_all_plugin_names_to_data(self):
        """Return plugin metadata for all steps in the solution.

        Returns
        -------
        dict
            Mapping from fully qualified plugin step names to dictionaries
            containing plugin class, plugin hash, and plugin options.

        """
        plugins_name_to_data = {}
        for name, data in self.steps.items():
            if isinstance(data, dict):
                plugins_name_to_data[name] = {
                    "plugin": data["plugin"],
                    "plugin_hash": data["plugin_hash"],
                    "plugin_options": data["plugin_options"],
                }
            elif isinstance(data, Group):
                data._get_all_plugin_names_to_data(plugins_name_to_data, name)

        return plugins_name_to_data

    def get_plugin_input_and_output_names(self, step_name):
        """Return linked input and output names for one top-level step.

        Parameters
        ----------
        step_name : str
            Top-level step name.

        Returns
        -------
        input_names : set of str
            Linked variable names entering ``step_name``.
        output_names : set of str
            Linked variable names produced by ``step_name``, including tracked
            outputs originating from the step.

        Notes
        -----
        Bracketed indices on independent variable links are removed before the
        names are reported.

        """
        input_names = []
        output_names = []
        for independent, dependent in self.links:
            # remove indices from index based links
            if "[" in independent:
                independent, _ = split_dotted_path_and_indices(independent)
            if dependent.split(".")[0] == step_name:
                input_names.append(dependent)
            if independent.split(".")[0] == step_name:
                output_names.append(independent)
        for independent in self.tracked_outputs:
            if independent.split(".")[0] == step_name:
                output_names.append(independent)
        return set(input_names), set(output_names)

    def get_all_groups(self):
        """Return all groups in the solution tree.

        Returns
        -------
        dict
            Mapping from fully qualified group names to :class:`Group`
            objects. The returned dictionary always includes ``"solution"``
            mapped to the top-level solution itself.

        """
        name_to_group = {"solution": self}
        for name, step in self.steps.items():
            if isinstance(step, Group):
                full_name = "solution" + "." + name
                name_to_group[full_name] = step
                step.get_all_groups(name_to_group, full_name)
        return name_to_group

    def get_group(self, dotted_path):
        """Return a group by dotted path.

        Parameters
        ----------
        dotted_path : str
            Group path. Paths may be given either relative to the solution root
            or with an explicit ``"solution."`` prefix.

        Returns
        -------
        Group
            Group resolved from ``dotted_path``.

        """
        split_path = dotted_path.split(".")
        groups = self.get_all_groups()
        if split_path[0] != "solution":
            dotted_path = "solution." + dotted_path
        return groups[dotted_path]

    def recurse_generated_plugins(self, plugins):
        """Return a flat mapping of generated plugin instances by step name.

        Parameters
        ----------
        plugins : dict
            Nested plugin structure returned by :meth:`generate_plugins`.

        Returns
        -------
        dict
            Dictionary mapping each discovered step name to its instantiated
            :class:`Plugin` object.

        Notes
        -----
        The traversal delegates recursive subgroup handling to
        :meth:`Group._recurse_generated_plugins`.

        """
        plugin_name_to_plugin_object = {}
        for step_name, data in plugins.items():
            if isinstance(data, Plugin):
                plugin_name_to_plugin_object[step_name] = data
            elif isinstance(data, dict):
                self._recurse_generated_plugins(data, plugin_name_to_plugin_object)

        return plugin_name_to_plugin_object

    def recurse_plugin_dict(self, split_name, plugins):
        """Resolve nested plugin mappings from a split path.

        Parameters
        ----------
        split_name : sequence of str
            Sequence of nested group or step names.
        plugins : dict
            Nested plugin-instance mapping.

        Returns
        -------
        object
            Nested plugin dictionary or plugin instance resolved from
            ``split_name``.

        """
        # last term in the initial string (is a group)
        # then plugins must be of form last_group:{data}
        if len(split_name) == 1:
            return plugins[split_name[0]]
        else:
            start = split_name[0]
            local_plug = plugins[start]
            return self.recurse_plugin_dict(split_name[1:], local_plug)

    def _check_link_types(self, plugins, study):
        """Validate type compatibility of all declared links.

        Parameters
        ----------
        plugins : dict
            Nested plugin-instance mapping produced by
            :meth:`generate_plugins`.
        study : Study
            Study used to resolve vehicle parameter types for links beginning
            with ``"vehicles"``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If indexed access is used on a non-array output link.
        TypeError
            If one or more link source and destination types are incompatible.

        Notes
        -----
        Validation is performed across all groups in the solution tree.

        """
        groups = self.get_all_groups()
        bad_links = []
        for group_name, group in groups.items():
            split_name = group_name.split(".")
            if split_name[0] == "solution" and len(split_name) == 1:
                local_plugins = plugins
            else:
                split_name = split_name[1:]
                local_plugins = self.recurse_plugin_dict(split_name, plugins)
            for output_link, input_link in group.links:
                indexes_specified = False

                if "[" in output_link:
                    output_link, _ = split_dotted_path_and_indices(output_link)
                    indexes_specified = True
                if output_link.split(".")[0] == "vehicles":
                    output_type = study.get_param_type(output_link)
                else:
                    output_type = self.get_var_type(output_link, local_plugins)

                input_type = self.get_var_type(input_link, local_plugins)
                if indexes_specified and output_type != np.ndarray:
                    raise ValueError(
                        f"Indexes may only be specified for arrays. The link: {input_link} is not valid"
                    )
                if (
                    indexes_specified
                    and output_type == np.ndarray
                    and is_continuous_type_or_value(input_type)
                    and input_type != np.ndarray
                ):
                    pass
                elif output_type != input_type:
                    bad_links.append((output_link, output_type, input_link, input_type))

        if bad_links:
            msg = (
                f"Link types must match, the following links have different types:\n"
                + "\n".join(
                    f"{out_}:{out_type}->{in_}:{in_type}"
                    for out_, out_type, in_, in_type in bad_links
                )
            )
            self.log.error(msg)
            raise TypeError(msg)

    def get_all_used_vehicle_params(self):
        """Return all vehicle parameter links used by the solution.

        Returns
        -------
        list of str
            Vehicle parameter paths referenced as independent link sources
            across all groups in the solution tree.

        """
        groups = self.get_all_groups()
        all_used_vehicle_params = []
        for _, group in groups.items():
            for independent, _ in group.links:
                independent_group = independent.split(".")[0]
                if independent_group == "vehicles":
                    all_used_vehicle_params.append(independent)
        return all_used_vehicle_params

    def _check_optimization_discrete(self, plugins):
        """Validate that optimization plugin links are continuous-valued.

        Parameters
        ----------
        plugins : dict
            Nested plugin-instance mapping produced by
            :meth:`generate_plugins`.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If a non-vehicle output link between plugins has a
            non-continuous type.

        Notes
        -----
        Vehicle links are excluded from this check.

        """
        groups = self.get_all_groups()
        bad_types = {}
        for group_name, group in groups.items():
            split_name = group_name.split(".")
            if split_name[0] == "solution" and len(split_name) == 1:
                local_plugins = plugins
            else:
                split_name = split_name[1:]
                local_plugins = self.recurse_plugin_dict(split_name, plugins)
            for output_link, input_link in group.links:
                if output_link.split(".")[0] == "vehicles":
                    continue

                if "[" in output_link:
                    output_link, _ = split_dotted_path_and_indices(output_link)

                output_type = self.get_var_type(output_link, local_plugins)

                if not is_continuous_type_or_value(output_type):
                    bad_types[output_link] = output_type
        if bad_types:
            bad_links = [
                f"{output_link}: {output_type}"
                for output_link, output_type in bad_types.items()
            ]
            detail = "\n".join(f"  - {line}" for line in bad_links)
            msg = (
                "Links between optimization components must be continuous (of type float)\n"
                "Discrete variables linked between plugins:\n"
                f"{detail}"
            )
            self.log.error(msg)
            raise TypeError(msg)

    def get_global_execution_order_plugins(self, plugins):
        """Return plugins in global execution order.

        Parameters
        ----------
        plugins : dict
            Nested plugin-instance mapping produced by
            :meth:`generate_plugins`.

        Returns
        -------
        dict
            Mapping from fully qualified step names to plugin instances in
            global execution order.

        """
        sol_name = "solution"
        global_plugin_name_to_plugin = {}
        for name, step in self._execution_steps.items():
            if isinstance(step, Group):
                full_name = sol_name + "." + name
                local_plugs = plugins[name]
                step._get_global_execution_order_plugins(
                    full_name, local_plugs, global_plugin_name_to_plugin
                )
            else:
                global_plugin_name_to_plugin[sol_name + "." + name] = plugins[name]

        return global_plugin_name_to_plugin

    def _build_global_links(self):
        """Build fully qualified links for the entire solution tree.

        Returns
        -------
        list of list[str]
            Fully qualified ``[output_name, input_name]`` link pairs for the
            top-level solution and all nested groups.

        """
        global_links = []
        base_name = "solution"
        for output, input in self.links:
            if output.split(".")[0] != "vehicles":
                output = base_name + "." + output
            input = base_name + "." + input
            global_links.append([output, input])
        for name, step in self._execution_steps.items():
            if isinstance(step, Group):
                child_name = base_name + "." + name
                global_links += step._build_links(child_name)
        return global_links

    def _check_link_indices(self, plugins, study):
        """Validate indexed source links against source and destination shapes.

        Parameters
        ----------
        plugins : dict
            Nested plugin-instance mapping produced by
            :meth:`generate_plugins`.
        study : Study
            Study used to resolve indexed vehicle parameter values and shapes.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If source indices are invalid or if the indexed source shape is
            incompatible with the destination shape.

        Notes
        -----
        This check applies only to links whose source specifies bracketed
        indices.

        """
        groups = self.get_all_groups()
        for group_name, group in groups.items():
            split_name = group_name.split(".")
            if split_name[0] == "solution" and len(split_name) == 1:
                local_plugins = plugins
            else:
                split_name = split_name[1:]
                local_plugins = self.recurse_plugin_dict(split_name, plugins)
            for output_link, input_link in group.links:
                indexes_specified = False

                if "[" in output_link:
                    cleaned_output_name, inds = split_dotted_path_and_indices(
                        output_link
                    )
                    if output_link.split(".")[0] == "vehicles":
                        try:
                            a = study.get_value(output_link)
                            src_shape = a.shape
                        except:
                            raise ValueError(
                                f"The indices in {output_link} provided are not a valid means to index\n"
                            )
                    else:
                        a = self.get_var_value(cleaned_output_name, local_plugins)
                        a_shape = self.get_var_shape(cleaned_output_name, local_plugins)
                        if a is not None:
                            try:
                                a[inds]
                                src_shape = a[inds].shape
                            except:
                                raise ValueError(
                                    f"Invalid indices or array. Array: {a} and inds: {inds}"
                                )
                        elif a_shape is not None:
                            try:
                                src_shape = np.ones(shape=a_shape)[inds].shape
                            except:
                                raise ValueError(
                                    f"Invalid inds or shape: shape: {a_shape}, inds:{inds}"
                                )
                        else:
                            # Note, this should already be caught for arrays in plugin declaration
                            raise ValueError(
                                "Developer: Check error catching capaibilities in plugin declaration"
                            )

                    b = self.get_var_value(input_link, local_plugins)
                    b_shape = self.get_var_shape(input_link, local_plugins)
                    b_type = self.get_var_type(input_link, local_plugins)

                    src_size = 1 if src_shape == () else int(np.prod(src_shape))
                    # float case
                    if src_size == 1 and b_type == float:
                        pass
                    elif b is not None:
                        if src_shape != b.shape:
                            raise ValueError(
                                f"The source indices must match the target shape. The input {output_link} with indices.\n"
                                f"The size of the source indices is {src_shape} and the destination has shape {b.shape}"
                            )

                    elif b_shape is not None:
                        if src_shape != b_shape:
                            raise ValueError(
                                f"The source indices must match the target shape. The input {output_link} with indices.\n"
                                f"The size of the source indices is {src_shape} and the destination has shape {b_shape}"
                            )


class Plugin:
    """Represent the base class for executable solution plugins.

    A ``Plugin`` declares its inputs, outputs, templates, executable metadata,
    and optional partial-derivative information. Concrete plugin subclasses
    implement setup and run behavior for one solution step.

    Notes
    -----
    Plugin subclasses typically override :meth:`setup_plugin` to process
    plugin options and :meth:`run_plugin` to perform execution.

    """

    log = get_log(log, object_name="Plugin base class")

    def __init__(self) -> None:
        """Initialize an empty plugin definition.

        Notes
        -----
        Construction initializes declared inputs, outputs, templates, executable
        metadata, and partial-derivative bookkeeping.

        """
        self._declare_partials = False
        self._check_partials_options = False
        self.partials_counter = 0
        self.initial_partials = []
        self.comprehensive_partials_dict = {}
        self.method_to_partial = {}
        self.plugin_input_names = {}
        self.plugin_output_names = {}
        self._plugin_templates = []
        self.log = Plugin.log
        self._template_to_checksum = {}
        self._executable_to_data = {}

    def __str__(self):
        """Return a human-readable string representation of the plugin.

        Returns
        -------
        str
            Plugin class name.

        """
        return f"{self.__class__.__name__}"

    def setup_plugin(self, plugin_options: dict):
        """Configure the plugin from step options.

        Parameters
        ----------
        plugin_options : dict
            Plugin options declared for the solution step.

        Returns
        -------
        None

        Notes
        -----
        Concrete plugin subclasses typically override this method.

        """
        self.log.debug("setup plugin")

    def run_plugin(self, links: dict, work_dir: str):
        """Execute the plugin for one solution step.

        Parameters
        ----------
        links : dict
            Mapping from declared plugin input names to values supplied by linked
            variables.
        work_dir : str or path-like
            Working directory for plugin execution.

        Returns
        -------
        None

        Notes
        -----
        Concrete plugin subclasses typically override this method.

        """
        self.log.debug("run plugin")

    def declare_partials_plugin(
        self,
        of: str | Sequence[str],
        wrt: str | Sequence[str],
        method="fd",
        step=None,
        form=None,
        rows: list | np.ndarray | None = None,
        cols: list | np.ndarray | None = None,
    ):
        """Declare plugin partial derivatives.

        Parameters
        ----------
        of : str or sequence of str
            Output variable name or names differentiated.
        wrt : str or sequence of str
            Input variable name or names with respect to which derivatives are
            taken.
        method : {"fd", "exact"}, optional
            Derivative method. ``"fd"`` declares finite-difference derivatives and
            ``"exact"`` declares analytic derivatives.
        step : real, optional
            Finite-difference step size. This is only valid when ``method="fd"``.
        form : {"forward", "backward", "central"}, optional
            Finite-difference form. This is only valid when ``method="fd"``.
        rows : list | np.ndarray | None, optional
            Row indices for sparse partial declarations.
        cols : list | np.ndarray | None, optional
            Column indices for sparse partial declarations.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``method`` is unsupported, if ``step`` or ``form`` are invalid for
            finite differences, or if ``step``/``form`` are provided when
            ``method="exact"``.

        Notes
        -----
        Strings are normalized to one-element lists. Wildcard declarations using
        ``"*"``, if supported by the caller, are expanded later by
        :meth:`build_partials`.

        """
        self.log.debug("declare partials plugin")
        # Check type of input, if string, wrap with list
        if type(of) == str:
            of = [of]
        # Check type of output, if string, wrap with list
        if type(wrt) == str:
            wrt = [wrt]

        # Error checking
        if method not in ["fd", "exact"]:
            self.log.error(
                f"Cannot use methods other than exact or fd, user used method: {method}"
            )
            raise ValueError(
                f"Cannot use methods other than exact or fd, user used method: {method}"
            )

        if method == "fd":
            if step is not None:
                if not isinstance(step, Real):
                    self.log.error(
                        f"Step size must be a real number. The step size {step} is not valid"
                    )
                    raise ValueError(
                        f"Step size must be a real number. The step size {step} is not valid"
                    )
                if step <= 0:
                    self.log.error(
                        f"Step size must be greater than zero. The step size {step} is not valid"
                    )
                    raise ValueError(
                        f"Step size must be greater than zero. The step size {step} is not valid"
                    )
            if form is not None:
                if form not in ["forward", "backward", "central"]:
                    self.log.error(
                        "Only allowed finite difference forms are: forward,\n"
                        f"backward, and central. The form: {form} is not valid"
                    )
                    raise ValueError(
                        "Only allowed finite difference forms are: forward,\n"
                        f"backward, and central. The form: {form} is not valid"
                    )
        else:
            if step is not None:
                self.log.error(
                    f"User specified {method} method, which does not take in a step.\n"
                    f"Instead, leave step blank or set it equal to None"
                )
                raise ValueError(
                    f"User specified {method} method, which does not take in a step.\n"
                    f"Instead, leave step blank or set it equal to None"
                )
            if form is not None:
                self.log.error(
                    f"User specified {method} method, which does not take in a form.\n"
                    f"Instead, leave step blank or set it equal to None"
                )
                raise ValueError(
                    f"User specified {method} method, which does not take in a form.\n"
                    f"Instead, leave step blank or set it equal to None"
                )

        self.partials_counter += 1
        self.initial_partials.append(
            {
                "of": of,
                "wrt": wrt,
                "method": method,
                "step": step,
                "form": form,
                "rows": rows,
                "cols": cols,
            }
        )

    def build_partials(self, input_names, output_names, step_name):
        """Expand and validate declared partial derivatives.

        Parameters
        ----------
        input_names : collection of str
            Fully qualified input variable names linked to the plugin step.
        output_names : collection of str
            Fully qualified output variable names linked to the plugin step.
        step_name : str
            Fully qualified plugin step name.

        Returns
        -------
        dict
            Mapping from ``(input_name, output_name)`` pairs to derivative metadata
            dictionaries containing method, form, step, rows, and cols.

        Raises
        ------
        KeyError
            If a declared partial references an input or output not declared by the
            plugin.
        ValueError
            If declared partial names are not a subset of the actual linked plugin
            inputs or outputs.
        TypeError
            If a variable declared in a partial is not continuous-valued.

        Notes
        -----
        Wildcard declarations are expanded against the actual linked plugin inputs
        and outputs after the step-name prefix is removed.

        """
        self.log.debug("build partials")

        # Validate names
        # Collect declared names
        partial_ins = set()
        partial_outs = set()
        for data in self.initial_partials:
            if data["wrt"] != ["*"]:
                for inp in data["wrt"]:
                    partial_ins.add(inp)
                    if inp in self.plugin_input_names:
                        input_type = self.plugin_input_names[inp]["type"]
                    else:
                        raise KeyError(
                            f"The input {inp} declared as a partial is not an input in the Plugin"
                        )
                    if not is_continuous_type_or_value(input_type):
                        raise TypeError(
                            f"Variables with declared partials must be continuous the variable:\n"
                            f"{inp} is not continuous and has type {input_type}"
                        )
            if data["of"] != ["*"]:
                for out in data["of"]:
                    partial_outs.add(out)
                    if out in self.plugin_output_names:
                        output_type = self.plugin_output_names[out]["type"]
                    else:
                        raise KeyError(
                            f"The output {out} declared as a partial is not an output in the Plugin"
                        )
                    if not is_continuous_type_or_value(output_type):
                        raise TypeError(
                            f"Variables with declared partials must be continuous the variable:\n"
                            f"{out} is not continuous and has type {output_type}"
                        )

        # Collect actual names
        local_output_names = set()
        local_input_names = set()
        start_ind = len(step_name.split("."))

        out_vio = []
        for output_name in output_names:
            local_output_name = ".".join(output_name.split(".")[start_ind:])
            local_output_names.add(local_output_name)
            output_type = self.plugin_output_names[local_output_name]["type"]

            if not is_continuous_type_or_value(output_type):
                out_vio.append(output_name)
        in_vio = []
        for input_name in input_names:
            local_input_name = ".".join(input_name.split(".")[start_ind:])
            local_input_names.add(local_input_name)
            input_type = self.plugin_input_names[local_input_name]["type"]

            if not is_continuous_type_or_value(input_type):
                in_vio.append(input_name)

        # Note the empty set is a subset of all other sets
        if not partial_ins.issubset(local_input_names):
            self.log.error(
                f"Inputs and partials do not match:\n"
                f"Inputs: {input_names}\n"
                f"Partial inputs: {partial_ins}\n"
            )
            raise ValueError(
                f"Inputs and partials do not match:\n"
                f"Inputs: {input_names}\n"
                f"Partial inputs: {partial_ins}\n"
            )
        if not partial_outs.issubset(local_output_names):
            self.log.error(
                f"Outputs and partials do not match:\n"
                f"Outputs: {output_names}\n"
                f"Partial outputs: {partial_outs}\n"
            )
            raise ValueError(
                f"Outputs and partials do not match:\n"
                f"Outputs: {output_names}\n"
                f"Partial outputs: {partial_outs}\n"
            )

        # overwrites so declared partials order is last out
        self.comprehensive_partials_dict = {}
        for data in self.initial_partials:
            if data["of"] == ["*"]:
                ofs = local_output_names
                if out_vio:
                    raise TypeError(
                        f"Variables with declared partials must be continuous the variables:\n"
                        f"{out_vio} are not continuous but were declared partials"
                    )
            else:
                ofs = data["of"]

            if data["wrt"] == ["*"]:
                wrts = local_input_names
                if in_vio:
                    raise TypeError(
                        f"Variables with declared partials must be continuous the variables:\n"
                        f"{in_vio} are not continuous but were declared partials"
                    )
            else:
                wrts = data["wrt"]

            for in_name, out_name in product(ofs, wrts):
                self.comprehensive_partials_dict[(in_name, out_name)] = {
                    "method": data["method"],
                    "form": data["form"],
                    "step": data["step"],
                    "rows": data["rows"],
                    "cols": data["cols"],
                }

        return self.comprehensive_partials_dict

    def categorize_partials(self):
        """Group partial declarations by differentiation method.

        Returns
        -------
        dict
            Mapping from derivative method name to lists of ``(output_name,
            input_name)`` pairs.

        Notes
        -----
        If any exact partials are present, ``_declare_partials`` is set to True.

        """
        self.log.debug("catagorize partials")
        method_to_partial = {"fd": [], "exact": []}
        for pair, data in self.comprehensive_partials_dict.items():
            if data["method"] == "fd":
                method_to_partial["fd"].append(pair)
            elif data["method"] == "exact":
                method_to_partial["exact"].append(pair)

        self.method_to_partial = method_to_partial

        if method_to_partial["exact"]:
            self._declare_partials = True

        return self.method_to_partial

    def compute_partials_plugin(self, links: dict):
        """Compute analytic partial derivatives for the plugin.

        Parameters
        ----------
        links : dict
            Mapping from declared plugin input names to values supplied by linked
            variables.

        Returns
        -------
        None

        Notes
        -----
        Concrete plugin subclasses override this method when exact partials are
        declared.

        """
        self.log.debug("compute partials plugin")

    def set_check_partials_options(self, form="forward", step=None):
        """Configure finite-difference settings for partial checks.

        Parameters
        ----------
        form : {"forward", "backward", "central"}, optional
            Finite-difference form used during partial checking.
        step : real, optional
            Finite-difference step size used during partial checking.

        Returns
        -------
        None

        """
        self.log.debug("set check partials options")
        self._check_partials_options = True
        self.check_form = form
        self.check_step = step

    @staticmethod
    def check_declared_IO(name, var_type, value, shape):
        """Validate declared plugin input or output metadata.

        Parameters
        ----------
        name : str
            Variable name.
        var_type : type
            Declared Python type of the variable.
        value : object
            Declared default value.
        shape : int or tuple of int, optional
            Declared array shape metadata.

        Returns
        -------
        tuple or None
            Normalized shape metadata. If ``shape`` is omitted for an array-valued
            declaration and the value is an array, the shape is inferred from the
            value.

        Raises
        ------
        TypeError
            If ``var_type`` is not a Python type.
        ValueError
            If the variable name contains unsupported characters, if array type and
            shape/value metadata are inconsistent, or if ``shape`` is invalid.

        """
        if not isinstance(var_type, type):
            raise TypeError(
                f"Declared type must be a type. {name} has declared type {var_type} which is not a type."
            )

        if "~" in name or "[" in name or "]" in name:
            raise ValueError(
                f'Tildes (~), square brackets "[","]" are not allowed in variable names\n'
                f"The following name is not allowed: {name}"
            )

        if shape is not None:

            if var_type is not np.ndarray:
                raise ValueError(
                    f"{name} has declared type: {var_type} but a shape argument was provided\n"
                    f"only np.ndarray types are allowed to have shapes"
                )
            else:
                if not check_numpy_shape(shape):
                    raise ValueError(
                        f"The shape must be declared as a tuple of integers or as a\n"
                        f"single integer or as NoneType. The shape of {name} is {shape} which is invalid"
                    )

                shape = normalize_shape(shape)
                if value is not None:
                    if is_scalar_type_or_value(value):
                        pass
                    elif value.shape != shape:
                        raise ValueError(
                            f"If the value of an array is not declared as a scalar, then the shape must match\n"
                            f"the declared shape. The variable {name} has declared shape {shape} and actual shape {value.shape}"
                        )
        else:

            if value is None and var_type is np.ndarray:
                raise ValueError(
                    f"Arrays must have either the shape or value defined (or both). The variable\n"
                    f"{name} has no shape or value."
                )
            if (
                value is not None
                and is_scalar_type_or_value(value)
                and var_type is np.ndarray
            ):
                raise ValueError(
                    f"If array value is declared with scalar, shape must be declared for broadcasting.\n"
                    f"The variable {name} has value: {value} and type: {var_type} but no shape declared."
                )
            elif (
                value is not None
                and isinstance(value, np.ndarray)
                and var_type is np.ndarray
            ):
                shape = value.shape

        return shape

    def declare_input(
        self, name, var_type, value=None, shape: int | tuple | None = None
    ):
        """Declare plugin input metadata.

        Parameters
        ----------
        name : str
            Input variable name.
        var_type : type
            Declared Python type of the input.
        value : object, optional
            Declared default value.
        shape : int | tuple | None, optional
            Declared array shape metadata.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``var_type`` is not a Python type.
        ValueError
            If the declared name, value, or shape metadata are invalid.

        Notes
        -----
        Metadata is stored for link validation, type checking, and default-value
        propagation.

        """
        shape = self.check_declared_IO(name, var_type, value, shape)
        self.plugin_input_names[name] = {
            "type": var_type,
            "value": value,
            "shape": shape,
        }

    def declare_output(
        self, name, var_type, value=None, shape: int | tuple | None = None
    ):
        """Declare plugin output metadata.

        Parameters
        ----------
        name : str
            Output variable name.
        var_type : type
            Declared Python type of the output.
        value : object, optional
            Declared default value.
        shape : int | tuple | None, optional
            Declared array shape metadata.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``var_type`` is not a Python type.
        ValueError
            If the declared name, value, or shape metadata are invalid.

        Notes
        -----
        Metadata is stored for link validation, type checking, and default-value
        propagation.

        """
        shape = self.check_declared_IO(name, var_type, value, shape)
        self.plugin_output_names[name] = {
            "type": var_type,
            "value": value,
            "shape": shape,
        }

    def declare_template(self, path):
        """Register a template file for provenance tracking.

        Parameters
        ----------
        path : str or path-like
            Path to a template file used by the plugin.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If ``path`` has already been registered.

        Notes
        -----
        If the file exists at setup time, its SHA-256 checksum is recorded. If it
        does not exist yet, a placeholder checksum value is stored.

        """
        if path in self._template_to_checksum:
            raise KeyError(f"The path: {path} is already in the template dictionary")
        self._plugin_templates.append(path)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
                h = hashlib.sha256(data).hexdigest()
        else:
            h = "file_does_not_exist_at_setup"
        self._template_to_checksum[str(path)] = h

    def add_executable_data(self, path, description):
        """Register external executable metadata for the plugin.

        Parameters
        ----------
        path : str or path-like
            Path to an external executable used by the plugin.
        description : str
            Human-readable description of how the executable is used.

        Returns
        -------
        None

        """
        self._executable_to_data[path] = description


def register_plugin(cls):
    """Register a plugin class for lookup and deserialization.

    Parameters
    ----------
    cls : type
        Plugin class to register.

    Returns
    -------
    type
        Registered plugin class. This allows the function to be used as a
        class decorator.

    Notes
    -----
    The plugin registry is updated in memory. When registry writes are enabled,
    the plugin source path is also written to ``conf.json``.

    """
    plugin_name = cls.__name__
    log.debug(f"register plugin {plugin_name}")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.json")
    _config = check_and_get_config(config_path)

    # Handle case where module is run as __main__ (no __spec__)
    try:
        module_origin = canonicalize_path(inspect.getfile(cls))
    except (ValueError, AttributeError):
        # If module is run as __main__, get the actual file path from sys.argv[0]
        import sys

        if sys.argv and len(sys.argv) > 0:
            module_origin = canonicalize_path(os.path.abspath(sys.argv[0]))
        else:
            # Fallback to current working directory + module name
            module_origin = canonicalize_path(os.path.join(os.getcwd(), cls.__module__ + ".py"))

    if (
        plugin_name in _config["PLUGINS"]
        and canonicalize_path(_config["PLUGINS"][plugin_name])
        != canonicalize_path(module_origin)
    ):
        log.warning(
            f"Plugin '{plugin_name}' already registered\nOverwritting existing data"
        )

    _config["PLUGINS"][plugin_name] = module_origin

    if is_main_process() and registry_writes_enabled():
        with open(config_path, "w") as configfile:
            json.dump(_config, configfile, indent=4)

    _plugin_registry[plugin_name] = cls
    return cls


def get_plugin_class(plugin_name):
    """Return a registered plugin class by name.

    Parameters
    ----------
    plugin_name : str
        Registered plugin class name.

    Returns
    -------
    type
        Registered plugin class.

    Raises
    ------
    ImportError
        If ``plugin_name`` is not registered.

    """
    if not plugin_name in _plugin_registry:
        msg = (
            f"Plugin '{plugin_name}' was not registered. Please check the following:\n"
        )
        msg += f"\t1. Did you register the plugin using the @register_plugin?\n"
        msg += f"\t2. Were there Plugin Load Errors when you imported uam4x?\n"
        log.error(msg)
        raise ImportError(msg)
    return _plugin_registry[plugin_name]


def get_plugin_names():
    """Return registered plugin names.

    Returns
    -------
    list of str
        Registered plugin class names.

    """
    log.debug("get plugin names")
    return list(_plugin_registry.keys())
