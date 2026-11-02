"""Native uam4x N-squared model visualization.

This module deliberately has no OpenMDAO dependency.  It translates the native
uam4x solution hierarchy into a small JSON model and embeds that model in a
standalone HTML/SVG viewer.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from uam4x.n2_template import HTML_TEMPLATE
from uam4x.solution import Group


def _without_solution_prefix(name: str) -> str:
    """Return a fully qualified uam4x name without the internal solution prefix."""
    if name.startswith("solution."):
        return name[len("solution.") :]
    return name


def _split_indices(name: str) -> tuple[str, str | None]:
    """Split textual array indices from a uam4x variable path."""
    bracket = name.find("[")
    if bracket == -1:
        return name, None
    return name[:bracket], name[bracket:]


def _type_name(value: Any) -> str:
    """Return a stable display name for a declared Python type."""
    return getattr(value, "__name__", str(value))


def _json_value(value: Any) -> Any:
    """Convert a declared value to JSON without allowing an oversized payload."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size > 100:
            summary = {"summary": f"array with shape {tuple(value.shape)}"}
            try:
                summary["min"] = _json_value(float(np.nanmin(value)))
                summary["max"] = _json_value(float(np.nanmax(value)))
            except (TypeError, ValueError):
                pass
            return summary
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def _variable_record(
    component_id: str,
    local_name: str,
    direction: str,
    metadata: dict,
    *,
    include_values: bool,
) -> dict:
    """Build one JSON-safe plugin-variable record."""
    record = {
        "id": f"{component_id}.{local_name}",
        "name": local_name,
        "component": component_id,
        "direction": direction,
        "dtype": _type_name(metadata.get("type")),
        "shape": list(metadata["shape"]) if metadata.get("shape") is not None else None,
        "connected": False,
        "tracked": False,
        "design_var": False,
        "constraint": False,
        "objective": False,
    }
    if include_values:
        record["value"] = _json_value(metadata.get("value"))
    return record


def _component_dependencies(plugin, component_id: str) -> list[dict]:
    """Return native input-to-output dependency edges for one plugin.

    Explicit partial declarations are expanded without evaluating the plugin.
    Plugins without declarations use the same dense dependency assumption used
    by an explicit component whose outputs may depend on all inputs.
    """
    input_names = list(plugin.plugin_input_names)
    output_names = list(plugin.plugin_output_names)
    declarations = getattr(plugin, "initial_partials", None) or []
    pairs: dict[tuple[str, str], dict] = {}

    if declarations:
        for declaration in declarations:
            wrts = (
                input_names
                if declaration.get("wrt") == ["*"]
                else declaration.get("wrt", [])
            )
            ofs = (
                output_names
                if declaration.get("of") == ["*"]
                else declaration.get("of", [])
            )
            for input_name in wrts:
                if input_name not in plugin.plugin_input_names:
                    continue
                for output_name in ofs:
                    if output_name not in plugin.plugin_output_names:
                        continue
                    pairs[(input_name, output_name)] = {
                        "declared": True,
                        "method": declaration.get("method"),
                    }
    else:
        for input_name in input_names:
            for output_name in output_names:
                pairs[(input_name, output_name)] = {
                    "declared": False,
                    "method": None,
                }

    return [
        {
            "source": f"{component_id}.{input_name}",
            "target": f"{component_id}.{output_name}",
            "source_component": component_id,
            "target_component": component_id,
            "source_indices": None,
            "forward": True,
            "cycle": False,
            "internal": True,
            **metadata,
        }
        for (input_name, output_name), metadata in pairs.items()
    ]


def _build_group_tree(
    group: Group,
    plugins: dict,
    prefix: str,
    components: list[dict],
    variables: list[dict],
    dependencies: list[dict],
    *,
    include_values: bool,
) -> dict:
    """Recursively translate a uam4x group and its plugin declarations."""
    group_id = prefix or "solution"
    tree = {
        "id": group_id,
        "name": group_id.rsplit(".", 1)[-1],
        "kind": "group",
        "cyclic": bool(group._is_cycle),
        "cycle_solver": group.cycle_solver or None,
        "children": [],
    }

    steps = group._execution_steps or group.steps
    for step_name, step_data in steps.items():
        step_id = f"{prefix}.{step_name}" if prefix else step_name
        if isinstance(step_data, Group):
            tree["children"].append(
                _build_group_tree(
                    step_data,
                    plugins[step_name],
                    step_id,
                    components,
                    variables,
                    dependencies,
                    include_values=include_values,
                )
            )
            continue

        plugin = plugins[step_name]
        component = {
            "id": step_id,
            "name": step_name,
            "kind": "component",
            "plugin_class": type(plugin).__name__,
            "parent": group_id,
            "order": len(components),
            "cyclic": bool(group._is_cycle),
            "cycle_solver": group.cycle_solver or None,
            "inputs": [],
            "outputs": [],
        }
        components.append(component)

        for local_name, metadata in plugin.plugin_input_names.items():
            variable = _variable_record(
                step_id,
                local_name,
                "input",
                metadata,
                include_values=include_values,
            )
            variables.append(variable)
            component["inputs"].append(variable["id"])

        for local_name, metadata in plugin.plugin_output_names.items():
            variable = _variable_record(
                step_id,
                local_name,
                "output",
                metadata,
                include_values=include_values,
            )
            variables.append(variable)
            component["outputs"].append(variable["id"])

        component_dependencies = _component_dependencies(plugin, step_id)
        component["dependencies"] = component_dependencies
        dependencies.extend(component_dependencies)

        tree["children"].append(
            {
                "id": step_id,
                "name": step_name,
                "kind": "component",
                "plugin_class": type(plugin).__name__,
            }
        )
    return tree


def _component_for_variable(variable_name: str, component_ids: list[str]) -> str | None:
    """Resolve a variable path to the longest matching component path."""
    if variable_name.startswith("vehicles."):
        return "vehicles"
    matches = [
        component_id
        for component_id in component_ids
        if variable_name == component_id or variable_name.startswith(component_id + ".")
    ]
    return max(matches, key=len) if matches else None


def _strongly_connected_components(
    component_ids: list[str], connections: list[dict]
) -> list[list[str]]:
    """Return cyclic strongly connected component sets using Tarjan's algorithm."""
    adjacency = {component_id: [] for component_id in component_ids}
    self_edges = set()
    for connection in connections:
        source = connection["source_component"]
        target = connection["target_component"]
        if source not in adjacency or target not in adjacency:
            continue
        adjacency[source].append(target)
        if source == target:
            self_edges.add(source)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    result: list[list[str]] = []

    def visit(node: str) -> None:
        """Record one component and update its strongly connected component state."""
        nonlocal index
        indices[node] = index
        low_links[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in adjacency[node]:
            if target not in indices:
                visit(target)
                low_links[node] = min(low_links[node], low_links[target])
            elif target in on_stack:
                low_links[node] = min(low_links[node], indices[target])

        if low_links[node] != indices[node]:
            return

        component = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in self_edges:
            result.append(sorted(component, key=component_ids.index))

    for component_id in component_ids:
        if component_id not in indices:
            visit(component_id)
    return result


def _optimization_sets(state: dict | None) -> tuple[set[str], set[str], set[str]]:
    """Extract normalized optimization-variable names from a trade state."""
    state = state or {}
    design_vars = {
        _split_indices(_without_solution_prefix(name))[0]
        for name in state.get("des_vars", {})
    }
    constraints = {
        _split_indices(_without_solution_prefix(name))[0]
        for name in state.get("constraints", {})
    }
    objective = state.get("objective_name")
    objectives = (
        {_split_indices(_without_solution_prefix(objective))[0]} if objective else set()
    )
    return design_vars, constraints, objectives


def build_n2_data(
    study,
    solution: Group,
    plugins: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
    trade_name: str | None = None,
    title: str | None = None,
    include_values: bool = False,
    initial_path: str | None = None,
    auto_collapse: bool = True,
    max_visible_rows: int = 120,
) -> dict[str, Any]:
    """Build the model data used by the standalone N² viewer.

    Read the solution hierarchy, plugin variable declarations, vehicle
    parameters, and links without executing plugins. Mark tracked outputs
    and, when a trade state is supplied, its design variables, constraints,
    and objective.

    ``initial_path`` selects the group or component shown first. A unique
    short name is accepted; an unknown or ambiguous path raises ValueError.
    Declared values are included only when ``include_values`` is True.
    """
    components: list[dict] = []
    variables: list[dict] = []
    dependencies: list[dict] = []

    vehicle_component = {
        "id": "vehicles",
        "name": "vehicles",
        "kind": "component",
        "plugin_class": "uam4x vehicle parameters",
        "parent": "solution",
        "order": 0,
        "cyclic": False,
        "cycle_solver": None,
        "inputs": [],
        "outputs": [],
        "dependencies": [],
    }
    components.append(vehicle_component)

    tree = _build_group_tree(
        solution,
        plugins,
        "",
        components,
        variables,
        dependencies,
        include_values=include_values,
    )
    tree["children"].insert(
        0,
        {
            "id": "vehicles",
            "name": "vehicles",
            "kind": "component",
            "plugin_class": "uam4x vehicle parameters",
        },
    )

    # _build_group_tree starts plugin ordering at one because vehicles is already present.
    for order, component in enumerate(components):
        component["order"] = order

    used_vehicle_names = []
    for name in solution.get_all_used_vehicle_params():
        clean_name, _ = _split_indices(name)
        if clean_name not in used_vehicle_names:
            used_vehicle_names.append(clean_name)
    for name in used_vehicle_names:
        try:
            if state is not None and name in state:
                value = state[name]
            else:
                value = study.get_value(name)
            dtype = study.get_param_type(name)
            shape = list(value.shape) if isinstance(value, np.ndarray) else None
        except Exception:
            value = None
            dtype = type(None)
            shape = None
        variable = {
            "id": name,
            "name": name.removeprefix("vehicles."),
            "component": "vehicles",
            "direction": "output",
            "dtype": _type_name(dtype),
            "shape": shape,
            "connected": False,
            "tracked": False,
            "design_var": False,
            "constraint": False,
            "objective": False,
        }
        if include_values:
            variable["value"] = _json_value(value)
        variables.insert(len(vehicle_component["outputs"]), variable)
        vehicle_component["outputs"].append(name)

    component_ids = [component["id"] for component in components]
    component_orders = {
        component["id"]: component["order"] for component in components
    }
    connections = []
    for source, target in solution._build_global_links():
        source = _without_solution_prefix(source)
        target = _without_solution_prefix(target)
        source, source_indices = _split_indices(source)
        target, _ = _split_indices(target)
        source_component = _component_for_variable(source, component_ids)
        target_component = _component_for_variable(target, component_ids)
        if source_component is None or target_component is None:
            continue
        connections.append(
            {
                "source": source,
                "target": target,
                "source_component": source_component,
                "target_component": target_component,
                "source_indices": source_indices,
                "forward": component_orders[source_component]
                <= component_orders[target_component],
                "cycle": False,
            }
        )

    cycles = _strongly_connected_components(component_ids, connections)
    cycle_by_component = {
        component_id: cycle_index
        for cycle_index, cycle in enumerate(cycles)
        for component_id in cycle
    }
    for connection in connections:
        source_cycle = cycle_by_component.get(connection["source_component"])
        target_cycle = cycle_by_component.get(connection["target_component"])
        connection["cycle"] = (
            source_cycle is not None and source_cycle == target_cycle
        )
    for component in components:
        component["cycle_id"] = cycle_by_component.get(component["id"])

    variable_by_id = {variable["id"]: variable for variable in variables}
    for connection in connections:
        if connection["source"] in variable_by_id:
            variable_by_id[connection["source"]]["connected"] = True
        if connection["target"] in variable_by_id:
            variable_by_id[connection["target"]]["connected"] = True

    tracked = {
        _split_indices(_without_solution_prefix(name))[0]
        for name in solution.tracked_outputs
    }
    design_vars, constraints, objectives = _optimization_sets(state)
    for variable in variables:
        variable["tracked"] = variable["id"] in tracked
        variable["design_var"] = variable["id"] in design_vars
        variable["constraint"] = variable["id"] in constraints
        variable["objective"] = variable["id"] in objectives

    system_ids = {tree["id"]}

    def collect_system_ids(node: dict) -> None:
        """Collect this hierarchy node's ID and all descendant IDs."""
        system_ids.add(node["id"])
        for child in node.get("children", []):
            collect_system_ids(child)

    collect_system_ids(tree)
    resolved_path = initial_path
    if resolved_path:
        resolved_path = _without_solution_prefix(
            str(resolved_path).strip().strip(".")
        )
        resolved_path = resolved_path or "solution"
        if resolved_path not in system_ids:
            suffix_matches = sorted(
                system_id
                for system_id in system_ids
                if system_id.rsplit(".", 1)[-1] == resolved_path
            )
            if len(suffix_matches) == 1:
                resolved_path = suffix_matches[0]
            else:
                available = ", ".join(sorted(system_ids)[:12])
                raise ValueError(
                    f"N2 path {initial_path!r} is not a unique uam4x group or "
                    f"component path. Available paths include: {available}"
                )

    return {
        "schema_version": 1,
        "title": title or f"uam4x N² Diagram — {solution.name}",
        "solution_name": solution.name,
        "trade_name": trade_name,
        "tree": tree,
        "components": components,
        "variables": variables,
        "connections": connections,
        "dependencies": dependencies,
        "cycles": cycles,
        "tracked_outputs": sorted(tracked),
        "optimization": {
            "enabled": bool((state or {}).get("optimization", False)),
            "design_variables": sorted(design_vars),
            "constraints": sorted(constraints),
            "objectives": sorted(objectives),
        },
        "viewer": {
            "initial_path": resolved_path or "solution",
            "auto_collapse": bool(auto_collapse),
            "max_visible_rows": max_visible_rows,
        },
    }


def render_n2_html(model_data: dict[str, Any], outfile: str | Path) -> Path:
    """Write a standalone N² HTML file and return its absolute path.

    Create the parent directory if needed, embed the model data in the
    HTML template, and write the file using UTF-8.
    """
    output_path = Path(outfile).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        model_data,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    document = (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(model_data["title"]))
        .replace("__MODEL_DATA__", encoded_payload)
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path
