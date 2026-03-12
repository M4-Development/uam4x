"""
Node Management System for UAM4X

Handles node registration and management similar to subsystems and plugins.
"""

from typing import Dict, List, Any, Optional, Callable
import json
import os
import importlib.util
import inspect
import sys

from uam4x.log import get_log
from uam4x.utils import log
# Global node registry
_node_registry = {}

class Node:
    """Represent the node component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """
    log = get_log(log, object_name='node')
    def __init__(self, name: str, description: str = "", function: Callable = None):
        """
        Initialize a node.

        Parameters
        ----------
        name : str
            Name of the node
        description : str, optional
            Description of what the node does
        function : Callable, optional
            The function that implements this node
        """
        self.log = Node.log
        self.name = name
        self.description = description
        self.function = function
        self.module_path = ""
        self.source_file = ""
        self.node_type = "Node"

    def execute(self, study):
        """Execute execute.

        Parameters
        ----------
        study : Any
            Input value for `study`.

        """
        if self.function:
            self.function(study)
        else:
            raise ValueError(f"Node '{self.name}' has no function to execute")

    def validate_signature(self) -> bool:
        """
        Validate that the node function has the correct signature.

        Returns
        -------
        bool
            True if the function signature is valid (takes exactly one Study parameter)
        """
        if not self.function:
            return False


        try:
            sig = inspect.signature(self.function)
            params = list(sig.parameters.values())

            # Check if there's exactly one parameter
            if len(params) != 1:
                return False

            # Check if the parameter has a type annotation for Study
            param = params[0]
            if param.annotation != inspect.Signature.empty:
                # Check if the annotation is Study or a string "Study"
                annotation_str = str(param.annotation)
                return "Study" in annotation_str
            else:
                # If no type annotation, assume it's valid for backward compatibility
                return True

        except (ValueError, TypeError):
            return False

    def get_docstring(self) -> str:
        """Get the docstring of the node function.

        Returns
        -------
        Any
            Requested value.

        """
        if self.function:
            return self.function.__doc__ or ""
        return ""

    def get_source_info(self) -> Dict[str, str]:
        """
        Get source file information for this node.

        Returns
        -------
        Dict[str, str]
            Dictionary containing 'file' and 'line' information
        """
        if self.function:
            try:
                source_file = inspect.getfile(self.function)
                source_lines = inspect.getsourcelines(self.function)
                return {
                    'file': source_file,
                    'line': str(source_lines[1])
                }
            except (OSError, TypeError):
                return {'file': 'unknown', 'line': 'unknown'}
        return {'file': 'unknown', 'line': 'unknown'}


    def _to_dict(self) -> dict:
        """Serialize the Node to a dictionary.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
            "node_type": "Node",
            "name": self.name
        }


def register_node(name: str = None, description: str = ""):
    """
    Decorator to register a function as a node.

    Parameters
    ----------
    name : str, optional
        Name for the node. If not provided, uses the function name.
    description : str, optional
        Description of what the node does.

    Returns
    -------
    Callable
        The decorator function
    """
    def decorator(func):
        """Execute decorator.

        Parameters
        ----------
        func : Any
            Input value for `func`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        node_name = name or func.__name__

        # Create the node
        node = Node(
            name=node_name,
            description=description or func.__doc__ or "",
            function=func
        )

        # Set source information
        try:
            node.source_file = inspect.getfile(func)
            node.module_path = func.__module__
        except (OSError, TypeError):
            node.source_file = "unknown"
            node.module_path = func.__module__

        # Validate the node signature
        if not node.validate_signature():
            log.warning(f"Node '{node_name}' does not have a valid signature. "
                  f"Expected: def {node_name}(study: Study)")

        # Register the node
        _node_registry[node_name] = node

        # Skip configuration file updates for now to avoid corruption
        # TODO: Fix the configuration file corruption issue

        # Update configuration file with proper error handling
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')

            # Get module origin for configuration
            try:
                module_origin = importlib.util.find_spec(func.__module__).origin
            except (ValueError, AttributeError):
                if sys.argv and len(sys.argv) > 0:
                    module_origin = os.path.abspath(sys.argv[0])
                else:
                    module_origin = os.path.join(os.getcwd(), func.__module__ + '.py')

            # Load existing config safely
            try:
                with open(config_path, 'r') as configfile:
                    existing_config = json.load(configfile)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_config = {}

            # Create a completely fresh config dictionary
            clean_config = {
                'SUBSYSTEMS': {},
                'PLUGINS': {},
                'NODES': {}
            }

            # Copy only string values from existing config
            if 'SUBSYSTEMS' in existing_config and isinstance(existing_config['SUBSYSTEMS'], dict):
                for key, value in existing_config['SUBSYSTEMS'].items():
                    if isinstance(key, str) and isinstance(value, str):
                        clean_config['SUBSYSTEMS'][key] = value

            if 'PLUGINS' in existing_config and isinstance(existing_config['PLUGINS'], dict):
                for key, value in existing_config['PLUGINS'].items():
                    if isinstance(key, str) and isinstance(value, str):
                        clean_config['PLUGINS'][key] = value

            if 'NODES' in existing_config and isinstance(existing_config['NODES'], dict):
                for key, value in existing_config['NODES'].items():
                    if isinstance(key, str) and isinstance(value, str):
                        clean_config['NODES'][key] = value

            # Update NODES section
            if node_name in clean_config['NODES'] and clean_config['NODES'][node_name] != module_origin:
                log.warning(f"Node '{node_name}' already registered\nOverwriting existing data")

            clean_config['NODES'][node_name] = module_origin

            # Save the clean config
            with open(config_path, 'w') as configfile:
                json.dump(clean_config, configfile, indent=4)

        except Exception as e:
            log.warning(f"Failed to update configuration file: {e}")
            # Continue execution even if config update fails

        return func

    return decorator


def get_node(node_name: str) -> Optional[Node]:
    """
    Get a registered node by name.

    Parameters
    ----------
    node_name : str
        Name of the node to retrieve

    Returns
    -------
    Optional[Node]
        The node if found, None otherwise
    """
    return _node_registry[node_name]

def get_node_names() -> List[str]:
    """
    Get a list of all registered node names.

    Returns
    -------
    List[str]
        List of registered node names
    """
    return list(_node_registry.keys())


def get_node_dict() -> Dict[str, Node]:
    """
    Get a dictionary of all registered nodes.

    Returns
    -------
    Dict[str, Node]
        Dictionary mapping node names to Node objects
    """
    return _node_registry.copy()

def validate_all_nodes() -> Dict[str, bool]:
    """
    Validate all registered nodes.

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping node names to validation results
    """
    results = {}
    for name, node in _node_registry.items():
        results[name] = node.validate_signature()
    return results


def get_nodes_by_module(module_name: str) -> List[Node]:
    """
    Get all nodes from a specific module.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    List[Node]
        List of nodes from the specified module
    """
    return [node for node in _node_registry.values() if node.module_path == module_name]


def get_nodes_by_file(file_path: str) -> List[Node]:
    """
    Get all nodes from a specific file.

    Parameters
    ----------
    file_path : str
        Path to the file

    Returns
    -------
    List[Node]
        List of nodes from the specified file
    """
    return [node for node in _node_registry.values() if node.source_file == file_path]

