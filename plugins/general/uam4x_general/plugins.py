from abc import ABC, abstractmethod
from os import PathLike
from typing import Any
import numpy as np


from uam4x.solution import Plugin, Solution,register_plugin, Group
from uam4x.utils import REVERSE_TYPE_MAP, declare_standard_io
import sympy as sp

def _validate_options(plugin_name, plugin_options, required_options):
    """Validate required plugin option keys and variable declarations."""
    if not isinstance(plugin_options, dict):
        raise ValueError(
            f"{plugin_name} plugin_options must be a dictionary, "
            f"got {type(plugin_options).__name__}."
        )

    missing = [name for name in required_options if name not in plugin_options]
    if missing:
        raise ValueError(
            f"{plugin_name} is missing required plugin option(s): "
            f"{', '.join(missing)}."
        )

    for option_name in ("input_dict", "output_dict"):
        if not isinstance(plugin_options[option_name], dict):
            raise ValueError(
                f"{plugin_name} option {option_name!r} must be a dictionary."
            )


def _describe_name_mismatch(expected, actual):
    """Describe missing and unexpected names in two sets."""
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing: {missing}")
    if unknown:
        details.append(f"unknown: {unknown}")
    return "; ".join(details)

#----------------------------------Plugins-------------------------------


@register_plugin
class SummationPlugin(Plugin):

    """Compute a weighted sum of declared inputs.

    ``plugin_options`` must include ``input_dict``, a single-entry
    ``output_dict``, and ``input_to_factor`` with one factor per input.
    The output is the sum of ``factor * input_value`` for every input.
    """
    def setup_plugin(self,plugin_options:dict):
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``input_dict``, one ``output_dict`` entry, and ``input_to_factor`` with one factor per input.

        Returns
        -------
        None
        """
        _validate_options(
            "SummationPlugin",
            plugin_options,
            ("input_dict", "output_dict", "input_to_factor"),
        )
        if len(plugin_options["output_dict"]) != 1:
            raise ValueError(
                "SummationPlugin requires exactly one declared output, "
                f"got {len(plugin_options['output_dict'])}."
            )
        if not plugin_options["input_dict"]:
            raise ValueError("SummationPlugin requires at least one declared input.")
        if not isinstance(plugin_options["input_to_factor"], dict):
            raise ValueError(
                "SummationPlugin option 'input_to_factor' must be a dictionary."
            )
        input_names = set(plugin_options["input_dict"])
        factor_names = set(plugin_options["input_to_factor"])
        if input_names != factor_names:
            raise ValueError(
                "SummationPlugin must define one factor for every input ("
                f"{_describe_name_mismatch(input_names, factor_names)})."
            )

        self.input_to_factor=plugin_options['input_to_factor']
        self.input_dict=plugin_options["input_dict"]
        self.output_dict=plugin_options["output_dict"]
        self.output_name=list(self.output_dict.keys())[0]

        declare_standard_io(self, plugin_options)


    def run_plugin(self, links: dict, work_dir: str):

        """Calculate the weighted sum of linked inputs.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            UAM4X working directory; unused by this calculation.

        Returns
        -------
        dict
            Contains the weighted sum under the declared output name.
        """
        out=0.0
        for var_name,value in links.items():
            out+=self.input_to_factor[var_name]*value

        return {self.output_name:out}


@register_plugin
class DivisionPlugin(Plugin):
    """Divide one input among declared outputs using per-output factors.

    ``plugin_options`` must declare exactly one input and at least one output.
    ``output_to_factor`` maps each output name to a divisor. A ``None`` factor
    divides the input by the number of outputs instead.
    """
    def setup_plugin(self,plugin_options:dict):
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains one ``input_dict`` entry, ``output_dict``, and ``output_to_factor`` with one divisor per output.

        Returns
        -------
        None
        """
        _validate_options(
            "DivisionPlugin",
            plugin_options,
            ("input_dict", "output_dict", "output_to_factor"),
        )
        if len(plugin_options["input_dict"]) != 1:
            raise ValueError(
                "DivisionPlugin requires exactly one declared input, "
                f"got {len(plugin_options['input_dict'])}."
            )
        if not plugin_options["output_dict"]:
            raise ValueError("DivisionPlugin requires at least one declared output.")
        if not isinstance(plugin_options["output_to_factor"], dict):
            raise ValueError(
                "DivisionPlugin option 'output_to_factor' must be a dictionary."
            )
        output_names = set(plugin_options["output_dict"])
        factor_names = set(plugin_options["output_to_factor"])
        if output_names != factor_names:
            raise ValueError(
                "DivisionPlugin must define one factor for every output ("
                f"{_describe_name_mismatch(output_names, factor_names)})."
            )
        zero_factor_outputs = [
            name
            for name, factor in plugin_options["output_to_factor"].items()
            if factor == 0
        ]
        if zero_factor_outputs:
            raise ValueError(
                "DivisionPlugin factors cannot be zero; zero factor specified "
                f"for output(s): {zero_factor_outputs}."
            )

        self.input_dict=plugin_options["input_dict"]
        self.output_dict=plugin_options["output_dict"]
        self.output_to_factor=plugin_options["output_to_factor"]

        declare_standard_io(self, plugin_options)

    def run_plugin(self, links: dict, work_dir: str):
        """Divide the linked input among configured outputs.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            UAM4X working directory; unused by this calculation.

        Returns
        -------
        dict
            Contains each declared output mapped to the input divided by its factor.
        """
        if len(links) != 1:
            raise ValueError(
                "DivisionPlugin requires exactly one input value at runtime, "
                f"got {len(links)}."
            )
        link_val=list(links.values())[0]
        num_outputs=len(self.output_dict)
        results={}
        for output_name,factor in self.output_to_factor.items():
            if factor is not None:
                results[output_name]=link_val/factor
            else:
                results[output_name]=link_val/num_outputs
        return results


@register_plugin
class UnitConverterPlugin(Plugin):

    """Apply configured multiplicative conversions to input values.

    ``input_to_output_conversion_factor_dict`` maps each input name to a list
    of ``(output_name, factor)`` pairs. Each declared output must appear
    exactly once across those pairs.
    """
    def setup_plugin(self, plugin_options: dict):
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``input_dict``, ``output_dict``, and ``input_to_output_conversion_factor_dict`` containing output/factor pairs.

        Returns
        -------
        None
        """
        _validate_options(
            "UnitConverter",
            plugin_options,
            (
                "input_dict",
                "output_dict",
                "input_to_output_conversion_factor_dict",
            ),
        )
        conversions = plugin_options["input_to_output_conversion_factor_dict"]
        if not isinstance(conversions, dict):
            raise ValueError(
                "UnitConverter option "
                "'input_to_output_conversion_factor_dict' must be a dictionary."
            )

        declared_inputs = set(plugin_options["input_dict"])
        conversion_inputs = set(conversions)
        if declared_inputs != conversion_inputs:
            raise ValueError(
                "UnitConverter must define conversions for every input ("
                f"{_describe_name_mismatch(declared_inputs, conversion_inputs)})."
            )

        declared_outputs = set(plugin_options["output_dict"])
        converted_outputs = []
        for input_name, conversion_list in conversions.items():
            if not isinstance(conversion_list, (list, tuple)):
                raise ValueError(
                    f"UnitConverter conversions for input {input_name!r} must "
                    "be a list or tuple of (output_name, factor) pairs."
                )
            for conversion in conversion_list:
                if not isinstance(conversion, (list, tuple)) or len(conversion) != 2:
                    raise ValueError(
                        f"Invalid conversion for input {input_name!r}: "
                        f"{conversion!r}; expected (output_name, factor)."
                    )
                converted_outputs.append(conversion[0])

        converted_output_names = set(converted_outputs)
        if declared_outputs != converted_output_names:
            raise ValueError(
                "UnitConverter must define one conversion for every output ("
                f"{_describe_name_mismatch(declared_outputs, converted_output_names)})."
            )
        duplicate_outputs = sorted(
            name for name in converted_output_names if converted_outputs.count(name) > 1
        )
        if duplicate_outputs:
            raise ValueError(
                "UnitConverter outputs may only be assigned by one conversion; "
                f"duplicates: {duplicate_outputs}."
            )

        self.in_to_out_factor=plugin_options["input_to_output_conversion_factor_dict"]
        self.output_dict=plugin_options["output_dict"]
        self.input_dict=plugin_options["input_dict"]

        declare_standard_io(self, plugin_options)



    def run_plugin(self, links: dict, work_dir: str):

        """Apply every configured input-to-output conversion.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            UAM4X working directory; unused by this calculation.

        Returns
        -------
        dict
            Contains each converted output mapped to its scaled input value.
        """
        converted={}
        for var_name, value in links.items():
            for out,factor in self.in_to_out_factor[var_name]:
                converted[out]=factor*value

        return converted

@register_plugin
class ExpressionPlugin(Plugin):
    """Evaluate SymPy expressions and their declared partial derivatives.

    ``output_to_expression`` maps every declared output to a SymPy-compatible
    expression string. ``partials_method`` may be ``"exact"`` (the default)
    or ``"fd"`` for finite differences. Input name ``pi`` is reserved.
    """
    def setup_plugin(self, plugin_options: dict):
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``input_dict``, ``output_dict``, ``output_to_expression``, and optional ``partials_method``.

        Returns
        -------
        None
        """
        _validate_options(
            "ExpressionPlugin",
            plugin_options,
            ("input_dict", "output_dict", "output_to_expression"),
        )

        self.output_dict=plugin_options["output_dict"]
        self.input_dict=plugin_options["input_dict"]
        self.output_to_expression=plugin_options["output_to_expression"]
        self.partials_method = plugin_options.get("partials_method", "exact")
        if self.partials_method not in ("exact", "fd"):
            raise ValueError(
                "ExpressionPlugin option 'partials_method' must be either "
                f"'exact' or 'fd', got {self.partials_method!r}."
            )
        if "pi" in self.input_dict:
            raise ValueError(
                "ExpressionPlugin input name 'pi' is reserved for the "
                "mathematical constant pi."
            )
        if not isinstance(self.output_to_expression, dict):
            raise ValueError(
                "ExpressionPlugin option 'output_to_expression' must be a "
                "dictionary."
            )

        declared_outputs = set(self.output_dict)
        expression_outputs = set(self.output_to_expression)
        if declared_outputs != expression_outputs:
            raise ValueError(
                "ExpressionPlugin requires exactly one expression per output "
                f"({_describe_name_mismatch(declared_outputs, expression_outputs)})."
            )

        continuous_inputs,continuous_outputs=declare_standard_io(self, plugin_options)

        self.symbols={}
        for input_name in plugin_options["input_dict"]:
            self.symbols[input_name]=sp.Symbol(input_name,real=True)
        self.out_to_sympy_func={}
        self.name_to_ordered_inps={}
        # Maps (output_name, input_name) to a numerical derivative function.
        self.partial_functions = {}
        for output_name in plugin_options["output_dict"]:
            str_expr=self.output_to_expression[output_name]
            if not isinstance(str_expr, str) or not str_expr.strip():
                raise ValueError(
                    f"Expression for output {output_name!r} must be a "
                    "non-empty string."
                )
            expr=sp.sympify(str_expr,
                            locals={**self.symbols,"pi":sp.pi})

            known_symbols = set(self.symbols.values())
            unknown_symbols = expr.free_symbols - known_symbols

            if unknown_symbols:
                raise ValueError(
                    f"Expression for output {output_name!r} references undeclared "
                    f"variable(s): {sorted(map(str, unknown_symbols))}."
                )

            ordered_input_names = [
            input_name
            for input_name in self.input_dict
            if self.symbols[input_name] in expr.free_symbols
                ]

            ordered_symbols = [
            self.symbols[input_name]
            for input_name in ordered_input_names
            ]
            self.name_to_ordered_inps[output_name] = ordered_input_names

            self.out_to_sympy_func[output_name]=sp.lambdify(
                        ordered_symbols,
                        expr,
                        modules='numpy'
                    )

            # OpenMDAO only supports partials between continuous variables.
            if output_name not in continuous_outputs:
                continue

            for input_name in ordered_input_names:
                if input_name not in continuous_inputs:
                    continue

                if self.partials_method == "fd":
                    self.declare_partials(
                        of=output_name,
                        wrt=input_name,
                        method="fd",
                    )
                    continue

                derivative_expr = sp.diff(
                    expr,
                    self.symbols[input_name],
                )

                # An omitted sub-Jacobian is implicitly zero.
                if derivative_expr.is_zero is True:
                    continue

                self.declare_partials(
                    of=output_name,
                    wrt=input_name,
                    method="exact",
                )

                self.partial_functions[(output_name, input_name)] = (
                    sp.lambdify(
                        ordered_symbols,
                        derivative_expr,
                        modules="numpy",
                    )
                )

    def run_plugin(self, links: dict, work_dir: str):

        """Evaluate the configured symbolic expressions.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            UAM4X working directory; unused by this calculation.

        Returns
        -------
        dict
            Contains each declared output mapped to its evaluated expression.
        """
        results={}
        for name,function in self.out_to_sympy_func.items():
            vals=[]
            for input_name in self.name_to_ordered_inps[name]:
                vals.append(links[input_name])
            results[name]=function(*vals)
        return results

    def compute_partials(self, links: dict):
        """Return partial derivatives keyed by output and input names."""
        partials = {}

        for pair, function in self.partial_functions.items():
            output_name, _ = pair

            values = [
                links[input_name]
                for input_name in self.name_to_ordered_inps[output_name]
            ]

            partials[pair] = function(*values)

        return partials

class LookupTablePlugin(Plugin):
    '''
    implement lookup table functionality, so like fluid props etc
    '''

@register_plugin
class LinearSystemofEquations(Plugin):
    """Solve A_mat @ x = b with a fixed square matrix."""

    def setup_plugin(self, plugin_options: dict[str, Any]) -> None:
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``num_eq``, ``bvec_name``, ``A_mat``, and ``xvec_name`` for a fixed square system.

        Returns
        -------
        None
        """
        for option in ('num_eq', 'bvec_name', 'A_mat', 'xvec_name'):
            if option not in plugin_options:
                raise ValueError(f"LinearSystemofEquations requires '{option}'.")

        self.num_eq = plugin_options['num_eq']
        self.bvec_name = plugin_options['bvec_name']
        self.xvec_name = plugin_options['xvec_name']
        self.A_mat = np.array(plugin_options['A_mat'], dtype=float, copy=True)
        if self.A_mat.shape != (self.num_eq, self.num_eq):
            raise ValueError(f'A_mat must have shape ({self.num_eq}, {self.num_eq}).')
        if self.bvec_name == self.xvec_name:
            raise ValueError('Input and output vector names must be different.')

        self.declare_input(self.bvec_name, np.ndarray, shape=self.num_eq)
        self.declare_output(self.xvec_name, np.ndarray, shape=self.num_eq)
        self.declare_partials(of=self.xvec_name, wrt=self.bvec_name, method='exact')

    def run_plugin(self, links: dict[str, Any], work_dir: str | PathLike):
        """Solve the fixed system for the supplied right-hand-side vector.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            UAM4X working directory; unused by this calculation.

        Returns
        -------
        dict
            Contains the solution vector under ``xvec_name``.
        """
        b_vec = np.asarray(links[self.bvec_name])
        if b_vec.shape != (self.num_eq,):
            raise ValueError(f'{self.bvec_name} must have shape ({self.num_eq},).')
        try:
            x_out = np.linalg.solve(self.A_mat, b_vec)
        except np.linalg.LinAlgError as error:
            raise ValueError('A_mat is singular; the system has no unique solution.') from error
        return {self.xvec_name: x_out}

    def compute_partials(self, links):
        """Return the exact Jacobian, the inverse of the coefficient matrix."""
        return {(self.xvec_name, self.bvec_name): np.linalg.solve(self.A_mat, np.eye(self.num_eq))}


#----------------------------------Plugin Generator APIs-------------------------------
def group_step_path(solution, group, relative_path):
    """Qualify a group-relative path for solution-level links or tracking."""
    if group is None:
        return relative_path
    if not isinstance(group, Group):
        raise TypeError("group must be a Group instance or None")
    paths = []

    def visit(container, prefix, ancestors):
        for name, child in container.steps.items():
            if not isinstance(child, Group):
                continue
            path = f"{prefix}.{name}" if prefix else name
            if child is group:
                paths.append(path)
            if id(child) not in ancestors:
                visit(child, path, ancestors | {id(child)})

    visit(solution, "", {id(solution)})
    if len(paths) != 1:
        raise ValueError("Attach group exactly once to the solution before using its variable paths")
    return f"{paths[0]}.{relative_path}"


class PluginGenerator(ABC):
    """Base builder for configuring and registering a plugin step.

    Declare variables before calling :meth:`add_to_study` or
    :meth:`add_to_group`. A generator can be registered only once; ``group``
    selects a parent group when supplied.
    """

    @property
    @abstractmethod
    def plugin_class(self):
        """Plugin class configured by this generator."""
        raise NotImplementedError

    def __init__(self,name,solution:Solution,group:Group=None):

        """Initialize the instance with its configuration and owning solution."""
        self.name=name
        self._input_dict={}
        self._output_dict={}
        self.solution=solution
        if group is not None and not isinstance(group, Group):
            raise TypeError("group must be a Group instance or None")
        self.group = group
        self.added_to_solution = False
        self.added_to_group = False
        self._plugin_options={
            'input_dict':self._input_dict,
            'output_dict': self._output_dict,
        }


    def add_input(self,name,var_type,default_value):
        """Declare an input variable for the generated plugin."""
        self._input_dict[name]={
            "type":REVERSE_TYPE_MAP[var_type],
            "default_value":default_value
                               }
    def add_float_input(self,name,default_value=0.0,**kwargs):
        """Declare a floating-point input variable."""
        self.add_input(name, float, default_value, **kwargs)

    def add_int_input(self,name,default_value=1,**kwargs):
        """Declare an integer input variable."""
        self.add_input(name, int, default_value, **kwargs)



    def add_output(self,name,var_type,default_value,tracked=False):
        """Declare an output variable for the generated plugin."""
        self._output_dict[name]={
            "type":REVERSE_TYPE_MAP[var_type],
            "default_value":default_value
        }
        if tracked:
            self.solution.add_tracked_output(self.solution_output(name))

    def add_float_output(self,name,default_value=0.0,tracked=False,**kwargs):
        """Declare a floating-point output variable."""
        self.add_output(name, float, default_value, tracked=tracked, **kwargs)

    def add_int_output(self,name,default_value=0,tracked=False,**kwargs):
        """Declare an integer output variable."""
        self.add_output(name, int, default_value, tracked=tracked, **kwargs)

    def _check_registration(self, target):
        """Reject duplicate or conflicting step registration."""
        if self.added_to_solution or self.added_to_group:
            raise RuntimeError(f"Plugin generator {self.name!r} is already added")
        if self.name in target.steps:
            raise ValueError(f"Step {self.name!r} already exists in the target container")

    def add_to_study(self):
        """Register the configured plugin as a solution step."""
        if self.group is not None:
            raise ValueError("Generator has a group; call add_to_group()")
        self._check_registration(self.solution)
        self.solution.add_step(
            step_name=self.name,
            step=self.plugin_class,
            plugin_options=self._plugin_options
        )
        self.added_to_solution = True

    def add_to_group(self):
        """Register the configured plugin in its parent group."""
        if self.group is None:
            raise ValueError("Generator has no group; call add_to_study()")
        self._check_registration(self.group)
        self.group.add_step(
            step_name=self.name,
            step=self.plugin_class,
            plugin_options=self._plugin_options
        )
        self.added_to_group = True

    def solution_output(self, name):
        """Solution path for tracking a generator output."""
        return group_step_path(self.solution, self.group, self.output(name))

    def input(self, name: str) -> str:
        """Return the qualified path for a declared input."""
        if '[' in name:
            test_name=name.split('[')[0]
        else:
            test_name=name
        if test_name not in self._input_dict:
            raise KeyError(
                f"Input {name!r} is not defined for step {self.name!r}"
            )

        return f"{self.name}.{name}"

    def output(self,name:str)->str:
        """Return the qualified path for a declared output."""
        if '[' in name:
            test_name=name.split('[')[0]
        else:
            test_name=name
        if test_name not in self._output_dict:
            raise KeyError(
                f"Output {name!r} is not defined for step {self.name!r}"
            )

        return f"{self.name}.{name}"



class UnitConverterGenerator(PluginGenerator):
    """Build a unit-conversion plugin and its input-to-output mappings."""
    plugin_class = UnitConverterPlugin

    def __init__(self,name,solution,group=None):

        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name,solution,group=group)

        self.in_to_out_factor={}
        self._converted_outputs=[]
        self._plugin_options['input_to_output_conversion_factor_dict']=self.in_to_out_factor


    def define_conversion(self,input_name,output_name,mult_factor):

        """Map an input to an output using a multiplicative factor."""
        if output_name in self._converted_outputs:
            raise ValueError(f'An output can only belong to one conversion, the'
                             f'var {output_name} has already been added')
        if input_name not in self.in_to_out_factor:
            self.in_to_out_factor[input_name]=[(output_name,mult_factor)]

        else:
            self.in_to_out_factor[input_name].append((output_name,mult_factor))

        self._converted_outputs.append(output_name)

class SummationPluginGenerator(PluginGenerator):
    """Build a weighted-sum plugin."""
    plugin_class = SummationPlugin

    def __init__(self,name,solution,group=None):
        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name,solution,group=group)

        self._input_to_factor={}
        self._plugin_options["input_to_factor"]=self._input_to_factor


    def add_input(self, name, var_type,default_value,factor=1.0):
        """Declare an input variable for the generated plugin."""
        super().add_input(name, var_type, default_value)
        self._input_to_factor[name]=factor

    def add_float_input(self, name, default_value=0.0, factor=1.0):
        """Declare a floating-point input variable."""
        self.add_input(name, float, default_value, factor=factor)

    def add_int_input(self, name, default_value=1, factor=1.0):
        """Declare an integer input variable."""
        self.add_input(name, int, default_value, factor=factor)

class DivisionPluginGenerator(PluginGenerator):
    """Build a division plugin with an optional factor per output."""
    plugin_class = DivisionPlugin

    def __init__(self,name,solution,group=None):
        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name,solution,group=group)

        self._output_to_factor={}
        self._plugin_options['output_to_factor']=self._output_to_factor

    def add_output(self, name, var_type, default_value, tracked=False,factor=None):
        """Declare an output variable for the generated plugin."""
        if factor == 0:
            raise ValueError(
                f"Division factor for output {name!r} cannot be zero."
            )
        super().add_output(name, var_type, default_value, tracked)
        self._output_to_factor[name]=factor

    def add_float_output(self, name, default_value=0.0, tracked=False, factor=None):
        """Declare a floating-point output variable."""
        self.add_output(name, float, default_value, tracked=tracked, factor=factor)

    def add_int_output(self, name, default_value=0, tracked=False, factor=None):
        """Declare an integer output variable."""
        self.add_output(name, int, default_value, tracked=tracked, factor=factor)


class ExpressionPluginGenerator(PluginGenerator):
    """Build an expression plugin and configure its derivative method."""
    plugin_class=ExpressionPlugin

    def __init__(self,name,solution,group=None):
        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name,solution,group=group)

        self._output_to_expression={}

        self._plugin_options['output_to_expression']=self._output_to_expression
        self._plugin_options['partials_method']='exact'

    def add_input(self, name, var_type, default_value):
        """Declare an input variable for the generated plugin."""
        if name == "pi":
            raise ValueError(
                "ExpressionPlugin input name 'pi' is reserved for the "
                "mathematical constant pi."
            )
        super().add_input(name, var_type, default_value)

    def set_partials_method(self, method):
        """Select exact or finite-difference expression derivatives."""
        if method not in ('exact', 'fd'):
            raise ValueError("Partials method must be either exact or fd")
        self._plugin_options['partials_method']=method

    def add_output(self, name, var_type, default_value, expr, tracked=False):
        """Declare an output variable for the generated plugin."""
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError(
                f"Expression for output {name!r} must be a non-empty string."
            )
        super().add_output(name, var_type, default_value, tracked)
        self._output_to_expression[name]=expr

    def add_float_output(self, name, expr, default_value=0.0, tracked=False):
        """Declare a floating-point output variable."""
        self.add_output(name, float, default_value, expr=expr, tracked=tracked)

    def add_int_output(self, name, expr, default_value=0, tracked=False):
        """Declare an integer output variable."""
        self.add_output(name, int, default_value, expr=expr, tracked=tracked)

class LinearSystemofEqGenerator(PluginGenerator):
    """Build a fixed-matrix linear-system plugin."""
    plugin_class = LinearSystemofEquations

    def __init__(self, name, solution: Solution, num_eq, group=None):
        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name, solution, group=group)
        if not isinstance(num_eq, int) or isinstance(num_eq, bool) or num_eq < 1:
            raise ValueError('num_eq must be a positive integer.')
        self.num_eq = num_eq
        self._plugin_options['num_eq'] = num_eq

    def add_bvec(self, name):
        """Declare the right-hand-side vector of the linear system."""
        self._plugin_options['bvec_name'] = name
        self.add_input(name, np.ndarray, np.zeros(self.num_eq))

    def add_Amat(self, A_mat):
        """Set the fixed coefficient matrix of the linear system."""
        A_mat = np.array(A_mat, dtype=float, copy=True)
        if A_mat.shape != (self.num_eq, self.num_eq):
            raise ValueError(f'A_mat must have shape ({self.num_eq}, {self.num_eq}).')
        self._plugin_options['A_mat'] = A_mat

    def add_xvec(self, name, tracked=False):
        """Declare the solution vector of the linear system."""
        self._plugin_options['xvec_name'] = name
        self.add_output(name, np.ndarray, np.zeros(self.num_eq), tracked=tracked)

    def _check_ready(self):
        """Check that all required linear-system data has been supplied."""
        for option, method in (('bvec_name', 'add_bvec'), ('A_mat', 'add_Amat'),
                               ('xvec_name', 'add_xvec')):
            if option not in self._plugin_options:
                raise ValueError(f'Call {method}() before registering the plugin.')

    def add_to_study(self):
        """Register the configured plugin as a solution step."""
        self._check_ready()
        super().add_to_study()

    def add_to_group(self):
        """Register the configured plugin in its parent group."""
        self._check_ready()
        super().add_to_group()


#--------------------------------Group stuff------------------------------------

class GroupGenerator:

    """Create a group and expose paths to its child step variables."""
    def __init__(self,name,solution):
        """Initialize the instance with its configuration and owning solution."""
        self.name=name
        self.solution=solution
        self.group=Group()
        self.solution.add_step(self.group,name)

    def group_input(self, step_generator, name):
        """Return a child step input path within this group."""
        if step_generator.group is not self.group:
            raise ValueError("Generator is not a child of this group")
        return step_generator.input(name)

    def group_output(self, step_generator, name):
        """Return a child step output path within this group."""
        if step_generator.group is not self.group:
            raise ValueError("Generator is not a child of this group")
        return step_generator.output(name)


class DerivativeGroupGenerator:

    """Placeholder for a derivative group generator; currently unimplemented."""
    def __init__(self,name,solution):
        """Unimplemented: derivative group setup is reserved for future work."""
        pass

class CoupledGroupGenerator:
    '''
    automatic creation of groups where coupling is meant to happen between two
    plugins?
    '''

# fan in fan out
#blending
#normalization group
