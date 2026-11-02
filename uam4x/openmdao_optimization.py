from tabnanny import check

import openmdao.api as om
from uam4x.study import Study
from uam4x.solution import Solution
import os
import shutil
import numpy as np
from uam4x.utils import (
    is_continuous_type_or_value,
    is_scalar_type_or_value,
    split_dotted_path_and_indices,
    is_numpy_array,
    parse_variable_reference
)
from uam4x.log import get_log
from uam4x.utils import log
from uam4x.results import Result
from uam4x.solution import Group
from uam4x.linking import uam4xLinkHandler,uam4xVariable

DRIVER_NAME_TO_DRIVER = {
    "pyOptSparseDriver": om.pyOptSparseDriver,
    "ScipyOptimizeDriver": om.ScipyOptimizeDriver,
}

CYCLE_SOLVER_NAME_TO_SOLVER = {"NonlinearBlockGS": om.NonlinearBlockGS}


class Optimization:
    """Build and execute an OpenMDAO optimization for a UAM4X analysis.

    This class is the internal bridge between the user-facing UAM4X
    optimization definition and the OpenMDAO problem that executes it. It
    constructs the OpenMDAO model from study data, plugins, links, and
    optimization settings, then returns a :class:`uam4x.results.Result`.

    Parameters
    ----------
    study : Study
        Study object containing vehicles, solution, and optimization settings.
    solution : Solution
        Solution workflow used to build the OpenMDAO model.
    state : dict
        Optimization state dictionary for the current analysis.
    plugins : dict
        Generated plugin-instance mapping aligned with the solution structure.
    out_dir : str or path-like
        Directory used for OpenMDAO and plugin execution artifacts.
    check_partials : bool, optional
        Flag set by the execution engine during its one-off partial-check run.
        When True, partial-check-specific logic is enabled inside the OpenMDAO
        execution path.

    """

    log = get_log(log, object_name="optimization")

    def __init__(
        self,
        study: Study,
        solution: Solution,
        state: dict,
        plugins: dict,
        out_dir: str | os.PathLike[str],
        check_partials: bool = False,
    ) -> None:
        """Initialize an OpenMDAO optimization wrapper.

        Parameters
        ----------
        study : Study
            Study object containing vehicles, solution, and optimization settings.
        solution : Solution
            Solution workflow used to build the OpenMDAO model.
        state : dict
            Optimization state dictionary for the current analysis.
        plugins : dict
            Generated plugin-instance mapping aligned with the solution structure.
        out_dir : str or path-like
            Directory used for OpenMDAO and plugin execution artifacts.
        check_partials : bool, optional
            Flag set by the execution engine during its one-off partial-check run.
            When True, partial-check-specific logic is enabled inside the OpenMDAO
            execution path.

        """
        self.study = study
        self.solution = solution
        self.state = state
        self.plugins = plugins
        self.out_dir = out_dir
        self.check_partials = check_partials
        self.log = Optimization.log

    def run_problem(self) -> "Result | om.Problem":
        """Build, execute, and collect results from an OpenMDAO optimization.

        Returns
        -------
        Result or openmdao.api.Problem
            A result containing tracked outputs, design-variable values,
            convergence status, and template checksums. When ``check_partials``
            is enabled, returns the configured OpenMDAO problem so the caller
            can invoke ``check_partials`` on it.

        Notes
        -----
        The OpenMDAO problem is built from the current study solution, including
        vehicle parameters, plugin steps, links, design variables, constraints,
        and the objective.

        """
        problem = om.Problem()
        self.log.debug("run problem")

        # make vehicle step
        used_vehicle_param_names = self.solution.get_all_used_vehicle_params()
        vehicle_val_dict = {}
        vehicle_type_dict = {}
        for param_name in used_vehicle_param_names:
            if param_name in self.state:
                value = self.state[param_name]
            else:
                value = self.study.get_value(param_name)
            vehicle_val_dict[param_name] = value
            vehicle_type_dict[param_name] = self.study.get_param_type(param_name)
        comp = problem.model.add_subsystem(
            "indep_var", OpenmdaoPreComp(), promotes=["*"]
        )
        comp.options["vehicle_val_dict"] = vehicle_val_dict
        comp.options["vehicle_type_dict"] = vehicle_type_dict

        # make plugin steps
        for step_name, data in self.solution._execution_steps.items():
            work_dir = os.path.join(self.out_dir, step_name)

            if isinstance(data, dict):
                input_names = set(
                    [
                        step_name + "." + input_name
                        for input_name in self.plugins[
                            step_name
                        ].plugin_input_names.keys()
                    ]
                )
                output_names = set(
                    [
                        step_name + "." + output_name
                        for output_name in self.plugins[
                            step_name
                        ].plugin_output_names.keys()
                    ]
                )
                comp = problem.model.add_subsystem(
                    step_name, UniversalOpenmdaoComp(), promotes=["*"]
                )
                comp.options["plugin"] = self.plugins[step_name]
                comp.options["input_names"] = input_names
                comp.options["output_names"] = output_names
                # clean the work dir between runs
                comp.options["work_dir"] = work_dir
                if os.path.isdir(work_dir):
                    shutil.rmtree(work_dir)
                os.makedirs(work_dir)
                comp.options["step_name"] = step_name

            elif isinstance(data, Group):
                group = problem.model.add_subsystem(
                    step_name, UniversalOpenMDAOGroup(), promotes=["*"]
                )
                group.options["group"] = data
                group.options["step_name"] = step_name
                group.options["work_dir"] = work_dir
                group.options["links"] = data.links
                group.options["cycle_solver_options"] = self.state.get("cycle_solver_options", {})
                if isinstance(self.plugins[step_name], dict):
                    group.options["sub_plugins"] = self.plugins[step_name]
                else:
                    group.options["sub_plugins"] = None

        # Add connections

        #general idea:
            #1.) when the output array has indices in a link, you need to
            #create a component, the values themselves link to a version of themselves
            #and then are transformed via the component into the entire array
        links=self.solution.links
        dependent_ind_names_to_sources, dependent_ind_names_to_data=organize_links(links=links,
                      vehicle_val_dict=vehicle_val_dict,
                      uamgroup=self.solution,
                      group_name='',
                      plugins=self.plugins,
                      mdao_group=problem.model
                      )
        if dependent_ind_names_to_sources:
            connect_indexed_variables(dependent_ind_names_to_sources=dependent_ind_names_to_sources,
                                      dependent_ind_names_to_data=dependent_ind_names_to_data,
                                      mdao_group=problem.model)

        # add desvars
        for name, data in self.state["des_vars"].items():
            dv_type = self.study.get_param_type(name)
            lower, upper = data["lower"], data["upper"]
            indices = data["indices"]

            # additional array wrapping here for floats, lists already wrapped
            if dv_type == np.ndarray:
                if lower is not None:
                    lower = np.array(lower)
                if upper is not None:
                    upper = np.array(upper)
            else:
                if lower is not None:
                    lower = dv_type(lower)
                if upper is not None:
                    upper = dv_type(upper)
            scaling = {
                key: data[key] for key in ('ref', 'ref0')
                if data.get(key) is not None
            }
            problem.model.add_design_var(
                name.replace(".", "~"), lower=lower, upper=upper, indices=indices,
                **scaling,
            )

        # add constraints
        for name, data in self.state["constraints"].items():

            const_type = self.solution.get_var_type(name, self.plugins)
            lower, upper = data["lower"], data["upper"]
            equals = data.get("equals")
            indices = data["indices"]

            # additional array wrapping here for floats, lists already wrapped
            if const_type == np.ndarray:
                if lower is not None:
                    lower = np.array(lower)
                if upper is not None:
                    upper = np.array(upper)
                if equals is not None:
                    equals = np.array(equals)
            else:
                if lower is not None:
                    lower = const_type(lower)
                if upper is not None:
                    upper = const_type(upper)
                if equals is not None:
                    equals = const_type(equals)
            constraint_options = {"lower": lower, "upper": upper, "indices": indices}
            if equals is not None:
                constraint_options["equals"] = equals
            problem.model.add_constraint(name.replace(".", "~"), **constraint_options)

        result = Result(state=self.state)
        # add template hashes to results before template files are run in optimization
        for plug_name, plug in self.solution.recurse_generated_plugins(
            self.plugins
        ).items():
            if plug._template_to_checksum:
                for path, checksum in plug._template_to_checksum.items():
                    result._add_template_checksum(path, checksum)

        # add objectives
        problem.model.add_objective(self.state["objective_name"].replace(".", "~"))
        driver_name = self.state["driver"]
        driver_cls = DRIVER_NAME_TO_DRIVER[driver_name]
        problem.driver = driver_cls()

        for name, value in self.state["driver_options"].items():
            problem.driver.options[name] = value
        if hasattr(problem.driver, "opt_settings"):
            for name, value in self.state["optimizer_options"].items():
                problem.driver.opt_settings[name] = value

        approx = self.solution._approx_totals
        if approx["compute"]:
            problem.model.approx_totals(
                method=approx["method"], form=approx["form"], step=approx["step"]
            )

        problem.model.options["auto_order"]=True
        problem.setup()
        for name, data in self.state["des_vars"].items():
            if data['initial_value'] is not None:
                value=vehicle_val_dict[name]
                initial_value=data['initial_value']
                if data["indices"] is not None:
                    value = np.array(vehicle_val_dict[name], copy=True)
                    value[data["indices"]] = data["initial_value"]
                else:
                    value=initial_value
                problem.set_val(name.replace(".", "~"),value)
        problem.run_driver()

        if self.check_partials:
            return problem

        if problem.driver.fail:
            result._add_convergence_status(False)
        else:
            result._add_convergence_status(True)

        for tracked_output in self.solution.tracked_outputs + list(
            self.state["des_vars"].keys()
        ):
            if "[" in tracked_output:
                no_ind_tracked_output, inds = split_dotted_path_and_indices(
                    tracked_output
                )

                result._add_result(
                    tracked_output,
                    problem.get_val(no_ind_tracked_output.replace(".", "~"))[inds],
                )
            else:
                name_start = tracked_output.split(".")[0]
                if name_start == "vehicles":
                    var_type = self.study.get_param_type(tracked_output)
                else:
                    var_type = self.solution.get_var_type(
                        tracked_output, self.plugins
                    )
                value = problem.get_val(tracked_output.replace(".", "~"))

                # assumes outputs to openmdao are essentially always arrays for scalars
                # strings might be treated differently
                if not var_type is np.ndarray:
                    if is_numpy_array(value) and is_scalar_type_or_value(var_type):
                        value = np.asarray(value).item()
                    else:
                        value = var_type(value)

                result._add_result(tracked_output, value)

        return result

def qualify_name(name: str, group_name: str | None) -> str:
    """Convert a UAM4X path to an OpenMDAO variable name.

    Prefix local names with ``group_name`` when given, and replace dotted
    path separators with OpenMDAO's internal ``~`` separator.
    """
    if name.startswith('vehicles.'):
        return name.replace(".","~")
    if group_name:
        return f"{group_name}.{name}".replace('.','~')
    return name.replace(".","~")

def organize_links(
    links: list[list[str]],
    vehicle_val_dict: dict,
    uamgroup: Group,
    group_name: str | None,
    plugins: dict,
    mdao_group: om.Group,
) -> tuple[dict, dict]:
    """Connect direct links and collect indexed target links.

    Direct links are connected on ``mdao_group``. Indexed target links are
    returned as source and target metadata for ``ArrayAdapter`` creation.
    """
    dependent_ind_names_to_sources={}
    dependent_ind_names_to_data={}
    for independent_name, dependent_name in links:
        independent=uam4xVariable(independent_name)
        dependent=uam4xVariable(dependent_name)
        dependent_cleaned_name=qualify_name(dependent.cleaned_name,group_name=group_name)
        independent_cleaned_name=qualify_name(independent.cleaned_name,group_name=group_name)
        if dependent.has_inds:
            #must be array since indexed
            dependent_data=uamgroup.get_var_data(
                dependent.cleaned_name,plugins
                )

            if independent.cleaned_name.startswith('vehicles.'):
                val=vehicle_val_dict[independent.cleaned_name]
                independent_data={'shape':np.shape(val),'value':val}

            else:
                temp_data=uamgroup.get_var_data(
                    independent.cleaned_name,plugins
                )
                if temp_data['type']==np.ndarray:
                    independent_data={'shape':temp_data['shape'],'value':temp_data['value']}
                else:
                    independent_data={'value':temp_data['value'],'shape':np.shape(temp_data['value'])}

            if dependent_cleaned_name not in dependent_ind_names_to_data:
                dependent_ind_names_to_data[dependent_cleaned_name]={
                    'shape':dependent_data['shape'],
                    'value':dependent_data['value']
                }
                dependent_ind_names_to_sources[dependent_cleaned_name]={
                    independent_cleaned_name:{
                        'src_inds':[independent.inds],
                        'target_inds':[dependent.inds],
                        'src_shape':independent_data['shape'],
                        'src_value':independent_data['value']
                                              }
                }
            else:
                dep_data=dependent_ind_names_to_sources[dependent_cleaned_name]
                if independent_cleaned_name not in dep_data:
                    dep_data[independent_cleaned_name]={
                            'src_inds':[independent.inds],
                            'target_inds':[dependent.inds],
                            'src_shape':independent_data['shape'],
                            'src_value':independent_data['value']
                                                  }
                else:
                    indep_data=dep_data[independent_cleaned_name]
                    indep_data['src_inds'].append(independent.inds)
                    indep_data['target_inds'].append(dependent.inds)
        else:
            mdao_group.connect(
                            independent_cleaned_name,
                            dependent_cleaned_name,
                            src_indices=independent.inds
                        )

    return dependent_ind_names_to_sources,dependent_ind_names_to_data

def connect_indexed_variables(
    dependent_ind_names_to_sources: dict,
    dependent_ind_names_to_data: dict,
    mdao_group: om.Group,
) -> None:
    """Add array adapters and connections for indexed target links."""
    for dependent_name,input_data in dependent_ind_names_to_sources.items():
                    step_name=dependent_name.replace('.','_').replace('~','_')+'_'+'array_helper'
                    # create component
                    array_helper=mdao_group.add_subsystem(step_name,ArrayAdapter(),promotes=['*'])
                    #set options
                    array_helper.options["output_data"]=dependent_ind_names_to_data[dependent_name]
                    array_helper.options["input_data"]=input_data
                    array_helper.options["step_name"]=step_name
                    mdao_dependent_name=dependent_name.replace('.','~')
                    array_helper.options["output_name"]=step_name+'~'+mdao_dependent_name
                    #create connections
                    for input_name in input_data.keys():
                        clean_input_name,inds=parse_variable_reference(input_name)
                        mdao_input_name=clean_input_name.replace('.','~')
                        mdao_group.connect(mdao_input_name,
                                              step_name+'~'+mdao_input_name,src_indices=inds)

                    mdao_group.connect(step_name+'~'+mdao_dependent_name,
                                        mdao_dependent_name)


class OpenmdaoPreComp(om.IndepVarComp):
    """Expose study vehicle parameters to OpenMDAO as independent variables."""

    log = get_log(log, object_name="OpenmdaoPreComp")

    def initialize(self) -> None:
        """Declare OpenMDAO options for vehicle values and declared types."""
        self.log = get_log(log, object_name="OpenmdaoPreComp")
        self.log.debug("initialize")
        self.options.declare("vehicle_val_dict")
        self.options.declare("vehicle_type_dict")

    def setup(self) -> None:
        """Add OpenMDAO outputs for vehicle parameters.

        Notes
        -----
        Continuous-valued vehicle parameters are added as standard outputs.
        Discrete-valued parameters are added as discrete outputs.

        """
        self.log.debug("setup")
        vehicle_val_dict = self.options["vehicle_val_dict"]
        vehicle_type_dict = self.options["vehicle_type_dict"]
        for param_name, value in vehicle_val_dict.items():
            param_type = vehicle_type_dict[param_name]
            if not is_continuous_type_or_value(param_type):
                self.add_discrete_output(param_name.replace(".", "~"), val=value)
            else:
                self.add_output(param_name.replace(".", "~"), val=value)


class UniversalOpenmdaoComp(om.ExplicitComponent):
    """Wrap a UAM4X plugin as an OpenMDAO explicit component."""

    log = get_log(log, object_name="UniversalOpenmdaoComp")

    def initialize(self) -> None:
        """Declare OpenMDAO options for the wrapped plugin and execution context."""
        self.log = UniversalOpenmdaoComp.log
        self.log.debug("initialize")
        self.options.declare("plugin")
        self.options.declare("input_names")
        self.options.declare("output_names")
        self.options.declare("work_dir")
        self.options.declare("step_name")

    def setup(self) -> None:
        """Declare OpenMDAO variables and partials for the wrapped plugin.

        Notes
        -----
        Plugin inputs and outputs are mapped to continuous or discrete OpenMDAO
        variables based on their declared UAM4X types. If the plugin declares
        partials, those declarations are converted into OpenMDAO partials here.

        """
        self.log.debug("setup")
        self.plugin = self.options["plugin"]
        self.input_names = self.options["input_names"]
        self.output_names = self.options["output_names"]
        self.work_dir = self.options["work_dir"]
        self.step_name = self.options["step_name"]
        cont_input = False
        cont_output = False
        if self.plugin._check_partials_options:
            self.set_check_partial_options(
                wrt=["*"], form=self.plugin.check_form, step=self.plugin.check_step
            )

        for input_name in self.input_names:
            # plugin_name = input_name.split('.')[0]
            plugin_name = self.options["step_name"]
            start_ind = len(plugin_name.split("."))
            name = ".".join(input_name.split(".")[start_ind:])
            metadata = self.plugin.plugin_input_names[name]
            input_type = metadata["type"]
            default = metadata["value"]
            if not is_continuous_type_or_value(input_type):
                if default is None:
                    default = 1 if input_type is int else "temp"
                self.add_discrete_input(input_name.replace(".", "~"), val=default)
            else:
                cont_input = True
                options = {"shape": metadata["shape"]}
                if default is not None:
                    options["val"] = default
                self.add_input(input_name.replace(".", "~"), **options)

        for output_name in self.output_names:
            plugin_name = self.options["step_name"]
            start_ind = len(plugin_name.split("."))
            name = ".".join(output_name.split(".")[start_ind:])
            metadata = self.plugin.plugin_output_names[name]
            output_type = metadata["type"]
            default = metadata["value"]

            if not is_continuous_type_or_value(output_type):
                if default is None:
                    default = 1 if output_type is int else "temp"
                self.add_discrete_output(output_name.replace(".", "~"), val=default)
            else:
                cont_output = True
                options = {"shape": metadata["shape"]}
                if default is not None:
                    options["val"] = default
                self.add_output(output_name.replace(".", "~"), **options)

        if cont_input and cont_output:
            partials = self.plugin.build_partials(
                self.input_names, self.output_names, self.step_name
            )
            self.plugin.categorize_partials()
            for pair, data in partials.items():
                # note since strings are immutable
                # self.step_name.replace does not modify original step name attribute
                of = self.step_name.replace(".", "~") + "~" + pair[0].replace(".", "~")
                wrt = self.step_name.replace(".", "~") + "~" + pair[1].replace(".", "~")
                method = data["method"]
                step, form = data["step"], data["form"]
                rows, cols = data["rows"], data["cols"]
                self.declare_partials(
                    of=of,
                    wrt=wrt,
                    method=method,
                    step=step,
                    form=form,
                    rows=rows,
                    cols=cols,
                )

    def compute(self, inputs, outputs, discrete_inputs={}, discrete_outputs={}) -> None:
        """Execute the wrapped plugin from OpenMDAO-provided inputs.

        Notes
        -----
        OpenMDAO variable names are converted back to the plugin's local UAM4X
        names before calling ``run_plugin``. Returned plugin outputs are then
        written into continuous or discrete OpenMDAO outputs.

        """
        self.log.debug("compute")
        self.plugin_dict = {}
        plugin_name = self.step_name.split(".")
        start_ind = len(plugin_name)
        for name, value in inputs.items():
            var_name = name.replace("~", ".")
            var_name = ".".join(var_name.split(".")[start_ind:])
            var_type = self.plugin.plugin_input_names[var_name]["type"]
            if not self.plugin.plugin_input_names[var_name]["type"] is np.ndarray:
                if is_numpy_array(value) and is_scalar_type_or_value(var_type):
                    value = np.asarray(value).item()
                else:
                    value = self.plugin.plugin_input_names[var_name]["type"](value)
            self.plugin_dict[var_name] = value
        for name, value in discrete_inputs.items():
            var_name = name.replace("~", ".")
            var_name = ".".join(var_name.split(".")[start_ind:])
            self.plugin_dict[var_name] = value
        results = self.plugin.run_plugin(self.plugin_dict, self.work_dir)
        for name, value in results.items():
            name_with_plugin = self.step_name + "." + name
            if name_with_plugin in self.output_names:
                if not is_continuous_type_or_value(
                    self.plugin.plugin_output_names[name]["type"]
                ):
                    discrete_outputs[name_with_plugin.replace(".", "~")] = value
                else:
                    outputs[name_with_plugin.replace(".", "~")] = value

    def compute_partials(self, inputs, partials, discrete_inputs={}) -> None:
        """Populate OpenMDAO partial derivatives from plugin-provided values.

        Notes
        -----
        Plugin-reported partial derivatives are written into the OpenMDAO
        partial accumulator using fully qualified OpenMDAO variable names.

        """
        self.log.debug("compute partials")
        if not self.plugin._declare_partials:
            return
        else:
            results = self.plugin.compute_partials(self.plugin_dict)
            for name, value in results.items():
                output_name = self.step_name + "." + name[0]
                input_name = self.step_name + "." + name[1]

                if output_name in self.output_names and input_name in self.input_names:
                    output_name_with_plugin = output_name.replace(".", "~")
                    input_name_with_plugin = input_name.replace(".", "~")
                    partials[output_name_with_plugin, input_name_with_plugin] = value
                else:
                    self.log.error("Missing links somewhere")
                    raise ValueError("Missing links somewhere")


class UniversalOpenMDAOGroup(om.Group):
    """Wrap a UAM4X solution group as an OpenMDAO group."""

    def initialize(self) -> None:
        """Declare OpenMDAO options for the wrapped solution group."""
        self.options.declare("group")
        self.options.declare("step_name")
        self.options.declare("sub_plugins")
        self.options.declare("work_dir")
        self.options.declare("links")
        self.options.declare("cycle_solver_options", default={})

    def setup(self) -> None:
        """Build OpenMDAO subsystems and connections for the wrapped solution group.

        Notes
        -----
        Nested UAM4X groups are wrapped recursively. Group links are converted into
        OpenMDAO connections. If the wrapped group is cyclic, the declared
        nonlinear solver is attached. Approximate total derivatives are enabled
        when requested by the UAM4X group settings.

        """
        group_name = self.options["step_name"]
        uamgroup = self.options["group"]
        sub_plugs = self.options["sub_plugins"]
        work_dir = self.options["work_dir"]
        for step_name, data in uamgroup._execution_steps.items():
            if isinstance(data, dict):
                # Match the single-point runner's directory for each step.
                step_work_dir = os.path.join(
                    os.path.dirname(work_dir),
                    f"{group_name}.{step_name}",
                )
                input_names = set(
                    [
                        group_name + "." + step_name + "." + input_name
                        for input_name in sub_plugs[step_name].plugin_input_names.keys()
                    ]
                )
                output_names = set(
                    [
                        group_name + "." + step_name + "." + output_name
                        for output_name in sub_plugs[
                            step_name
                        ].plugin_output_names.keys()
                    ]
                )
                comp = self.add_subsystem(
                    step_name, UniversalOpenmdaoComp(), promotes=["*"]
                )
                comp.options["plugin"] = sub_plugs[step_name]
                comp.options["input_names"] = input_names
                comp.options["output_names"] = output_names
                # Clean this step's work directory without deleting sibling outputs.
                comp.options["work_dir"] = step_work_dir
                if os.path.isdir(step_work_dir):
                    shutil.rmtree(step_work_dir)
                os.makedirs(step_work_dir)
                comp.options["step_name"] = group_name + "." + step_name
            elif isinstance(data, Group):
                nested_om_group = self.add_subsystem(
                    step_name, UniversalOpenMDAOGroup(), promotes=["*"]
                )
                nested_om_group.options["group"] = data
                nested_om_group.options["step_name"] = group_name + "." + step_name
                nested_om_group.options["work_dir"] = work_dir
                nested_om_group.options["links"] = data.links
                nested_om_group.options["cycle_solver_options"] = self.options["cycle_solver_options"]
                if isinstance(sub_plugs[step_name], dict):
                    nested_om_group.options["sub_plugins"] = sub_plugs[step_name]
                else:
                    nested_om_group.options["sub_plugins"] = None

        dependent_ind_names_to_sources,dependent_ind_names_to_data=organize_links(links=self.options['links'],
                      vehicle_val_dict={},
                      uamgroup=uamgroup,
                      plugins=sub_plugs,
                      mdao_group=self,
                      group_name=group_name
                      )
        if dependent_ind_names_to_sources:
                connect_indexed_variables(dependent_ind_names_to_sources=dependent_ind_names_to_sources,
                                          dependent_ind_names_to_data=dependent_ind_names_to_data,
                                          mdao_group=self)

        if uamgroup._is_cycle:
            self.nonlinear_solver = CYCLE_SOLVER_NAME_TO_SOLVER[uamgroup.cycle_solver]()
            for name, value in self.options["cycle_solver_options"].items():
                self.nonlinear_solver.options[name] = value

        approx_totals_dict = uamgroup._approx_totals
        if approx_totals_dict["compute"]:
            method = approx_totals_dict["method"]
            form = approx_totals_dict["form"]
            step = approx_totals_dict["step"]
            self.approx_totals(method=method, form=form, step=step)

        self.options["auto_order"]=True


class ArrayAdapter(om.ExplicitComponent):
    """Assemble an array output from one or more indexed OpenMDAO inputs."""

    #you essentially need to know all of the variables connecting
    # into a particular array at the outset, you need to collect them all
    # then declare them as inputs to that particular output then you just assign
    def initialize(self) -> None:
        """Declare the target array and source-link metadata options."""
        self.options.declare("output_data")
        self.options.declare("input_data")
        self.options.declare("step_name")
        self.options.declare("output_name")

    def setup(self) -> None:
        """Declare adapter inputs, output shape, and finite-difference partials."""
        step_name=self.options["step_name"]
        for input_name,data in self.options["input_data"].items():
            if data['src_value'] is None:
                self.add_input(step_name+'~'+input_name.replace('.','~'),val=np.zeros(data['src_shape']),shape=data['src_shape'])
            else:
                self.add_input(step_name+'~'+input_name.replace('.','~'),val=data['src_value'],shape=data['src_shape'])

        output_data=self.options["output_data"]
        output_name=self.options["output_name"]
        #todo: add analytical sparse matrix partials
        self.add_output(output_name,val=output_data["value"],shape=output_data["shape"])

        self.declare_partials('*','*',method='fd')
    def compute(self,inputs,outputs,discrete_inputs={},discrete_outputs={}) -> None:
        """Fill target indices from connected inputs, retaining other values."""
        step_name=self.options["step_name"]
        output_name=self.options["output_name"]
        shape = self.options["output_data"]["shape"]
        value = self.options["output_data"]["value"]

        if value is None:
            output_val = np.zeros(shape)
        elif np.shape(value) == ():
            output_val = np.full(shape, value)
        else:
            output_val = np.array(value, copy=True)

        input_data=self.options["input_data"]
        for input_name,data in input_data.items():
            mdao_input_name=step_name+'~'+input_name.replace('.','~')
            for src_ind,target_ind in zip(data['src_inds'],data['target_inds']):
                if src_ind is not None:
                    output_val[target_ind]=inputs[mdao_input_name][src_ind]
                else:
                    output_val[target_ind]=inputs[mdao_input_name]

        outputs[output_name]=output_val

    #def compute_partials(self,inputs,partials,discrete_inputs={}):
    #    for input_name in self.options["input_data"]:
    #        mdao_input_name=self.options["step_name"]+'~'+input_name.replace('.','~')
    #        partials[self.options["output_name"],mdao_input_name]=np.eye(self.options['output_data']['shape'])
