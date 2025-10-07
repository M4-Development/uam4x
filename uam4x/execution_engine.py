from uam4x.study import Study
import os
from uam4x.openmdao_optimization import Optimization
import copy

from uam4x.utils import check_continuous, get_param_type

class Engine:
    """
    Execution engine for running UAM4X studies and trade studies.

    The Engine class manages the execution of study solutions, including
    optimization runs, single-point analyses, and trade studies. It handles
    plugin initialization, state management, and result processing.

    Parameters
    ----------
    study : Study
        The study object containing all configuration and data.
    work_dir : str
        Working directory for execution outputs.
    num_processes : int, optional
        Number of processes for parallel execution, by default 1.

    Attributes
    ----------
    study : Study
        The study object being executed.
    num_processes : int
        Number of processes for parallel execution.
    study_results : dict
        Dictionary storing study execution results.
    sweep_combos : dict
        Dictionary storing parameter sweep combinations.
    current_sweep_values : dict
        Dictionary storing current sweep parameter values.
    work_dir : str
        Working directory for execution outputs.
    """

    def __init__(self, study:Study, work_dir, num_processes=1) -> None:
        self.study = study
        self.num_processes = num_processes
        self.study_results = {} # change once there is a better way to store outputs
        self.sweep_combos = {}
        self.current_sweep_values = {}
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def check_partials(self, trade_name, form='forward', step=1.e-6):
        """
        Check partial derivatives for a trade study.

        This method validates the partial derivatives of the optimization
        problem by running a finite difference check. It creates a temporary
        directory and runs the optimization with partial checking enabled.

        Parameters
        ----------
        trade_name : str
            Name of the trade study to check partials for.
        form : str, optional
            Form of finite difference ('forward', 'backward', 'central'),
            by default 'forward'.
        step : float, optional
            Step size for finite difference calculation, by default 1.e-6.

        Raises
        ------
        RuntimeError
            If circular solution sequence is detected.
        """

        #Standard setup:
        self.sort_steps()
        tradestudy = self.study.control.trade_studies[trade_name]

        # Sample state, doesn't really matter here, special check
        # partials directory created
        sample_state = copy.deepcopy(tradestudy.get_states()[0])
        out_dir = os.path.join(self.work_dir, f"check_partials")
        os.makedirs(out_dir, exist_ok=True)

        plugins = {}
        for step_name, step_info in self.study.solution.steps.items():
            plugin = step_info["plugin"]()
            plugin.setup_plugin(step_info["plugin_options"])
            plugins[step_name] = plugin

        cwd = os.getcwd()
        os.chdir(out_dir)
        opt = Optimization(self.study, sample_state, plugins, out_dir, check_partials=True)
        problem = opt.run_problem()

        problem.check_partials(step=step, form=form, compact_print=True)
        os.chdir(cwd)



    # def are_plugins_threadsafe(self):
    #     for step_name, step_info in self.study.solution.steps.items():
    #         plugin = get_new_plugin_instance(step_info["plugin_type"])
    #         plugin.setup_plugin(step_info["plugin_options"])
    #         if not plugin.is_threadsafe:
    #             return False
    #     return True

    # def init_plugins(self):
    #     plugins = {}
    #     # thread_safe = False
    #     for step_name, step_info in self.study.solution.steps.items():
    #         plugins[step_name] = step_info["plugin"]()
    #         plugins[step_name].setup_plugin(step_info["plugin_options"])
    #     return plugins

    def run_solution(self, state, plugins, iteration=None, check=True):
        """
        Run a solution for a given state.

        Executes either an optimization or single-point analysis based on the
        state configuration. Initializes plugins if not provided and performs
        model validation if requested.

        Parameters
        ----------
        state : dict
            State dictionary containing all parameters and configuration.
        plugins : dict or None
            Dictionary of plugin instances. If None, plugins will be initialized.
        iteration : int, optional
            Iteration number for output directory naming, by default None.
        check : bool, optional
            Whether to perform model validation checks, by default True.

        Notes
        -----
        If iteration is provided, outputs will be saved to 'out_{iteration}'
        subdirectory. Otherwise, outputs go to the main work directory.
        """
        if not iteration is None:
            out_dir = os.path.join(self.work_dir, f"out_{iteration}")
        else:
            out_dir = os.path.join(self.work_dir)

        if not plugins:
            plugins = {}
            for step_name, step_info in self.study.solution.steps.items():
                plugin = step_info["plugin"]()
                plugin.setup_plugin(step_info["plugin_options"])
                plugins[step_name] = plugin

        if check:
            self.check_model(plugins, state)

        os.makedirs(out_dir, exist_ok=True)
        if state['optimization']:
            self.run_optimization(state, plugins, out_dir)
        else:
            self.run_single_point(state, plugins, out_dir)

    def run_optimization(self, state, plugins, out_dir):
        """
        Run an optimization analysis.

        Executes an optimization problem using the OpenMDAO optimization
        framework. Changes to the output directory during execution.

        Parameters
        ----------
        state : dict
            State dictionary containing optimization parameters.
        plugins : dict
            Dictionary of plugin instances for the optimization.
        out_dir : str
            Output directory for optimization results.
        """
        cwd = os.getcwd()
        os.chdir(out_dir)
        opt = Optimization(self.study, state, plugins, out_dir)
        opt.run_problem()

        os.chdir(cwd)

    def run_single_point(self, state, plugins, out_dir):
        """
        Run a single-point analysis.

        Executes each plugin in sequence for a single-point analysis.
        Creates individual work directories for each step and updates
        the state with results from each plugin.

        Parameters
        ----------
        state : dict
            State dictionary containing analysis parameters.
        plugins : dict
            Dictionary of plugin instances for the analysis.
        out_dir : str
            Output directory for analysis results.
        """
        for step_name, step_info in self.study.solution.steps.items():
            plugin = plugins[step_name]
            work_dir = os.path.join(out_dir, step_name)
            os.makedirs(work_dir, exist_ok=True)

            plugin_inputs = self.study.get_plugin_inputs(step_name, state)
            results = plugin.run_plugin(plugin_inputs, work_dir)
            self.update_state_from_results(state, results, step_name)


    def update_state_from_results(self, state, results, step_name):
        """
        Update state dictionary with plugin results.

        Adds results from a plugin execution to the state dictionary
        with the step name as a prefix to avoid naming conflicts.

        Parameters
        ----------
        state : dict
            State dictionary to update.
        results : dict
            Results dictionary from plugin execution.
        step_name : str
            Name of the step that produced the results.
        """
        for name, value in results.items():
            self.study._add_overwrite(state, step_name + "." + name, value)

    def run_tradestudy(self, trade_name, num_processesors=None, commit=False):
        """
        Run a trade study analysis.

        Executes all states in a trade study sequentially. Sorts the solution
        steps to ensure proper execution order and runs each state as a
        separate solution.

        Parameters
        ----------
        trade_name : str
            Name of the trade study to execute.
        num_processesors : int, optional
            Number of processors for parallel execution, by default None.
        commit : bool, optional
            Whether to commit results, by default False.

        Notes
        -----
        The num_processesors parameter is currently not implemented.
        All states are executed sequentially.
        """
        self.sort_steps()
        tradestudy = self.study.control.trade_studies[trade_name]
        # plugins = self.init_plugins()

        # if num_processesors:
        #     if not self.are_plugins_threadsafe():
        #         print("WARNING: NON-THREADSAFE PLUGIN DETECTED, CREATING UNIQUE INSTANCE FOR EACH THREAD")
        #         plugins = None
        #     else:
        #         pass
        plugins = None
        for i, state in enumerate(tradestudy.get_states()):
            self.run_solution(state, plugins, i)

    def sort_steps(self):
        """
        Sort solution steps to ensure proper execution order.

        Uses a topological sort algorithm to determine the correct order
        of plugin execution based on the links between steps. Detects
        circular dependencies and raises an error if found.

        Raises
        ------
        RuntimeError
            If a circular solution sequence is detected.

        Notes
        -----
        This method modifies the study.solution._steps dictionary
        to reflect the sorted execution order.
        """
        step_names = self.study.solution.get_step_names()
        new_step_order = []
        num_steps = 0
        changes = True
        while num_steps < len(self.study.solution.get_step_names()) and changes:
            changes = False
            for step_name in step_names:
                add = True
                for link in self.study.solution.links:
                    left_link_split = link[0].split(".")
                    right_link_split = link[1].split(".")
                    if right_link_split[0] == step_name:
                        if left_link_split[0] in step_names:
                            add = False
                if add:
                    new_step_order.append((step_name, self.study.solution.steps[step_name]))
                    step_names.remove(step_name)
                    num_steps += 1
                    changes = True
        if not changes:
            raise RuntimeError("Circular Solution Sequence Detected!")
        else:
            self.study.solution._steps = dict(new_step_order)

    #def get_param_type(self, dotted_path: str, plugins):
        #'''
        #Get a the type of a given parameter
        #assumes general form vehicles. ... .attribute
        #'''
        #parts = dotted_path.split('.')
        #obj = self.study
        #if parts[0] == 'vehicles':
            #for part in parts:
                ## Try dict access first (for vehicles, parameters, etc.)
                #if isinstance(obj, dict) and part in obj:
                    #obj = obj[part]
                ## Try attribute access
                #elif hasattr(obj, part):
                    #if hasattr(obj,'_modes'):
                        #modes = list(obj._modes.keys())
                    #else:
                        #modes = []
                    #if hasattr(obj, 'get_parameter_type') :
                            #if part not in modes:
                                #return obj.get_parameter_type(part)
                    #else:
                        #obj = getattr(obj, part)
                #else:
                    #raise AttributeError(f"Cannot access '{part}' in path '{dotted_path}'")
        #elif plugins:
            #plugin_name = dotted_path.split('.')[0]
            #part = '.'.join(dotted_path.split('.')[1:])
            #plugin = plugins[plugin_name]
            #if part in  plugin.plugin_input_names:
                #obj = plugin.plugin_input_names[part]['type']
            #elif part in  plugin.plugin_output_names:
                #obj = plugin.plugin_output_names[part]['type']

        #return obj

    def check_model(self, plugins, state):
        """
        Perform comprehensive model validation checks.

        Validates the model configuration by checking links, types,
        design variables, constraints, objectives, and discrete variables
        for optimization problems.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.
        state : dict
            State dictionary containing model configuration.

        Raises
        ------
        ValueError
            If plugin I/O declarations are inconsistent.
        TypeError
            If parameter types are incompatible.
        """
        self.check_links_to_plugins(plugins)
        self.check_link_types(plugins)
        if state['optimization']: #TODO can this not exist?
            self.check_desvars(plugins, state)
            self.check_constraints(plugins, state)
            self.check_objective(plugins, state)
            self.check_optimization_discrete(plugins, state)

    def check_links_to_plugins(self, plugins):
        """
        Validate that links match declared plugin I/O.

        Checks that all input and output names in the solution links
        are properly declared in the corresponding plugins.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.

        Raises
        ------
        ValueError
            If input or output names are not declared in plugins.
        """
        for step_name in self.study.solution.steps:
            # Check that links matched declared plugin I/O
            input_names, output_names = self.study.get_plugin_input_and_output_names(step_name)

            #check coherance of declared I/O from plugin and from branching script
            for input_name in  input_names:
                name = '.'.join(input_name.split('.')[1:])
                if name not  in plugins[step_name].plugin_input_names:
                    raise ValueError(f'The input {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                                     f"Check your plugin's setup() step for I/O declaration")
            for output_name in  output_names:
                name = '.'.join(output_name.split('.')[1:])
                if name not  in plugins[step_name].plugin_output_names:
                    raise ValueError(f'The output {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                                     f"Check your plugin's setup() step for I/O declaration")

    def check_link_types(self, plugins):
        """
        Validate that linked parameters have compatible types.

        Checks that input and output parameters in links have matching
        types to ensure data compatibility between plugins.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.

        Raises
        ------
        TypeError
            If linked parameters have incompatible types.
        """
        bad_links = []
        for link in  self.study.solution.links:
            output_link = link[0]
            input_link = link[1]
            output_type = get_param_type(output_link, self.study, plugins)
            input_type = get_param_type(input_link,self.study, plugins)
            if output_type != input_type:
                bad_links.append((output_link,output_type, input_link, input_type))

        if bad_links:
            msg = f'Link types must match, the following links have different types:\n' + "\n".join(f"{out_}:{out_type}->{in_}:{in_type}" for  out_,out_type, in_,in_type in  bad_links)
            raise TypeError(msg)

    def check_desvars(self, plugins, state):
        """
        Validate design variables for optimization.

        Checks that all design variables are of continuous type (float)
        as required for gradient-based optimization.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.
        state : dict
            State dictionary containing design variable information.

        Raises
        ------
        TypeError
            If any design variable is not of continuous type.
        """
        # add desvars
        for name, lower, upper in zip(
            state['des_var_names'],
            state['des_var_lower'],
            state['des_var_upper'],
            ):

            des_var_type = get_param_type(name,self.study, plugins)
            if not check_continuous(des_var_type):
                raise TypeError(f'The design variable: {name} is of type {des_var_type}\n'
                                f'Only floats allowed as design variables')
    def check_constraints(self, plugins, state):
        """
        Validate constraints for optimization.

        Checks that all constraints are of continuous type (float)
        as required for gradient-based optimization.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.
        state : dict
            State dictionary containing constraint information.

        Raises
        ------
        TypeError
            If any constraint is not of continuous type.
        """
        for name, lower, upper in zip(
            state['constraint_names'],
            state['constraint_lower'],
            state['constraint_upper'],
        ):
            constraint_type = get_param_type(name,self.study, plugins)
            is_continuous = check_continuous(constraint_type)
            if not is_continuous:
                raise TypeError(f'The constraint: {name} is of type {constraint_type}\n'
                                f'Only floats allowed as constraints')
    def check_objective(self, plugins, state):
        """
        Validate objective function for optimization.

        Checks that the objective function is of continuous type (float)
        as required for gradient-based optimization.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.
        state : dict
            State dictionary containing objective information.

        Raises
        ------
        TypeError
            If the objective is not of continuous type.
        """
        objective_type = get_param_type(state["objective_name"], self.study, plugins)
        is_continuous = check_continuous(objective_type)
        if not is_continuous:
            raise TypeError(f'The objective: {state["objective_name"]} is of type {objective_type}\n'
                            f'Only floats allowed as objectives')

    def check_optimization_discrete(self, plugins, state):
        """
        Validate that optimization links are continuous.

        Checks that all links between optimization components use
        continuous variables (float type) as required for gradient-based
        optimization.

        Parameters
        ----------
        plugins : dict
            Dictionary of plugin instances to validate.
        state : dict
            State dictionary containing optimization configuration.

        Raises
        ------
        TypeError
            If any optimization link uses discrete variables.
        """
        # Check for discrete I/O passing
        for link in  self.study.solution.links:
            input_name = link[0]
            output_name = link[1]
            link_input_plugin = input_name.split('.')[0]
            link_output_plugin = output_name.split('.')[0]
            if link_input_plugin in  self.study.solution.steps and  link_output_plugin in  self.study.solution.steps:
                #not checking output since it should be the same as input
                link_input_var_name =  '.'.join(input_name.split('.')[1:])
                input_type = plugins[link_input_plugin].plugin_output_names[link_input_var_name]['type']
                if not check_continuous(input_type):
                    raise TypeError(f"Links between optimization components must be continuous (of type float)\n"
                                    f"Check I/O declarations in plugin: {link_input_plugin}\n"
                                        f"Variable in violation: {input_name}")