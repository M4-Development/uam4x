from uam4x.study import Study
import os
from uam4x.openmdao_optimization import Optimization
import copy
from uam4x.log import get_log
from uam4x.custom_errors import PluginException

import numpy as  np
from uam4x.utils import check_continuous, get_param_type, check_if_scalar,log, parse_index, handle_brackets

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
    log = get_log(log, object_name='engine')
    def __init__(self, study:Study, work_dir, num_processes=1) -> None:
        self.study = study
        self.num_processes = num_processes
        self.study_results = {} # change once there is a better way to store outputs
        self.sweep_combos = {}
        self.current_sweep_values = {}
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.log = Engine.log


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
        self.log.debug('check partials')

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
            self.log.debug(f'setup {step_name} plugin')
            plugin = step_info["plugin"]()
            plugin.setup_plugin(step_info["plugin_options"])
            plugins[step_name] = plugin
            
        # May need to refactor this, but def want a check here. 
        self.check_model(plugins, sample_state)
        
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
        self.log.debug('run solution')
        if not iteration is None:
            out_dir = os.path.join(self.work_dir, f"out_{iteration}")
        else:
            out_dir = os.path.join(self.work_dir)

        if not plugins:
            plugins = {}
            for step_name, step_info in self.study.solution.steps.items():
                self.log.debug(f'setup {step_name} plugin')
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
        self.log.debug('run optimization')
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
        self.log.debug('run single point')
        for step_name, step_info in self.study.solution.steps.items():
            self.log.debug(f'run single point: {step_name}')
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
        self.log.debug('update state from results')
        for name, value in results.items():
            self.log.debug(f'update state from results: {name}')
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
        self.log.debug(f'run tradestudy: {trade_name}')
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
        failed_states = []
        successful_states = []
        for i, state in enumerate(tradestudy.get_states()):

            try:
                self.run_solution(state, plugins,  iteration=i)
                self.log.info(f'state successful: {state}')
                successful_states.append(state)
            except PluginException as e:
                tb = e.__traceback__  # get traceback object from exception
                while tb.tb_next:
                    tb = tb.tb_next  # go to the last traceback frame (where error originated)

                lineno = tb.tb_lineno
                filename = tb.tb_frame.f_code.co_filename
                self.log.error(f"{type(e).__name__}: state failed - {state}")
                failed_states.append(state)
            except Exception as e:
                tb = e.__traceback__  # get traceback object from exception
                while tb.tb_next:
                    tb = tb.tb_next  # go to the last traceback frame (where error originated)

                lineno = tb.tb_lineno
                filename = tb.tb_frame.f_code.co_filename
                self.log.error(f"An error occurred: {e}. Called in file {filename} in line {lineno}")
                raise RuntimeError(f"An error occurred: {e}. Called in file {filename} in line {lineno}")

        # self.log.info(f'succesful states: {successful_states}')
        # self.log.info(f'failed states: {failed_states}')

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
        self.log.debug(f'sort steps')
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
            self.log.error("Circular Solution Sequence Detected!")
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
        self.log.debug('check model')
        self.check_links_to_plugins(plugins)
        self.check_link_types(plugins)
        self.check_link_indices(plugins)
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
        self.log.debug(f'check links to plugins')
        for step_name in self.study.solution.steps:
            # Check that links matched declared plugin I/O
            input_names, output_names = self.study.get_plugin_input_and_output_names(step_name)
            
            #check coherance of declared I/O from plugin and from branching script
            for input_name in  input_names:
                name = '.'.join(input_name.split('.')[1:])
                if name not  in plugins[step_name].plugin_input_names:
                    self.log.error(f'The input {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                                     f"Check your plugin's setup() step for I/O declaration")
                    raise ValueError(f'The input {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                                     f"Check your plugin's setup() step for I/O declaration")
            for output_name in  output_names:
                name = '.'.join(output_name.split('.')[1:])
                if '[' in  name:
                    #any time where indexes are specified in the link, this is required to be
                    # an index specification, nothing to do with the variable name. 
                    name, _ = handle_brackets(name)                
                if name not  in plugins[step_name].plugin_output_names:
                    self.log.error(f'The output {name} was not declared in the plugin: {step_name.split(".")[0]}\n'
                                     f"Check your plugin's setup() step for I/O declaration")
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
        self.log.debug('check link types')
        bad_links = []
        for link in  self.study.solution.links:
            output_link = link[0]
            input_link = link[1]
            indexes_specified = False
            if '[' in  output_link:
                output_link, _ = handle_brackets(output_link)
                indexes_specified = True
            output_type = get_param_type(output_link, self.study, plugins)
            input_type = get_param_type(input_link,self.study, plugins)
            #Since only float arrays are allowed, always assumed that the elements of the arrays are floats
            if indexes_specified and output_type != np.ndarray:
                raise ValueError(f'Indexes may only be specified for arrays. The link: {link[1]} is not valid')
            if indexes_specified and  output_type == np.ndarray and  input_type == float:
                pass
            elif output_type != input_type:
                bad_links.append((output_link,output_type, input_link, input_type))

        if bad_links:
            msg = f'Link types must match, the following links have different types:\n' + "\n".join(f"{out_}:{out_type}->{in_}:{in_type}" for  out_,out_type, in_,in_type in  bad_links)
            self.log.error(msg)
            raise TypeError(msg)
        
                
    def check_link_indices(self, plugins):
        '''
        check source/destination for:
        valid indexing for source and destination
        '''
        for link in  self.study.solution.links:
            if link[0].split('.')[0] == 'vehicles':
                category = 'parameter'
            else:
                category = 'plugin_out'
            
            #checks exist to limit [] for links only as index specification
            if '[' in link[0]:
                _, inds = handle_brackets(link[0])
                # Handle the source side
                if category == 'parameter':
                    try:
                        a = self.study.get_value(link[0])
                        src_shape = a.shape
                    except:
                        raise ValueError(f'The indices in {link[0]} provided are not a valid means to index\n')
                else:
                    out_plugin_name, out_var_name = link[0].split('.')[0], handle_brackets('.'.join(link[0].split('.')[1:]))[0]
                    src_var_dict = copy.deepcopy(plugins[out_plugin_name].plugin_output_names[out_var_name])
                    a = src_var_dict['value']
                    a_shape = src_var_dict['shape']
                    if a is not None:
                        try:
                            a[inds]
                            src_shape = a[inds].shape
                        except:
                            raise ValueError('')
                    elif a_shape is  not None:
                        try:
                            np.ones(shape=a_shape)[inds]
                            src_shape = np.ones(shape=a_shape)[inds].shape
                        except:
                            raise ValueError('')
                    else:
                        #Note, this should already be caught for arrays in plugin declaration
                        raise ValueError('Developer: Check error catching capaibilities in plugin declaration')
                    
                # Handle the destination side (always a plugin), guard against memory access of actual dictionary
                in_plugin_name, in_var_name = link[1].split('.')[0], '.'.join(link[1].split('.')[1:])
                dest_var_dict = copy.deepcopy(plugins[in_plugin_name].plugin_input_names[in_var_name])
                b = dest_var_dict['value']
                b_shape = dest_var_dict['shape']
                b_type = dest_var_dict['type']
                
                src_size = 1 if src_shape == () else int(np.prod(src_shape))
                #float case
                if src_size ==1  and  b_type == float:
                        pass
                elif b is not None:
                    if src_shape != b.shape:
                        raise ValueError(f'The source indices must match the target shape. The input {link[0]} with indices {src_link_split[-1]}.\n'
                                         f'The size of the source indices is {src_shape} and the destination has shape {b.shape}')                    

                elif b_shape is  not None:
                    if src_shape != b_shape:
                        raise ValueError(f'The source indices must match the target shape. The input {link[0]} with indices {src_link_split[-1]}.\n'
                                         f'The size of the source indices is {src_shape} and the destination has shape {b_shape}')
                    
            

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
        for name,data in  state['des_vars'].items():
            des_var_type = get_param_type(name,self.study, plugins)
            if not check_continuous(des_var_type):
                self.log.error(f'The design variable: {name} is of type {des_var_type}. Only floats allowed as design variables')
                raise TypeError(f'The design variable: {name} is of type {des_var_type}\n'
                                f'Only floats allowed as design variables')
            self.check_upper_lower(des_var_type, data['lower'], data['upper'], 'Design Variable', name, plugins, self.study) 
            
            
    def check_constraints(self, plugins, state):
        for name,data in  state['constraints'].items():
            constraint_type = get_param_type(name,self.study, plugins)
            if not check_continuous(constraint_type):
                raise TypeError(f'The constraint: {name} is of type {constraint_type}\n'
                                f'Only floats allowed as constraints')
            self.check_upper_lower(constraint_type, data['lower'], data['upper'], 'Constraint', name, plugins, self.study) 
 
 
    @staticmethod
    def check_upper_lower(var_type,  data_lower, data_upper, dv_or_const, var_name, plugins, study):
        '''
        Check if desvars/constraints upper and lower bounds have been
        declared according to paradigm
        Allow:
        1.) Blank declarations for des vars(e.g only an upper limit, or no limits)
        2.) Np.ndarrays but declared as scalars (broadcasting)
        3.) Np.ndarrays if inputted as arrays must have same shape
        4.) No mixing (scalars and numpy arrays)
        '''
        
        # get the vehicle value or plugin shape/value
        base_name = var_name.split('.')[0]
        rest_name = '.'.join(var_name.split('.')[1:])
        if base_name == 'vehicles':
            var_value = study.get_value(var_name)
            var_shape = None
        else:
            step_plugin = plugins[base_name]
            if var_name in  step_plugin.plugin_output_names:
                var_shape = step_plugin.plugin_output_names[rest_name]['shape']
                var_value = step_plugin.plugin_output_names[rest_name]['value']
            elif var_name in  step_plugin.plugin_input_names:
                var_shape = step_plugin.plugin_input_names[rest_name]['shape']
                var_shape = step_plugin.plugin_input_names[rest_name]['value']
        # Checks on allowables for arrays        
        if var_type is  np.ndarray:
            # If the type has been declared as an ndarray, then
            # ndarrays and scalars (broadcasting) are okay
            # if both are None, this is fine (unbounded)
            
            if data_lower is not None and data_upper is  not None:
                #if the upper and lower are declared as scalars, this is fine
                
                if check_if_scalar(data_lower) and check_if_scalar(data_upper):
                    return
                # Mixed types already checked upon constraint/DV declaration
                if data_lower.shape != data_upper.shape:
                    raise ValueError(f'{dv_or_const} shapes must match\n'
                               f'The {dv_or_const} lower has shape: {data_lower.shape}\n'
                               f'The {dv_or_const} upper has shape: {data_upper.shape}')
                
            # Check declared constraint shapes match intended (via plugin or parameter)
            
            # For plugin inputs/outputs, shape vs. declared value (when applicable) already checked
            # First check here is if value is declared
            if data_lower is  not None and  not check_if_scalar(data_lower):
                #scalars allowed for broadcasting
                if not check_if_scalar(var_value):
                    
                    if data_lower.shape != var_value.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} lower has shape {data_lower.shape} but must have shape {var_value.shape}')
                # if var_value is not declared but shape is
                elif var_shape:
                    if var_shape != data_lower.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} lower has shape {data_lower.shape} but must have shape {var_value.shape}')
                    
            # For plugin inputs/outputs, shape vs. declared value (when applicable) already checked
            # First check here is if value is declared
            if data_upper is  not None and  not check_if_scalar(data_upper):
                #scalars allowed for broadcasting
                if not check_if_scalar(var_value):
                    if data_upper.shape != var_value.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} upper has shape {data_upper.shape} but must have shape {var_value.shape}')                # if var_value is not declared but shape is
                elif var_shape is  not None:
                    if var_shape != data_upper.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} upper has shape {data_upper.shape} but must have shape {var_value.shape}')                                    
        else:
            # now since check_continuous has been run, we know
            # values are already acceptable types in theory but
            # need to check for scalars being declared as arrays or some
            # other unwanted type
            if data_lower is not None and data_upper is  not None:
                if not (check_if_scalar(data_lower) or  check_if_scalar(data_upper)):
                    raise ValueError(f'{dv_or_const} was declared as {var_type} which is a scalar\n'
                                 f'but {dv_or_const} upper is {data_lower}\n'
                                 f'and {dv_or_const} lower is {data_upper}\n')
            
            
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
        self.log.debug('check objective')
        objective_type = get_param_type(state["objective_name"], self.study, plugins)
        is_continuous = check_continuous(objective_type)
        if not is_continuous:
            self.log.error(f'The objective: {state["objective_name"]} is of type {objective_type}. Only floats allowed as objectives')
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
        self.log.debug('Check for discrete I/O passing')
        # Check for discrete I/O passing
        for link in  self.study.solution.links:
            input_name = link[0]
            output_name = link[1]
            link_input_plugin = input_name.split('.')[0]
            link_output_plugin = output_name.split('.')[0]
            if link_input_plugin in  self.study.solution.steps and  link_output_plugin in  self.study.solution.steps:
                #not checking output since it should be the same as input
                link_input_var_name =  '.'.join(input_name.split('.')[1:])
                if '[' in  link_input_var_name:
                    link_input_var_name, _ = handle_brackets(link_input_var_name)
                input_type = plugins[link_input_plugin].plugin_output_names[link_input_var_name]['type']
                if not check_continuous(input_type):
                    self.log.error(f"Links between optimization components must be continuous (of type float). Check I/O declarations in plugin: {link_input_plugin}. Variable in violation: {input_name}")
                    raise TypeError(f"Links between optimization components must be continuous (of type float)\n"
                                    f"Check I/O declarations in plugin: {link_input_plugin}\n"
                                        f"Variable in violation: {input_name}")
