from uam4x.study import Study
import os
from uam4x.openmdao_optimization import Optimization
from uam4x.control import Optimization as UAM4XOptimization
import copy
from uam4x.log import get_log
from uam4x.custom_errors import PluginException
import inspect
import hashlib
import numpy as  np
from uam4x.utils import check_continuous, get_param_type, check_if_scalar,log, parse_index, handle_brackets
from collections import  defaultdict
from uam4x.results import Results, Result, Tradestudy  
from uam4x.log import FileLogger
import concurrent.futures
import time

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
    def __init__(self, study:Study, work_dir) -> None:
        self.study = study
        self.study_results = {} # change once there is a better way to store outputs
        self.sweep_combos = {}
        self.current_sweep_values = {}
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.log = get_log(log, object_name='engine')


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
        if isinstance(tradestudy,UAM4XOptimization):
            sample_state = tradestudy._to_dict()
            sample_state['optimization']=True
        else:
            sample_state = copy.deepcopy(tradestudy.get_states()[0])

        out_dir = os.path.join(self.work_dir, f"check_partials")
        os.makedirs(out_dir, exist_ok=True)

        plugins=self._generate_plugins()

            
        # May need to refactor this, but def want a check here. 
        self.check_model(plugins, sample_state)
        
        cwd = os.getcwd()
        os.chdir(out_dir)
        opt = Optimization(self.study, sample_state, plugins, out_dir, check_partials=True)
        problem = opt.run_problem()

        problem.check_partials(step=step, form=form, compact_print=True)
        os.chdir(cwd)


    def _generate_plugins(self):
        plugins = {}
        for step_name, step_info in self.study.solution.steps.items():
            self.log.debug(f'setup {step_name} plugin')
            plugin = step_info["plugin"]()
            plugin.setup_plugin(step_info["plugin_options"])
            plugins[step_name] = plugin
        return plugins

    def _run_state(self, state,iteration=None, check=True):
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

        os.makedirs(out_dir, exist_ok=True)

        plugins=self._generate_plugins()
            

        if state['optimization']:
            result=self._run_optimization(state, plugins, out_dir)
        else:
            result=self._run_single_point(state, plugins, out_dir)

        return result
    
    @staticmethod
    def _run_state_parallel(study_dict, state, iteration=None, work_dir=None):
        """
        Static method to run a solution in parallel.

        This method is designed to be used with concurrent execution
        frameworks. It reconstructs the Study object from a dictionary
        and runs the solution for the given state.

        Parameters
        ----------
        study_dict : dict
            Dictionary representation of the Study object.
        state : dict
            State dictionary containing all parameters and configuration.
        iteration : int, optional
            Iteration number for output directory naming, by default None.
        work_dir : str, optional
            Working directory for execution outputs, by default None.

        Returns
        -------
        Result
            The result object containing outputs from the solution.
        """
        if not iteration is None:
            out_dir = os.path.join(work_dir, f"out_{iteration}")
        else:
            out_dir = os.path.join(work_dir)

        os.makedirs(out_dir, exist_ok=True)


        study = Study._from_dict(study_dict, out_dir)
        engine = Engine(study, out_dir)    
        result = engine._run_state(state, iteration=iteration)
        return result


    def _run_optimization(self, state, plugins, out_dir):
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
        temp_state=copy.deepcopy(state)
        opt = Optimization(self.study, temp_state,plugins, out_dir)
        opt_result=opt.run_problem()

        os.chdir(cwd)

        return opt_result

    def _run_single_point(self, state, plugins, out_dir):
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
        temp_state=copy.deepcopy(state)
        for step_name, step_info in self.study.solution.steps.items():
            self.log.debug(f'run single point: {step_name}')
            plugin = plugins[step_name]
            work_dir = os.path.join(out_dir, step_name)
            os.makedirs(work_dir, exist_ok=True)
            
            plugin_inputs={}
            for (independent,dependent) in self.study.solution.links:  
                dependent_split = dependent.split(".")
                if dependent_split[0]==step_name:
                    var_name = ".".join(dependent_split[1:])
                    if independent in temp_state:
                        plugin_inputs[var_name] = temp_state[independent]
                    else:
                        plugin_inputs[var_name]=self.study.get_value(independent)

            results = plugin.run_plugin(plugin_inputs, work_dir)
            
            for var_name, result in results.items():
                var_name=step_name + "." + var_name
                if var_name in temp_state:
                    self.log.warning(f'Plugin {step_name} returned output {var_name} which overwrites an existing state variable.')
                temp_state[var_name] = result


        result=Result(state=state)

        for tracked_output in self.study.solution.tracked_outputs:
            if '[' in  tracked_output:
                no_ind_tracked_output, inds = handle_brackets(tracked_output)
                var_value = temp_state[no_ind_tracked_output][inds]
            else:
                var_value = temp_state[tracked_output]
            result._add_result(tracked_output, var_value)        

        return result   
                
    def run_solution(self):
        start_time = time.perf_counter()
        name='Default State'
        self.log.debug('run solution')
        self.capture_execution_data(name)
        self.sort_steps()
        plugins=self._generate_plugins()
        state={}
        out_dir=self.work_dir
        result=self._run_single_point(state, plugins, out_dir=out_dir)
        study_result=self.study.results
        if name in study_result.tradestudies:
            self.log.warning(f'Overwriting existing results for trade study {name}')

        ts=Tradestudy(name)
        ts._results.append(result)
        ts._hash_to_result[result.state_hash] = result
        
        study_result.add_tradestudy(name,ts)
        #add time to results metadata
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.study._runtime_meta_data[name]['Execution_Time'] = f'Analysis took {elapsed_time} seconds'


    def capture_execution_data(self,trade_name):
        # populate environment at execution time
        self.study._capture_environment(trade_name)
        # hash plugins at runtime
        plugin_name_to_hash = {}
        external_executable_to_versioning_data = {}
        for step_name, step_info in self.study.solution.steps.items():
            self.log.debug(f'setup {step_name} plugin')
            plugin_class = step_info["plugin"]
            #get the hash
            src = inspect.getsource(plugin_class)
            h = hashlib.sha256()
            h.update(src.encode("utf-8"))
            plug_hash = h.hexdigest()
            plugin_name_to_hash[step_name] = plug_hash
            if self.study.solution.steps[step_name]['plugin_hash'] != plug_hash:
                log.warning(f"Plugin '{plugin_class.__name__}' contains a different checksum from the one generated at branching script. Check plugin code for changes")

            plugin = step_info["plugin"]()
            plugin.setup_plugin(step_info["plugin_options"])
            if plugin._executable_to_data and not 'step_external_executable_data' in self.study._runtime_meta_data:
                #copying here because i don't want to carry and accidentally modify state
                external_executable_to_versioning_data[step_name] = copy.deepcopy(plugin._executable_to_data)

        if external_executable_to_versioning_data:
            self.study._add_runtime_external_executable_data(trade_name,external_executable_to_versioning_data)        
        
        self.study._add_runtime_plugin_checksums(trade_name,plugin_name_to_hash)

    def run_tradestudy(self, trade_name,num_processors=None, commit=False,check=True):
        """
        Run a trade study analysis.
        Executes a trade study, which can be either a parameter sweep
        or an optimization. Sorts solution steps, captures execution data,
        and manages parallel execution if specified.
        Parameters
        ----------      

        trade_name : str
            Name of the trade study to run.
        num_processors : int, optional
            Number of processors for parallel execution, by default None.
        commit : bool, optional 
            Whether to commit results to the study, by default False.
        check : bool, optional  
            Whether to perform model validation checks, by default True.
        """
        start_time = time.perf_counter()
        self.log.debug(f'run tradestudy: {trade_name}')
        self.sort_steps()
        tradestudy = self.study.control.trade_studies[trade_name]
        # addition for UQ workflows where orchestration is more complex
        if hasattr(tradestudy,'_check_analysis'):
            tradestudy._check_analysis()

        import uam4x
        os.environ['UAM4X_DISABLE_REGISTRY_WRITES'] = '1'
        # if num_processesors:
        #     if not self.are_plugins_threadsafe():
        #         print("WARNING: NON-THREADSAFE PLUGIN DETECTED, CREATING UNIQUE INSTANCE FOR EACH THREAD")
        #         plugins = None
        #     else:
        #         pass
        
        self.capture_execution_data(trade_name)
        if isinstance(tradestudy,UAM4XOptimization) and num_processors is not None:
            log.warning('num_processors is ignored for single point optimizations')
        if isinstance(tradestudy, UAM4XOptimization):
            #run optimization as defined by the base state
            state=tradestudy._to_dict()
            state['optimization']=True
            plugins=self._generate_plugins()
            
            if check:
                self.check_model(plugins, state)

            result=self._run_optimization(state, plugins, self.work_dir)
            study_result=self.study.results
            if trade_name in study_result.tradestudies:
                self.log.warning(f'Overwriting existing results for trade study {trade_name}')


            ts=Tradestudy(trade_name)
            ts._results.append(result)
            ts._hash_to_result[result.state_hash]=result
            study_result.add_tradestudy(trade_name,ts)

        else:
            # Note a parameter sweep can contain optimizations
            trade_states=tradestudy.get_states()
            self._run_parameter_sweep(trade_name,trade_states,num_processors=num_processors)

        #add time to results metadata
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.study._runtime_meta_data[trade_name]['Execution_Time'] = f'Analysis took {elapsed_time} seconds'

    
    def _run_parameter_sweep(self,trade_name,trade_states,num_processors=None):
        failed_states = []
        successful_states = []
        states= trade_states
        num_states = len(states)
        if num_processors and num_processors >1:
            if num_processors > num_states:
                log.warning(f'num_processors {num_processors} is greater than number of states {num_states}. Setting num_processors to {num_states}')
                num_processors = num_states
            if num_processors > os.cpu_count():
                log.warning(f'num_processors {num_processors} is greater than available CPU count {os.cpu_count()}. Setting num_processors to {os.cpu_count()}')
                num_processors = os.cpu_count()
            max_workers = num_processors 
            outputs=[]
            study_dict=self.study._to_dict()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures={i:executor.submit(Engine._run_state_parallel, study_dict, state, iteration=i,work_dir=self.work_dir)  for i, state in enumerate(states)}
                for j in sorted(futures.keys()):
                    try:
                        outputs.append(futures[j].result())
                        self.log.info(f'state successful: {states[j]}')
                        successful_states.append(states[j])
                    except PluginException as e:
                        tb = e.__traceback__
                        while tb.tb_next:
                            tb = tb.tb_next
                        lineno = tb.tb_lineno
                        filename = tb.tb_frame.f_code.co_filename
                        self.log.error(f"{type(e).__name__}: state failed - {states[j]}")
                        states[j]["Failure Message"]=f"{type(e).__name__}: An error occurred: {e}. Called in file {filename} in line {lineno}"
                        failed_states.append(states[j])
                        outputs.append(Result(state=states[j]))  

                    except Exception as e:
                        tb = e.__traceback__
                        while tb.tb_next:
                            tb = tb.tb_next
                        lineno = tb.tb_lineno
                        filename = tb.tb_frame.f_code.co_filename
                        self.log.error(f"An error occurred: {e}. Called in file {filename} in line {lineno}")
                        raise RuntimeError(f"An error occurred: {e}. Called in file {filename} in line {lineno}")
        elif num_processors ==1 or not num_processors:
            outputs=[]
            for i, state in enumerate(states):
#
                try:
                    outputs.append(self._run_state(state,  iteration=i))
                    self.log.info(f'state successful: {state}')
                    successful_states.append(state)
                except PluginException as e:
                    tb = e.__traceback__  # get traceback object from exception
                    while tb.tb_next:
                        tb = tb.tb_next  # go to the last traceback frame (where error originated)
                    lineno = tb.tb_lineno
                    filename = tb.tb_frame.f_code.co_filename
                    self.log.error(f"{type(e).__name__}: state failed - {state}")
                    state["Failure Message"]=f"{type(e).__name__}: An error occurred: {e}. Called in file {filename} in line {lineno}"
                    outputs.append(Result(state=state))  
                    failed_states.append(state)
                    
                except Exception as e:
                    tb = e.__traceback__  # get traceback object from exception
                    while tb.tb_next:
                        tb = tb.tb_next  # go to the last traceback frame (where error originated)
#
                    lineno = tb.tb_lineno
                    filename = tb.tb_frame.f_code.co_filename
                    self.log.error(f"An error occurred: {e}. Called in file {filename} in line {lineno}")
                    raise RuntimeError(f"An error occurred: {e}. Called in file {filename} in line {lineno}")
        else:
            raise ValueError('num_processors must be None or an integer >=1')
        #Consolidate results
        study_result=self.study.results
        if trade_name in study_result.tradestudies:
            self.log.warning(f'Overwriting existing results for trade study {trade_name}')

        ts=Tradestudy(trade_name)
        spec = self.study.control.trade_studies[trade_name]
        post_cls = getattr(spec, "post_spec", None)
        if post_cls is not None:
            ts.set_post(post_cls)
        for result in outputs:
            ts._results.append(result)
            ts._hash_to_result[result.state_hash] = result

        study_result.add_tradestudy(trade_name,ts)
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
        if state.get('optimization'):
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
            if indexes_specified and  output_type == np.ndarray and  check_continuous(input_type) and input_type != np.ndarray:
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
                        raise ValueError(f'The source indices must match the target shape. The input {link[0]} with indices {link[0].split(".")[-1]}.\n'
                                         f'The size of the source indices is {src_shape} and the destination has shape {b.shape}')                    

                elif b_shape is  not None:
                    if src_shape != b_shape:
                        raise ValueError(f'The source indices must match the target shape. The input {link[0]} with indices {link[0].split(".")[-1]}.\n'
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
            if rest_name in  step_plugin.plugin_output_names:
                var_shape = step_plugin.plugin_output_names[rest_name]['shape']
                var_value = step_plugin.plugin_output_names[rest_name]['value']
            #elif rest_name in  step_plugin.plugin_input_names:
                #var_shape = step_plugin.plugin_input_names[rest_name]['shape']
                #var_shape = step_plugin.plugin_input_names[rest_name]['value']
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
                if not check_if_scalar(var_value) and var_value is not None:
                    
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
                                         f'The {dv_or_const} {var_name} lower has shape {data_lower.shape} but must have shape {var_shape}')
                    
            # For plugin inputs/outputs, shape vs. declared value (when applicable) already checked
            # First check here is if value is declared
            if data_upper is  not None and  not check_if_scalar(data_upper):
                #scalars allowed for broadcasting
                if not check_if_scalar(var_value) and  var_value is  not None:
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
                                         f'The {dv_or_const} {var_name} upper has shape {data_upper.shape} but must have shape {var_shape}')                                    
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
                            f'Only continuous variables allowed as objectives')

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
