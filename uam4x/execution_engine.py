import os
from uam4x.study import Study
from uam4x.openmdao_optimization import Optimization
from uam4x.control import Optimization as UAM4XOptimization
import copy
from uam4x.log import get_log
from uam4x.custom_errors import PluginException
import inspect
import hashlib
import numpy as np
from uam4x.utils import is_continuous_type_or_value, is_scalar_type_or_value, log, split_dotted_path_and_indices
from uam4x.results import Result, Tradestudy
import concurrent.futures
import time
from datetime import datetime
from zoneinfo import ZoneInfo
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
        """Initialize object state and default configuration.

        Parameters
        ----------
        study : Study
            Input value for `study`.
        work_dir : Any
            Input value for `work_dir`.

        """
        self.study = study
        self.study_results = {} # change once there is a better way to store outputs
        self.sweep_combos = {}
        self.current_sweep_values = {}
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.log = get_log(log, object_name='engine')
        self.create_step_work_dirs=True


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
        self.study.solution.sort_steps()
        tradestudy = self.study.control.trade_studies[trade_name]

        # Sample state, doesn't really matter here, special check
        # partials directory created
        if isinstance(tradestudy,UAM4XOptimization):
            sample_state = tradestudy._to_dict()
            sample_state['optimization'] = True
        elif hasattr(tradestudy, 'get_optim_child') and tradestudy.get_optim_child() is not None:
            # this will have optimization=True already
            sample_state = copy.deepcopy(tradestudy.get_states()[0])
        else:
            print('Checking Partials Only Available For Optimization Based Analysis')
            return

        out_dir = os.path.join(self.work_dir, f"check_partials")
        os.makedirs(out_dir, exist_ok=True)

        plugins=self.study.solution.generate_plugins()

            
        # May need to refactor this, but def want a check here. 
        self.check_model(plugins, sample_state)
        
        cwd = os.getcwd()
        os.chdir(out_dir)
        opt = Optimization(self.study, sample_state, plugins, out_dir, check_partials=True)
        problem = opt.run_problem()

        problem.check_partials(step=step, form=form, compact_print=True)
        os.chdir(cwd)

    def _run_state(self, state, iteration=None,resort_steps=False):
        """
        Run a solution for a given state.

        Executes either an optimization or single-point analysis based on the
        state configuration.

        Parameters
        ----------
        state : dict
            State dictionary containing all parameters and configuration.
        iteration : int, optional
            Iteration number for output directory naming, by default None.
        resort_steps : bool, optional
            If True, regenerate the solution execution order before running.

        Notes
        -----
        If iteration is provided, outputs will be saved to 'out_{iteration}'
        subdirectory. Otherwise, outputs go to the main work directory.
        """
        self.log.debug('run solution')
        # Parallel workers rebuild Study from a serialized form, which does not
        # preserve the derived execution order stored in `_execution_steps`.
        # Recompute it here so serial and parallel runs build the same model.
        if not iteration is None and self.create_step_work_dirs:
            out_dir = os.path.join(self.work_dir, f"out_{iteration}")
        else:
            out_dir = os.path.join(self.work_dir)

        os.makedirs(out_dir, exist_ok=True)

        plugins=self.study.solution.generate_plugins()
        
        #since parallelism rebuilds the study, need to re sort
        if resort_steps:
            self.study.solution.sort_steps()
        
        if state.get('optimization',False):
            result=self._run_optimization(state, plugins, out_dir)
        else:
            result=self._run_single_point(state, plugins, out_dir)

        return result
    
    @staticmethod
    def _run_state_parallel(study_dict, state, iteration=None, work_dir=None,create_step_work_dirs=True):
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

        study = Study._from_dict(study_dict, work_dir)
        engine = Engine(study, work_dir)    
        engine.create_step_work_dirs=create_step_work_dirs
        result = engine._run_state(state, iteration=iteration,resort_steps=True)
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
        temp_state=copy.deepcopy(state)
        cwd = os.getcwd()
        os.chdir(out_dir)
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
        ordered_plugins=self.study.solution.get_global_execution_order_plugins(plugins)
        full_links=self.study.solution._build_global_links()
        for step_name,plugin in ordered_plugins.items():
            step_name_ind=len(step_name.split('.'))
            self.log.debug(f'run single point: {step_name}')
            if self.create_step_work_dirs:
                work_dir = os.path.join(out_dir, step_name)
                os.makedirs(work_dir, exist_ok=True)
            else:
                work_dir=out_dir
            plugin_inputs={}
            for independent,dependent in full_links:
                dependent_split=dependent.split('.')
                if len(dependent_split)<=step_name_ind:
                    continue
                else:
                    if '.'.join(dependent_split[:step_name_ind])==step_name:
                        var_name='.'.join(dependent_split[step_name_ind:])
                        if independent in temp_state:
                            plugin_inputs[var_name]=temp_state[independent]
                        else:
                            plugin_inputs[var_name]= self.study.get_value(independent)
            results=plugin.run_plugin(plugin_inputs,work_dir)

            for var_name,result in results.items():
                var_name=step_name+'.'+var_name
                if var_name in temp_state:
                    self.log.warning(f'Plugin {step_name} returned output {var_name} which overwrites an existing state variable.')
                temp_state[var_name] = result
            


        result=Result(state=state)

        for tracked_output in self.study.solution.tracked_outputs:
            temp_output_name='solution.'+tracked_output
            if '[' in  tracked_output:
                no_ind_tracked_output, inds = split_dotted_path_and_indices(tracked_output)
                var_value = temp_state[no_ind_tracked_output][inds]
            else:
                var_value = temp_state[temp_output_name]
            result._add_result(tracked_output, var_value)        

        return result   
                


    def run_solution(self):
        """Run solution.

        Notes
        -----
        This method performs internal setup or validation logic.

        """
        start_time = time.perf_counter()
        trade_name='Default State'
        self.log.debug('run solution')
        self.capture_execution_data(trade_name)
        self.study.solution.sort_steps()
        plugins=self.study.solution.generate_plugins()
        state={}
        out_dir=self.work_dir
        result=self._run_single_point(state, plugins, out_dir=out_dir)
        study_result=self.study.results
        if trade_name in study_result.tradestudies:
            self.log.warning(f'Overwriting existing results for trade study {trade_name}')
        ts=Tradestudy(trade_name)
        ts._results.append(result)
        ts._hash_to_result[result.state_hash] = result
        
        study_result.add_tradestudy(trade_name,ts)
        #add time to results metadata
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.add_end_time_meta_data(trade_name,elapsed_time)

        
    def add_end_time_meta_data(self,trade_name,elapsed_time):
        """Add end time meta data.

        Parameters
        ----------
        trade_name : Any
            Identifier used by this operation.
        elapsed_time : Any
            Input value for `elapsed_time`.

        """
        self.study._runtime_meta_data[trade_name]['time_info']['Elapsed_Time'] = f'Analysis took {elapsed_time} seconds'
        
        dt=datetime.now(tz=ZoneInfo("America/Los_Angeles"))

        message=f'Completed on'
        keywords=['A','B','d','Y','I','M','S','p']
        for keyword in keywords:
            if not keyword in ['M','S']:
                message+=f' {dt.strftime(f"%{keyword}")}'
            else:
                message+=f':{dt.strftime(f"%{keyword}")}'
            if keyword in ['A','d']:
                message+=','
            if keyword=='Y':
                message+=' at'
            
        message+=' Pacific Time'
        self.study._runtime_meta_data[trade_name]['time_info']['Completion_Time'] = message

    def capture_execution_data(self,trade_name):
        # populate environment at execution time
        """Execute capture execution data.

        Parameters
        ----------
        trade_name : Any
            Identifier used by this operation.

        """
        self.study._capture_environment(trade_name)
        # hash plugins at runtime
        plugin_name_to_hash = {}
        external_executable_to_versioning_data = {}
        for plugin_name, data in self.study.solution.get_all_plugin_names_to_data().items():
            #get the hash
            plugin_class=data['plugin']
            reference_plugin_hash=data['plugin_hash']
            plugin_options=data['plugin_options']
            plugin=plugin_class()
            src = inspect.getsource(plugin_class)
            h = hashlib.sha256()
            h.update(src.encode("utf-8"))
            plug_hash = h.hexdigest()
            plugin_name_to_hash[plugin_name] = plug_hash

            #NEED TO UPDATE TO REFLECT NEW STRUCTURE FOR COMPARISON
            if reference_plugin_hash != plug_hash:
                log.warning(f"Plugin '{plugin_class.__name__}' contains a different checksum from the one generated at branching script. Check plugin code for changes")
#
            #NEED TO UPDATE THIS TO ALIGN WITH GROUPS, MAYBE NEED A MORE CENTRALIZED
            #SET UP TO MINIMIZE REPETITION 
            plugin = plugin_class()
            plugin.setup_plugin(plugin_options)
            if plugin._executable_to_data and not 'step_external_executable_data' in self.study._runtime_meta_data:
                #copying here because i don't want to carry and accidentally modify state
                external_executable_to_versioning_data[plugin_name] = copy.deepcopy(plugin._executable_to_data)

        if external_executable_to_versioning_data:
            self.study._add_runtime_external_executable_data(trade_name,external_executable_to_versioning_data)        
        
        self.study._add_runtime_plugin_checksums(trade_name,plugin_name_to_hash)

    def run_tradestudy(self, trade_name,num_processors=None,check=True,create_step_work_dirs=True):
        """
        Run an analysis.
        Executes a trade study, which can be either a parameter sweep
        or an optimization. Sorts solution steps, captures execution data,
        and manages parallel execution if specified.
        Parameters
        ----------      

        trade_name : str
            Name of the trade study to run.
        num_processors : int, optional
            Number of processors for parallel execution, by default None.
        check : bool, optional  
            Whether to perform model validation checks, by default True.
        """
        self.create_step_work_dirs=create_step_work_dirs
        start_time = time.perf_counter()
        self.log.debug(f'run tradestudy: {trade_name}')
        self.study.solution.sort_steps()
        tradestudy = self.study.control.trade_studies[trade_name]
        # addition for UQ workflows where orchestration is more complex
        if hasattr(tradestudy,'_check_analysis'):
            tradestudy._check_analysis()

        import uam4x
        os.environ['UAM4X_DISABLE_REGISTRY_WRITES'] = '1'
        
        self.capture_execution_data(trade_name)
        if isinstance(tradestudy,UAM4XOptimization) and num_processors is not None:
            log.warning('num_processors is ignored for single point optimizations')
        if isinstance(tradestudy, UAM4XOptimization):
            #run optimization as defined by the base state
            state=tradestudy._to_dict()
            state['optimization']=True
            plugins=self.study.solution.generate_plugins()
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
            if check:
                plugins=self.study.solution.generate_plugins()
                self.check_model(plugins,trade_states[0])
            self._run_parameter_sweep(trade_name,trade_states,num_processors=num_processors)

        #add time to results metadata
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.add_end_time_meta_data(trade_name,elapsed_time)

    
    def _run_parameter_sweep(self,trade_name,trade_states,num_processors=None):
        """Run parameter sweep.

        Parameters
        ----------
        trade_name : Any
            Identifier used by this operation.
        trade_states : Any
            Input value for `trade_states`.
        num_processors : Any, optional
            Input value for `num_processors`.

        """
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
                futures={i:executor.submit(Engine._run_state_parallel, study_dict, state, iteration=i,work_dir=self.work_dir,
                                           create_step_work_dirs=self.create_step_work_dirs)  for i, state in enumerate(states)}
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
                    outputs.append(self._run_state(state, iteration=i))
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

    def check_model(self, plugins, state=None):
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
        if state.get('optimization',False):
            self.check_desvars(plugins, state)
            self.check_constraints(plugins, state)
            self.check_objective(plugins, state)
            self.check_optimization_discrete(plugins)

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
        self.study.solution._check_links_to_plugins(plugins)

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
        self.study.solution._check_link_types(plugins,self.study)
                
    def check_link_indices(self, plugins):
        """Validate link indices.

        Parameters
        ----------
        plugins : Any
            Input value for `plugins`.

        """
        self.study.solution._check_link_indices(plugins,self.study)

    def _get_analysis_var_data(self, var_name, plugins):
        """Return analysis var data.

        Parameters
        ----------
        var_name : Any
            Identifier used by this operation.
        plugins : Any
            Input value for `plugins`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        if var_name.startswith('vehicles.'):
            return {
                'type': self.study.get_param_type(var_name),
                'value': self.study.get_value(var_name),
                'shape': None,
            }
        return self.study.solution.get_var_data(var_name, plugins)
                    
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
            var_data = self._get_analysis_var_data(name, plugins)
            des_var_type = var_data['type']
            if not is_continuous_type_or_value(des_var_type):
                self.log.error(f'The design variable: {name} is of type {des_var_type}. Only floats allowed as design variables')
                raise TypeError(f'The design variable: {name} is of type {des_var_type}\n'
                                f'Only floats allowed as design variables')
            self.check_upper_lower(des_var_type, data['lower'], data['upper'], 'Design Variable', name, var_data)
            
            
    def check_constraints(self, plugins, state):
        """Validate constraints.

        Parameters
        ----------
        plugins : Any
            Input value for `plugins`.
        state : Any
            Input value for `state`.

        """
        for name,data in  state['constraints'].items():
            var_data = self._get_analysis_var_data(name, plugins)
            constraint_type = var_data['type']
            if not is_continuous_type_or_value(constraint_type):
                raise TypeError(f'The constraint: {name} is of type {constraint_type}\n'
                                f'Only floats allowed as constraints')
            self.check_upper_lower(constraint_type, data['lower'], data['upper'], 'Constraint', name, var_data)
 

    def check_upper_lower(self, var_type, data_lower, data_upper, dv_or_const, var_name, var_data):
        """Validate upper lower.

        Parameters
        ----------
        var_type : Any
            Input value for `var_type`.
        data_lower : Any
            Input value for `data_lower`.
        data_upper : Any
            Input value for `data_upper`.
        dv_or_const : Any
            Input value for `dv_or_const`.
        var_name : Any
            Identifier used by this operation.
        var_data : Any
            Input value for `var_data`.

        """
        var_value = var_data['value']
        var_shape = var_data['shape']
        # Checks on allowables for arrays        
        if var_type is  np.ndarray:
            # If the type has been declared as an ndarray, then
            # ndarrays and scalars (broadcasting) are okay
            # if both are None, this is fine (unbounded)
            
            if data_lower is not None and data_upper is  not None:
                #if the upper and lower are declared as scalars, this is fine
                
                if is_scalar_type_or_value(data_lower) and is_scalar_type_or_value(data_upper):
                    return
                # Mixed types already checked upon constraint/DV declaration
                if data_lower.shape != data_upper.shape:
                    raise ValueError(f'{dv_or_const} shapes must match\n'
                               f'The {dv_or_const} lower has shape: {data_lower.shape}\n'
                               f'The {dv_or_const} upper has shape: {data_upper.shape}')
                
            # Check declared constraint shapes match intended (via plugin or parameter)
            
            # For plugin inputs/outputs, shape vs. declared value (when applicable) already checked
            # First check here is if value is declared
            if data_lower is not None and  not is_scalar_type_or_value(data_lower):
                #scalars allowed for broadcasting
                if not is_scalar_type_or_value(var_value) and var_value is not None:
                    
                    if data_lower.shape != var_value.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} lower has shape {data_lower.shape} but must have shape {var_value.shape}')
                # if var_value is not declared but shape is
                elif var_shape is not None:
                    if var_shape != data_lower.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} lower has shape {data_lower.shape} but must have shape {var_shape}')
                    
            # For plugin inputs/outputs, shape vs. declared value (when applicable) already checked
            # First check here is if value is declared
            if data_upper is not None and  not is_scalar_type_or_value(data_upper):
                #scalars allowed for broadcasting
                if not is_scalar_type_or_value(var_value) and  var_value is  not None:
                    if data_upper.shape != var_value.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} upper has shape {data_upper.shape} but must have shape {var_value.shape}')                # if var_value is not declared but shape is
                elif var_shape is not None:
                    if var_shape != data_upper.shape:
                        raise ValueError(f'If arrays are declared as bounds for {dv_or_const}\n'
                                         f'Then either a scalar (broadcasting) or an array matching the shape of\n'
                                         f'the value/declared shape in the vehicle component parameter or \n'
                                         f'plugin input output is accepted.\n'
                                         f'The {dv_or_const} {var_name} upper has shape {data_upper.shape} but must have shape {var_shape}')                                    
        else:
            # now since is_continuous_type_or_value has been run, we know
            # values are already acceptable types in theory but
            # need to check for scalars being declared as arrays or some
            # other unwanted type
            if data_lower is not None and data_upper is  not None:
                if not (is_scalar_type_or_value(data_lower) or  is_scalar_type_or_value(data_upper)):
                    raise ValueError(f'{dv_or_const} was declared as {var_type} which is a scalar\n'
                                 f'but {dv_or_const} upper is {data_upper}\n'
                                 f'and {dv_or_const} lower is {data_lower}\n')
            
            
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
        objective_data = self._get_analysis_var_data(state["objective_name"], plugins)
        objective_type = objective_data['type']
        is_continuous = is_continuous_type_or_value(objective_type)
        if not is_continuous:
            self.log.error(f'The objective: {state["objective_name"]} is of type {objective_type}. Only floats allowed as objectives')
            raise TypeError(f'The objective: {state["objective_name"]} is of type {objective_type}\n'
                            f'Only continuous variables allowed as objectives')

    def check_optimization_discrete(self, plugins):
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
        self.study.solution._check_optimization_discrete(plugins)
#
