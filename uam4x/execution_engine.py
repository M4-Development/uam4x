from uam4x.study import Study
import os
from uam4x.openmdao_optimization import Optimization
import copy

from uam4x.utils import check_continuous, get_param_type

class Engine:

    def __init__(self, study:Study, work_dir, num_processes=1) -> None:
        self.study = study
        self.num_processes = num_processes
        self.study_results = {} # change once there is a better way to store outputs
        self.sweep_combos = {}
        self.current_sweep_values = {}
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def check_partials(self, trade_name, form='forward', step=1.e-6):

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
        cwd = os.getcwd()
        os.chdir(out_dir)
        opt = Optimization(self.study, state, plugins, out_dir)
        opt.run_problem()

        os.chdir(cwd)

    def run_single_point(self, state, plugins, out_dir):
        for step_name, step_info in self.study.solution.steps.items():
            plugin = plugins[step_name]
            work_dir = os.path.join(out_dir, step_name)
            os.makedirs(work_dir, exist_ok=True)

            plugin_inputs = self.study.get_plugin_inputs(step_name, state)
            results = plugin.run_plugin(plugin_inputs, work_dir)
            self.update_state_from_results(state, results, step_name)


    def update_state_from_results(self, state, results, step_name):
        for name, value in results.items():
            self.study._add_overwrite(state, step_name + "." + name, value)

    def run_tradestudy(self, trade_name, num_processesors=None, commit=False):
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
        self.check_links_to_plugins(plugins)
        self.check_link_types(plugins)
        if state['optimization']: #TODO can this not exist?
            self.check_desvars(plugins, state)
            self.check_constraints(plugins, state)
            self.check_objective(plugins, state)
            self.check_optimization_discrete(plugins, state)

    def check_links_to_plugins(self, plugins):
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
        objective_type = get_param_type(state["objective_name"], self.study, plugins)
        is_continuous = check_continuous(objective_type)
        if not is_continuous:
            raise TypeError(f'The objective: {state["objective_name"]} is of type {objective_type}\n'
                            f'Only floats allowed as objectives')

    def check_optimization_discrete(self, plugins, state):
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