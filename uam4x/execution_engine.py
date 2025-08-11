from nasa_uam.study import Study
import os
import sys
import importlib
import json
import concurrent.futures
from nasa_uam.openmdao_optimization import Optimization

class Engine:

    def __init__(self, study:Study, num_processes=1) -> None:
        self.study = study
        self.num_processes = num_processes
        self.study_results = {} # change once there is a better way to store outputs
        self.sweep_combos = {}
        self.current_sweep_values = {}
        self.work_dir = os.path.join(os.path.dirname(self.study.json_path), "work")
        os.makedirs(self.work_dir, exist_ok=True)

    def are_plugins_threadsafe(self):
        for step_name, step_info in self.study.solution.steps.items():
            plugin = self.load_plugin(plugin_name=step_info["plugin_type"], step_name=step_name)
            plugin.setup_plugin(step_info["plugin_options"])
            if not plugin.is_threadsafe:
                return False
        return True

    def init_plugins(self):
        plugins = {}
        thread_safe = False
        for step_name, step_info in self.study.solution.steps.items():
            plugins[step_name] = self.load_plugin(plugin_name=step_info["plugin_type"], step_name=step_name)
            plugins[step_name].setup_plugin(step_info["plugin_options"])
        return plugins

    def run_solution(self, state, plugins, iteration=None):
        if not iteration is None:
            out_dir = os.path.join(self.work_dir, f"out_{iteration}")
        else:
            out_dir = os.path.join(self.work_dir)

        if not plugins:
            plugins = {}
            for step_name, step_info in self.study.solution.steps.items():
                plugin = self.load_plugin(plugin_name=step_info["plugin_type"], step_name=step_name)
                plugin.setup_plugin(step_info["plugin_options"])
                plugins[step_name] = plugin

        os.makedirs(out_dir, exist_ok=True)
        if state['optimization']:
            self.run_optimization(state, plugins, out_dir)
        else:
            self.run_single_point(state, plugins, out_dir)

    def run_optimization(self, state, plugins, out_dir):
        opt = Optimization(self.study, state, plugins, out_dir)
        opt.run_problem()

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

        plugins = self.init_plugins()

        if num_processesors:
            if not self.are_plugins_threadsafe():
                print("WARNING: NON-THREADSAFE PLUGIN DETECTED, CREATING UNIQUE INSTANCE FOR EACH THREAD")
                plugins = None
            else:
                pass

        for i, state in enumerate(tradestudy.trade.get_states()):
            self.run_solution(state, plugins, i)
        self.study.save()

    def load_plugin(self, plugin_name, step_name):
        python_files = []
        working_dir = os.getcwd()
        plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        if plugins_dir not in sys.path:
            sys.path.append(plugins_dir)
        plugins_dir_relative_path = os.path.relpath(plugins_dir, working_dir)
        script_path = os.path.abspath(sys.argv[0])
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        script_dir_relative_path = os.path.relpath(script_dir, working_dir)
        directories = [plugins_dir_relative_path, script_dir_relative_path]
        for dir in directories:
            python_files = []
            for f in os.listdir(dir):
                if f.endswith(".py") and f != "__init__.py":
                    python_files.append(f)
            for f in python_files:
                file_name = f[:-3]
                imported = importlib.import_module(file_name)
                if hasattr(imported, plugin_name):
                    plugin = getattr(imported, plugin_name)()
                    return plugin

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