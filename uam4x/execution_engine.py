from uam4x.study import Study
import os
from uam4x.openmdao_optimization import Optimization
from uam4x.plugin import get_new_plugin_instance

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
            plugin = get_new_plugin_instance(step_info["plugin_type"])
            plugin.setup_plugin(step_info["plugin_options"])
            if not plugin.is_threadsafe:
                return False
        return True

    def init_plugins(self):
        plugins = {}
        thread_safe = False
        for step_name, step_info in self.study.solution.steps.items():
            plugins[step_name] = get_new_plugin_instance(step_info["plugin_type"])
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
                plugin = self.get_new_plugin_instance(step_info["plugin_type"])
                plugin.setup_plugin(step_info["plugin_options"])
                plugins[step_name] = plugin

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