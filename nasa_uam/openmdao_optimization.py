import openmdao.api as om
from nasa_uam.study import Study
import os
import shutil

class Optimization:
    def __init__(self, study:Study, state, plugins, out_dir) -> None:
        self.study = study
        self.state = state
        self.plugins = plugins
        self.out_dir = out_dir

    def run_problem(self):
        problem = om.Problem()

        #make vehicle step
        should_be_no_inputs, used_vehicle_param_names = self.study.get_plugin_input_and_output_names('vehicles')
        vehicle_val_dict = {}
        for param_name in used_vehicle_param_names:
            value = self.study._get_value_with_overwrites(param_name, self.state)
            vehicle_val_dict[param_name] = value
        comp = problem.model.add_subsystem('indep_var', OpenmdaoPreComp(), promotes=['*'])
        comp.options["vehicle_val_dict"] = vehicle_val_dict

        #make plugin steps
        for step_name in self.study.solution.steps:
            input_names, output_names = self.study.get_plugin_input_and_output_names(step_name)
            comp = problem.model.add_subsystem(step_name, UniversalOpenmdaoComp(), promotes=['*'])
            comp.options["plugin"] = self.plugins[step_name]
            comp.options["input_names"] = input_names
            comp.options["output_names"] = output_names
            comp.options["work_dir"] = os.path.join(self.out_dir, step_name)
            comp.options["step_name"] = step_name

        # Add connections
        for independent, dependent in self.study.solution.links:
            problem.model.connect(independent.replace(".", "~"), dependent.replace(".", "~"))


        # add desvars
        for name, lower, upper in zip(
            self.state['des_var_names'],
            self.state['des_var_lower'],
            self.state['des_var_upper'],
        ):
            problem.model.add_design_var(name.replace(".", "~"), float(lower), float(upper))

        # add constraints
        for name, lower, upper in zip(
            self.state['constraint_names'],
            self.state['constraint_lower'],
            self.state['constraint_upper'],
        ):
            problem.model.add_constraint(name.replace(".", "~"), float(lower), float(upper))

        # add objectives
        problem.model.add_objective(self.state['objective_name'].replace(".", "~"))
        problem.driver = om.ScipyOptimizeDriver()
        problem.driver.options['optimizer'] = 'SLSQP'
        problem.setup()
        problem.run_driver()
        for tracked_output in self.study.solution.tracked_outputs + self.state['des_var_names']:
            self.study._add_overwrite(self.state, tracked_output, problem.get_val(tracked_output.replace(".", "~"))[0])

class OpenmdaoPreComp(om.IndepVarComp):
    def initialize(self):
        self.options.declare("vehicle_val_dict")
    def setup(self):
        vehicle_val_dict = self.options["vehicle_val_dict"]
        for param_name, value in vehicle_val_dict.items():
            self.add_output(param_name.replace(".", "~"), val=value)

class UniversalOpenmdaoComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plugin")
        self.options.declare("input_names")
        self.options.declare("output_names")
        self.options.declare("work_dir")
        self.options.declare("step_name")
    def setup(self):
        self.plugin = self.options["plugin"]
        self.input_names = self.options["input_names"]
        self.output_names = self.options["output_names"]
        self.work_dir = self.options["work_dir"]
        self.step_name = self.options["step_name"]
        cont_input = False
        cont_output = False
        for input_name in self.input_names:
            if '.'.join(input_name.split('.')[1:]) in self.plugin.discrete_names:
                self.add_discrete_input(input_name.replace(".", "~"), val='temp')
            else:
                cont_input = True
                self.add_input(input_name.replace(".", "~"))
        for output_name in self.output_names:
            if '.'.join(output_name.split('.')[1:]) in self.plugin.discrete_names:
                self.add_discrete_output(output_name.replace(".", "~"), val='temp')
            else:
                cont_output = True
                self.add_output(output_name.replace(".", "~"))

        if cont_input and cont_output:
            self.declare_partials("*", "*", method='fd')

    def compute(self, inputs, outputs, discrete_inputs={}, discrete_outputs={}):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)
        plugin_dict = {}
        for name, value in inputs.items():
            plugin_name = name.replace("~", ".")
            pluing_name = ".".join(plugin_name.split(".")[1:])
            plugin_dict[pluing_name] = value[0]
        for name, value in discrete_inputs.items():
            plugin_name = name.replace("~", ".")
            pluing_name = ".".join(plugin_name.split(".")[1:])
            plugin_dict[pluing_name] = value
        results = self.plugin.run_plugin(plugin_dict, self.work_dir)
        for name, value in results.items():
            name_with_plugin = self.step_name + "." + name
            if name_with_plugin in self.output_names:
                if name in self.plugin.discrete_names:
                    discrete_outputs[name_with_plugin.replace(".", "~")] = value
                else:
                    outputs[name_with_plugin.replace(".", "~")] = value
