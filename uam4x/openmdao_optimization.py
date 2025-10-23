import openmdao.api as om
from uam4x.study import Study
import os
import shutil
import numpy as np
from uam4x.utils import check_continuous, get_param_type
from uam4x.log import get_log
from uam4x.utils import log

class Optimization:
    log = get_log(log, object_name='optimization')
    def __init__(self, study:Study, state, plugins, out_dir,check_partials=False) -> None:
        self.study = study
        self.state = state
        self.plugins = plugins
        self.out_dir = out_dir
        self.check_partials = check_partials
        self.log = Optimization.log

    def run_problem(self):
        problem = om.Problem()
        self.log.debug('run problem')

        #make vehicle step
        should_be_no_inputs, used_vehicle_param_names = self.study.get_plugin_input_and_output_names('vehicles')
        vehicle_val_dict = {}
        vehicle_type_dict = {}
        for param_name in used_vehicle_param_names:
            value = self.study._get_value_with_overwrites(param_name, self.state)
            vehicle_val_dict[param_name] = value
            vehicle_type_dict[param_name] = get_param_type(param_name, self.study)
        # comp = problem.model.add_subsystem('indep_var', OpenmdaoPreComp(self.log), promotes=['*'])
        comp = problem.model.add_subsystem('indep_var', OpenmdaoPreComp(), promotes=['*'])
        comp.options["vehicle_val_dict"] = vehicle_val_dict
        comp.options["vehicle_type_dict"] = vehicle_type_dict

        #make plugin steps
        for step_name in self.study.solution.steps:
            input_names = set([step_name + '.' + input_name for input_name in self.plugins[step_name].plugin_input_names.keys()])
            output_names = set([step_name + '.' + output_name for output_name in self.plugins[step_name].plugin_output_names.keys()])
            comp = problem.model.add_subsystem(step_name, UniversalOpenmdaoComp(), promotes=['*'])
            # comp = problem.model.add_subsystem(step_name, UniversalOpenmdaoComp(self.log), promotes=['*'])
            comp.options["plugin"] = self.plugins[step_name]
            comp.options["input_names"] = input_names
            comp.options["output_names"] = output_names
            work_dir = os.path.join(self.out_dir, step_name)
            # clean the work dir between runs
            comp.options["work_dir"] = work_dir
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
            comp.options["step_name"] = step_name

        # Add connections
        for independent, dependent in self.study.solution.links:
            problem.model.connect(independent.replace(".", "~"), dependent.replace(".", "~"))

        # add desvars
        for name, data in  self.state['des_vars'].items():
            dv_type = get_param_type(name, self.study)
            lower, upper = data['lower'], data['upper']
            indices = data['indices']
            
            # additional array wrapping here for floats, lists already wrapped
            if dv_type == np.ndarray:
                if lower is  not None:
                    lower = np.array(lower)
                if upper is  not None:
                    upper = np.array(upper)
            else:
                if lower is  not None:
                    lower = dv_type(lower)
                if upper is  not None:
                    upper = dv_type(upper)
            problem.model.add_design_var(name.replace(".", "~"), lower=lower, upper=upper, indices=indices)

        # add constraints
        for name, data in  self.state['constraints'].items():

            const_type = get_param_type(name, self.study, self.plugins)
            lower,upper = data['lower'], data['upper']
            indices = data['indices']
            
            # additional array wrapping here for floats, lists already wrapped            
            if const_type == np.ndarray:
                if lower is not None:
                    lower = np.array(lower)
                if upper is  not None:
                    upper = np.array(upper)
            else:
                if lower is  not None:
                    lower = const_type(lower)
                if upper is  not None:
                    upper = const_type(upper)            
            problem.model.add_constraint(name.replace(".", "~"), lower=lower, upper=upper, indices=indices)

        # add objectives
        problem.model.add_objective(self.state['objective_name'].replace(".", "~"))
        problem.driver = om.ScipyOptimizeDriver()
        problem.driver.options['optimizer'] = 'SLSQP'
        problem.setup()
        problem.run_driver()

        if not self.check_partials:
            for tracked_output in self.study.solution.tracked_outputs + list(self.state['des_vars'].keys()):
                self.study._add_overwrite(self.state, tracked_output, problem.get_val(tracked_output.replace(".", "~"))[0])

        return problem

class OpenmdaoPreComp(om.IndepVarComp):
    log = get_log(log, object_name='OpenmdaoPreComp')
    def initialize(self):
        self.log = get_log(log, object_name='OpenmdaoPreComp')
        self.log.debug('initialize')
        self.options.declare("vehicle_val_dict")
        self.options.declare("vehicle_type_dict")

    def setup(self):
        self.log.debug('setup')
        vehicle_val_dict = self.options["vehicle_val_dict"]
        vehicle_type_dict = self.options["vehicle_type_dict"]
        for param_name, value in vehicle_val_dict.items():
            param_type = vehicle_type_dict[param_name]
            if not check_continuous(param_type):
                self.add_discrete_output(param_name.replace(".", "~"), val=value)
            else:
                self.add_output(param_name.replace(".", "~"), val=value)

class UniversalOpenmdaoComp(om.ExplicitComponent):
    log = get_log(log, object_name='UniversalOpenmdaoComp')
    def initialize(self):
        self.log = UniversalOpenmdaoComp.log
        self.log.debug('initialize')
        self.options.declare("plugin")
        self.options.declare("input_names")
        self.options.declare("output_names")
        self.options.declare("work_dir")
        self.options.declare("step_name")

    def setup(self):
        self.log.debug('setup')
        self.plugin = self.options["plugin"]
        self.input_names = self.options["input_names"]
        self.output_names = self.options["output_names"]
        self.work_dir = self.options["work_dir"]
        self.step_name = self.options["step_name"]
        cont_input = False
        cont_output = False
        if self.plugin._check_partials_options:
            self.set_check_partial_options(wrt=['*'], form=self.plugin.check_form, step=self.plugin.check_step)

        for input_name in self.input_names:
            plugin_name = input_name.split('.')[0]
            name = '.'.join(input_name.split('.')[1:])
            input_type = self.plugin.plugin_input_names[name]['type']
            input_shape = self.plugin.plugin_input_names[name]['shape']
            if not check_continuous(input_type):
                if input_type is int:
                    temp = 1
                else:
                    temp = 'temp'
                self.add_discrete_input(input_name.replace(".", "~"), temp)
            else:
                cont_input = True
                self.add_input(input_name.replace(".", "~"), shape=input_shape)

        for output_name in self.output_names:
            plugin_name = output_name.split('.')[0]
            name = '.'.join(output_name.split('.')[1:])
            output_type = self.plugin.plugin_output_names[name]['type']
            output_shape = self.plugin.plugin_output_names[name]['shape']
            
            if not check_continuous(output_type):
                if output_type is int:
                    temp = 1
                else:
                    temp = 'temp'
                self.add_discrete_output(output_name.replace(".", "~"), temp)
            else:
                cont_output = True
                self.add_output(output_name.replace(".", "~"), shape=output_shape)

        if cont_input and cont_output:
            partials = self.plugin.build_partials(self.input_names, self.output_names)
            self.plugin.categorize_partials()
            for pair,data in  partials.items():
                of = self.step_name + '~' + pair[0].replace('.', '~')
                wrt = self.step_name + '~' + pair[1].replace('.', '~')
                method = data['method']
                step,form = data['step'], data['form']
                rows,cols = data['rows'], data['cols']
                self.declare_partials(of=of, wrt=wrt, method=method, step=step, form=form, rows=rows, cols=cols)


    def compute(self, inputs, outputs, discrete_inputs={}, discrete_outputs={}):
        self.log.debug('compute')
        self.plugin_dict = {}
        for name, value in inputs.items():
            plugin_name = name.replace("~", ".")
            plugin_name = ".".join(plugin_name.split(".")[1:])
            if not self.plugin.plugin_input_names[plugin_name]['type'] is np.ndarray:
                value = self.plugin.plugin_input_names[plugin_name]['type'](value)
            self.plugin_dict[plugin_name] = value
        for name, value in discrete_inputs.items():
            plugin_name = name.replace("~", ".")
            plugin_name = ".".join(plugin_name.split(".")[1:])
            self.plugin_dict[plugin_name] = value
        results = self.plugin.run_plugin(self.plugin_dict, self.work_dir)
        for name, value in results.items():
            name_with_plugin = self.step_name + "." + name
            if name_with_plugin in self.output_names:
                if not check_continuous(self.plugin.plugin_output_names[name]['type']):
                    discrete_outputs[name_with_plugin.replace(".", "~")] = value
                else:
                    outputs[name_with_plugin.replace(".", "~")] = value


    def compute_partials(self, inputs, partials, discrete_inputs={}):
        self.log.debug('compute partials')
        if not self.plugin._declare_partials:
            return
        else:
            results = self.plugin.compute_partials_plugin(self.plugin_dict)
            for name, value in  results.items():
                output_name = self.step_name + '.' + name[0]
                input_name = self.step_name + '.' + name[1]

                if output_name in self.output_names and  input_name in  self.input_names:
                    output_name_with_plugin = output_name.replace('.', '~')
                    input_name_with_plugin = input_name.replace('.', '~')
                    partials[output_name_with_plugin, input_name_with_plugin] = value
                else:
                    self.log.error('Missing links somewhere')
                    raise ValueError('Missing links somewhere')