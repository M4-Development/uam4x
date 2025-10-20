from typing import Callable
import os
import json
import importlib.util
from uam4x.utils import check_and_get_config
_plugin_registry = {}
from collections.abc import Sequence
from numbers import  Real
from itertools import  product
from uam4x.log import get_log
from uam4x.utils import log
import numpy as np

class Solution:
    log = get_log(log, object_name='solution')
    def __init__(self) -> None:
        self.tracked_outputs = []
        self.links = []
        self.steps = {}
        self.log = Solution.log
        self.log.debug(f'initialize')

    def add_link(self, independent:str, dependent:str):
        self.log.debug(f'add link. independent link: {independent}, dependent link: {dependent}')
        link = [independent, dependent]
        if link not in self.links:
            self.links.append(link)

    def remove_link(self, independent:str, dependent:str):
        self.log.debug(f'remove link. independent link: {independent}, dependent link: {dependent}')
        link = [independent, dependent]
        if link in self.links:
            self.links.remove(link)

    def get_step_names(self):
        self.log.debug(f'get step names')
        return [step_name for step_name in self.steps]

    def add_step(self, plugin_class, step_name=None, plugin_options:dict={}):
        self.log.debug(f'add step: {step_name}')
        if not isinstance(plugin_class, Callable):
            self.log.error("add_step: plugin_class should be a callable (class).")
            raise TypeError("plugin_class should be a callable (class).")
        # plugin_name should be a string, not a class
        if step_name is None:
            step_name = "step" + str(len(self.steps)+1)
        step_info = {"plugin": plugin_class, "plugin_options": plugin_options}
        self.steps[step_name] = step_info

    def remove_step(self, step_name):
        self.log.debug(f'remove step: {step_name}')
        if step_name in self.steps:
            self.steps.pop(step_name)

    def edit_step(self, step_name, plugin_options:dict=None):
        self.log.debug(f'edit step: {step_name}')
        if step_name in self.steps:
            step = self.steps[step_name]
            if plugin_options is not None:
                current_options = step["plugin_options"]
                for option_name, info in plugin_options.items():
                    current_options[option_name] = info
            else:
                plugin_options = {}

    def add_tracked_output(self, output:str):
        self.log.debug(f'add tracked output: {output}')
        if output in self.tracked_outputs:
            self.log.error(f"output already being tracked: {output}")
            raise RuntimeError(f"output already being tracked: {output}")
        self.tracked_outputs.append(output)

    def remove_tracked_output(self, output:str):
        self.log.debug(f'remove tracked output: {output}')
        self.tracked_outputs.remove(output)

    def add_plugin_option(self, step_name:str, option_name:str, option_value):
        self.log.debug(f'add plugin option {option_name} to step {step_name}')
        if step_name in self.steps:
            step = self.steps[step_name]
            step["plugin_options"][option_name] = option_value

    def remove_plugin_option(self, step_name:str, option_name:str):
        self.log.debug(f'remove plugin option {option_name} to step {step_name}')
        if step_name in self.steps:
            step = self.steps[step_name]
            step["plugin_options"].pop(option_name)

    def _to_dict(self) -> dict:
        # Serialize steps, converting plugin class to its name
        self.log.debug(f'write to dict')
        steps_serialized = {}
        for step_name, step_info in self.steps.items():
            plugin_class = step_info["plugin"]
            plugin_name = plugin_class.__name__
            steps_serialized[step_name] = {
                "plugin": plugin_name,
                "plugin_options": step_info["plugin_options"]
            }
        return {
            "steps": steps_serialized,
            "tracked_outputs": self.tracked_outputs,
            "links": self.links
        }

    @classmethod
    def _from_dict(cls, data: dict) -> 'Solution':
        sol = cls()
        cls.log.debug('solution from dict')
        # Reconstruct steps, mapping plugin name to class if possible
        for step_name, step_data in data["steps"].items():
            plugin_name = step_data["plugin"]
            plugin_class = get_plugin_class(plugin_name)
            sol.add_step(plugin_class, step_name, step_data["plugin_options"])
        sol.tracked_outputs = data["tracked_outputs"]
        sol.links = data["links"]
        return sol

class Plugin():
    log = get_log(log, object_name='Plugin base class')
    def __init__(self) -> None:
        self._declare_partials = False
        self._check_partials_options = False
        self.partials_counter = 0
        self.initial_partials = []
        self.comprehensive_partials_dict = {}
        self.method_to_partial = {}
        self.plugin_input_names = {}
        self.plugin_output_names = {}
        self.partial_2d_list = []
        self.log = Plugin.log

    def setup_plugin(self, plugin_options:dict):
        self.log.debug('setup plugin')

    def run_plugin(self, links:dict, work_dir:str):
        self.log.debug('run plugin')

    def declare_partials_plugin(self, of: str|Sequence[str], wrt: str|Sequence[str], method='fd', step=None, form=None,
                                rows: list|np.ndarray|None=None, cols: list|np.ndarray| None=None):
        '''
        User can declare partials as many times as they want here, and this is
        stored in a dictionary which will eventually be consolidated into a single
        partials dictionary.
        '''
        self.log.debug('declare partials plugin')
        # Check type of input, if string, wrap with list
        if type(of) == str:
            of = [of]
        # Check type of output, if string, wrap with list
        if type(wrt) == str:
            wrt = [wrt]

        # Error checking
        if method not in ['fd', 'exact']:
            self.log.error(f'Cannot use methods other than exact or fd, user used method: {method}')
            raise ValueError(f'Cannot use methods other than exact or fd, user used method: {method}')

        if method == 'fd':
            if step is  not None:
                if not isinstance(step, Real):
                    self.log.error(f'Step size must be a real number. The step size {step} is not valid')
                    raise ValueError(f'Step size must be a real number. The step size {step} is not valid')
                if step <= 0:
                    self.log.error(f'Step size must be greater than zero. The step size {step} is not valid')
                    raise ValueError(f'Step size must be greater than zero. The step size {step} is not valid')
            if form is not None:
                if form not in ['forward', 'backward', 'central']:
                    self.log.error('Only allowed finite difference forms are: forward,\n'
                                     f'backward, and central. The form: {form} is not valid')
                    raise ValueError('Only allowed finite difference forms are: forward,\n'
                                     f'backward, and central. The form: {form} is not valid')
        else:
            if step is  not None:
                self.log.error(f'User specified {method} method, which does not take in a step.\n'
                                 f'Instead, leave step blank or set it equal to None')
                raise ValueError(f'User specified {method} method, which does not take in a step.\n'
                                 f'Instead, leave step blank or set it equal to None')
            if form is  not None:
                self.log.error(f'User specified {method} method, which does not take in a form.\n'
                                 f'Instead, leave step blank or set it equal to None')
                raise ValueError(f'User specified {method} method, which does not take in a form.\n'
                                 f'Instead, leave step blank or set it equal to None')

        self.partials_counter += 1
        self.initial_partials.append({'of': of,'wrt': wrt,'method': method, 'step': step, 'form': form,
                                      'rows': rows,'cols': cols,})



    def build_partials(self, input_names, output_names):
        '''
        Builds a comprehensive dictionary of partials
        seperated by method (note only fd and exact allowed so far)
        '''
        self.log.debug('build partials')

        # Validate names
        # Collect declared names
        partial_ins = set()
        partial_outs = set()
        for data in  self.initial_partials:
            if data['wrt'] != ['*']:
                for inp in  data['wrt']:
                    partial_ins.add(inp)
            if data['of'] != ['*']:
                for out in  data['of']:
                    partial_outs.add(out)

        # Collect actual names
        local_output_names = set()
        local_input_names = set()
        for output_name in  output_names:
            output_name = '.'.join(output_name.split('.')[1:])
            local_output_names.add(output_name)

        for input_name in  input_names:
            input_name = '.'.join(input_name.split('.')[1:])
            local_input_names.add(input_name)

        #Note the empty set is a subset of all other sets
        if not partial_ins.issubset(local_input_names):
            self.log.error(f'Inputs and partials do not match:\n'
                             f'Inputs: {input_names}\n'
                             f'Partial inputs: {partial_ins}\n')
            raise ValueError(f'Inputs and partials do not match:\n'
                             f'Inputs: {input_names}\n'
                             f'Partial inputs: {partial_ins}\n')
        if not partial_outs.issubset(local_output_names):
            self.log.error(f'Outputs and partials do not match:\n'
                             f'Outputs: {output_names}\n'
                             f'Partial outputs: {partial_outs}\n')
            raise ValueError(f'Outputs and partials do not match:\n'
                             f'Outputs: {output_names}\n'
                             f'Partial outputs: {partial_outs}\n')

        #overwrites so declared partials order is last out
        self.comprehensive_partials_dict = {}
        for data in  self.initial_partials:
            if data['of'] == ['*']:
                ofs = local_output_names
            else:
                ofs = data['of']

            if data['wrt'] == ['*']:
                wrts = local_input_names
            else:
                wrts = data['wrt']

            for in_name, out_name in product(ofs, wrts):
                self.comprehensive_partials_dict[(in_name, out_name)] =  {'method': data['method'],
                                                                                     'form': data['form'],
                                                                                     'step':data['step'] ,
                                                                                     'rows': data['rows'],
                                                                                     'cols': data['cols'],}

        return self.comprehensive_partials_dict

    def categorize_partials(self):
        self.log.debug('catagorize partials')
        method_to_partial = {'fd': [],'exact': []}
        for pair, data in  self.comprehensive_partials_dict.items():
            if data['method'] == 'fd':
                method_to_partial['fd'].append(pair)
            elif data['method'] == 'exact':
                method_to_partial['exact'].append(pair)

        self.method_to_partial = method_to_partial

        if method_to_partial['exact']:
            self._declare_partials = True

        return self.method_to_partial


    def compute_partials_plugin(self, links: dict):
        self.log.debug('compute partials plugin')

    def set_check_partials_options(self, form='forward', step=None):
        self.log.debug('set check partials options')
        self._check_partials_options = True
        self.check_form = form
        self.check_step = step

    def declare_input(self, name, var_type, value=None, shape: tuple|None=None, shape_by_conn: bool=False, copy_shape: str|None=None):
        if '~' in  name:
            self.log.error('Tildes (~) are not allowed in variable names\n'
                             f'The following name is not allowed: {name}')
            raise ValueError('Tildes (~) are not allowed in variable names\n'
                             f'The following name is not allowed: {name}')
        self.plugin_input_names[name] = {'type': var_type,
                                         'value': value,
                                         'shape': shape,
                                         'shape_by_conn': shape_by_conn,
                                         'copy_shape': copy_shape}

    def declare_output(self, name, var_type, value=None, shape: tuple|None=None, shape_by_conn: bool=False, copy_shape: str|None=None):
        if '~' in  name:
            self.log.error('Tildes (~) are not allowed in variable names\n'
                             f'The following name is not allowed: {name}')
            raise ValueError('Tildes (~) are not allowed in variable names\n'
                             f'The following name is not allowed: {name}')
        self.plugin_output_names[name] = {'type': var_type,
                                          'value': value,
                                          'shape': shape,
                                          'shape_by_conn': shape_by_conn,
                                          'copy_shape': copy_shape}


def register_plugin(cls):
    plugin_name = cls.__name__
    log.debug(f'register plugin {plugin_name}')

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')
    _config = check_and_get_config(config_path)

    # Handle case where module is run as __main__ (no __spec__)
    try:
        module_origin = importlib.util.find_spec(cls.__module__).origin
    except (ValueError, AttributeError):
        # If module is run as __main__, get the actual file path from sys.argv[0]
        import sys
        if sys.argv and len(sys.argv) > 0:
            module_origin = os.path.abspath(sys.argv[0])
        else:
            # Fallback to current working directory + module name
            module_origin = os.path.join(os.getcwd(), cls.__module__ + '.py')

    if plugin_name in _config['PLUGINS'] and _config['PLUGINS'][plugin_name] != module_origin:
        log.warning(f"Plugin '{plugin_name}' already registered\nOverwritting existing data")

    _config['PLUGINS'][plugin_name] = module_origin
    with open(config_path, 'w') as configfile:
        json.dump(_config, configfile, indent=4)

    _plugin_registry[plugin_name] = cls
    return cls

def get_plugin_class(plugin_name):
    if not plugin_name in _plugin_registry:
        msg = f"Plugin '{plugin_name}' was not registered. Please check the following:\n"
        msg += f"\t1. Did you register the plugin using the @register_plugin?\n"
        msg += f"\t2. Were there Plugin Load Errors when you imported uam4x?\n"
        log.error(msg)
        raise ImportError(msg)
    return _plugin_registry[plugin_name]

def get_plugin_names():
    """Get a list of all registered plugin names."""
    log.debug('get plugin names')
    return list(_plugin_registry.keys())

