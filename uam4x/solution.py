from __future__ import annotations
from typing import Callable
import os
import json
import importlib.util
from uam4x.utils import check_and_get_config, check_numpy_shape, check_if_scalar, is_numpy_array, normalize_shape
_plugin_registry = {}
from collections.abc import Sequence
from numbers import  Real
from itertools import  product
from uam4x.log import get_log
from uam4x.utils import log
import numpy as np
import inspect
import hashlib

class Solution:
    log = get_log(log, object_name='solution')
    def __init__(self) -> None:
        self.tracked_outputs = []
        self._template_file_paths = []
        self.links = []
        self.steps = {}
        self.log = Solution.log
        self.log.debug(f'initialize')

    def add_link(self, independent:str, dependent:str):
        self.log.debug(f'add link. independent link: {independent}, dependent link: {dependent}')
        
        # need slicing here dependent cannot 
        if  '[' in dependent:
            raise ValueError(rf'Only independent variables can have indices specified. The independent variable {dependent}\n'
                             rf'has slices indices specified.')
        
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
    
    def _make_step_info(self, plugin_class, plugin_options, plugin_hash=None):
        if plugin_hash is None:
            src = inspect.getsource(plugin_class)
            h = hashlib.sha256()
            h.update(src.encode("utf-8"))
            plugin_hash = h.hexdigest()

        return {
            "plugin": plugin_class,
            "plugin_options": plugin_options,
            "plugin_hash": plugin_hash,
        }

    def add_step(self, plugin_class, step_name=None, plugin_options:dict={}):
        self.log.debug(f'add step: {step_name}')
        if not isinstance(plugin_class, Callable):
            self.log.error("add_step: plugin_class should be a callable (class).")
            raise TypeError("plugin_class should be a callable (class).")
        if step_name is  not None and  ('[' in  step_name or  ']' in  step_name):
            self.log.error(f'steps cannot have the character "[" or "]" in them')            
            raise ValueError(f'steps cannot have the character "[" or "]" in them')         
        # plugin_name should be a string, not a class
        if step_name is None:
            step_name = "step" + str(len(self.steps)+1)
        
        step_info = self._make_step_info(plugin_class, plugin_options)
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
                step['plugin_options'] = {}

    def add_tracked_output(self, output:str):
        self.log.debug(f'add tracked output: {output}')
        if output in self.tracked_outputs:
            self.log.error(f"output already being tracked: {output}")
            raise RuntimeError(f"output already being tracked: {output}")
        self.tracked_outputs.append(output)

    def remove_tracked_output(self, output:str):
        self.log.debug(f'remove tracked output: {output}')
        if output in  self.tracked_outputs:
            self.tracked_outputs.remove(output)

    def add_plugin_option(self, step_name:str, option_name:str, option_value):
        self.log.debug(f'add plugin option {option_name} to step {step_name}')
        if step_name in self.steps:
            step = self.steps[step_name]
            if option_name in  step['plugin_options']:
                print(f'{option_name} already a plugin option. Overwriting value: {step["plugin_options"][option_name]}')
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
                "plugin_options": step_info["plugin_options"],
                "plugin_hash": step_info["plugin_hash"],
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
            plugin_options = step_data["plugin_options"]
            plugin_hash = step_data["plugin_hash"]
    
            step_info = sol._make_step_info(plugin_class, plugin_options, plugin_hash)
            sol.steps[step_name] = step_info            
            
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
        self._plugin_templates = []
        self.log = Plugin.log
        self._template_to_checksum = {}
        self._executable_to_data = {}
    
    def __str__(self):
        return f'{self.__class__.__name__}'

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

    @staticmethod
    def check_declared_IO(name, var_type, value, shape):
        if not isinstance(var_type, type):
            raise TypeError(f'Declared type must be a type. {name} has declared type {var_type} which is not a type.')

        
        if '~' in  name or  '[' in  name or  ']' in  name:
            #self.log.error(r'Tildes (~), square brackets "[","]" are not allowed in variable names\n'
                             #rf'The following name is not allowed: {name}')
            raise ValueError(f'Tildes (~), square brackets "[","]" are not allowed in variable names\n'
                             f'The following name is not allowed: {name}')
        
        if shape is  not None:
            
            if var_type is not np.ndarray:
                raise ValueError(f'{name} has declared type: {var_type} but a shape argument was provided\n'
                                 f'only np.ndarray types are allowed to have shapes')
            else:
                if not check_numpy_shape(shape):
                    raise ValueError(f'The shape must be declared as a tuple of integers or as a\n'
                                 f'single integer or as NoneType. The shape of {name} is {shape} which is invalid')
            
                shape = normalize_shape(shape)
                if value is  not None:
                    if check_if_scalar(value):
                        pass
                    elif value.shape != shape:
                        raise ValueError(f'If the value of an array is not declared as a scalar, then the shape must match\n'
                                         f'the declared shape. The variable {name} has declared shape {shape} and actual shape {value.shape}')
        else:
            
            if value is  None and  var_type is np.ndarray:
                raise ValueError(f'Arrays must have either the shape or value defined (or both). The variable\n'
                                 f'{name} has no shape or value.')
            if value is not None and check_if_scalar(value) and var_type is  np.ndarray:
                raise ValueError(f'If array value is declared with scalar, shape must be declared for broadcasting.\n'
                                 f'The variable {name} has value: {value} and type: {var_type} but no shape declared.')
        
        return shape
        
                    
                    
    def declare_input(self, name, var_type, value=None, shape: int|tuple|None=None):
        shape = self.check_declared_IO(name, var_type, value, shape)
        self.plugin_input_names[name] = {'type': var_type,
                                         'value': value,
                                         'shape': shape}

    def declare_output(self, name, var_type, value=None, shape: int|tuple|None=None):
        shape = self.check_declared_IO(name, var_type, value, shape)
        self.plugin_output_names[name] = {'type': var_type,
                                          'value': value,
                                          'shape': shape}
    def declare_template(self,path):
        '''
        Should be run during execution engine
        stores the template files e.g .bdf in a dictionary
        '''
        if path in  self._template_to_checksum:
            raise KeyError(f'The path: {path} is already in the template dictionary')
        self._plugin_templates.append(path)
        if os.path.exists(path):
            with open(path, 'rb') as  f:
                data = f.read()
                h = hashlib.sha256(data).hexdigest()
        else:
            h = 'file_does_not_exist_at_setup'
        self._template_to_checksum[str(path)] = h
        
    def add_executable_data(self, path, description):
        '''
        Allows user to version/provide data about the executable used
        should be used in the setup portion of the plugin.
        Allows for custom scripting within the plugin setup.
        User should pair the path with either a string or dictionary
        of data but ultimately up to the user, should be JSONable though. 
        '''
        self._executable_to_data[path] = description
           
        

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

