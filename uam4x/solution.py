from __future__ import annotations
from typing import Callable
import os
import json
import importlib.util
from uam4x.utils import check_and_get_config, check_numpy_shape, check_if_scalar, is_numpy_array, normalize_shape,is_main_process,registry_writes_enabled,check_continuous,handle_brackets
from collections.abc import Sequence
from numbers import  Real
from itertools import  product
from uam4x.log import get_log
from uam4x.utils import log
import numpy as np
import inspect
import hashlib
import copy
_plugin_registry = {}

class Group:
    log = get_log(log, object_name='group')

    def __init__(self):
        self.log = Group.log
        self.log.debug(f'initialize')
        self.links=[]
        self.steps={}
        self._execution_steps={}
        self.cycle_solver=''
        self._approx_totals={'compute':False}
        self._is_cycle=False
        self.group_name=''


    def set_cycle_solver(self,solver_name):
        self.cycle_solver=solver_name

    def approx_totals(self,method='fd',step=10**-6,form='forward'):
        self._approx_totals={'compute':True,'method':method,'step':step,'form':form}

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

    def _make_step_info(self,subsystem, subsystem_options, plugin_hash=None):
        if isinstance(subsystem,type) and issubclass(subsystem,Plugin):

            if plugin_hash is None:
                src = inspect.getsource(subsystem)
                h = hashlib.sha256()
                h.update(src.encode("utf-8"))
                plugin_hash = h.hexdigest()

            return {
                "plugin": subsystem,
                "plugin_options": subsystem_options,
                "plugin_hash": plugin_hash,
            }
        
        elif isinstance(subsystem,Group):
            #group_dict={}
            #for name,info in plugin_class.steps.items():
            #    group_dict[name]=info
            #return group_dict

            # two approaches here: one is that the groups are fixed and added
            #as they are added (logic above) or the group itself is added only
            #which still allows for access to the groups plugin based data
            #then the group will be resolve at the _to_dict stage
            return subsystem
        
        else: 
            raise TypeError(f'Only plugin classes and group objects are allowed')
        

    def add_step(self, subsystem, step_name=None, plugin_options:dict={}):
        self.log.debug(f'add step: {step_name}')
        if isinstance(subsystem,type) and not issubclass(subsystem, Plugin):
            self.log.error("add_step: plugin_class should be a callable (class).")
            raise TypeError("plugin_class should be a callable (class).")
        if step_name is  not None and  ('[' in  step_name or  ']' in  step_name):
            self.log.error(f'steps cannot have the character "[" or "]" in them')
            raise ValueError(f'steps cannot have the character "[" or "]" in them')
        # plugin_name should be a string, not a class
        if step_name is None:
            step_name = "step" + str(len(self.steps)+1)
        if isinstance(subsystem,Group):
            subsystem.group_name=step_name

        step_info = self._make_step_info(subsystem, plugin_options)
        self.steps[step_name] = step_info


    def remove_step(self, step_name):
        self.log.debug(f'remove step: {step_name}')
        if step_name in self.steps:
            self.steps.pop(step_name)

    def edit_step(self, step_name, plugin_options:dict=None):
        '''
        You can only edit plugin steps directly, to edit Groups, 
        should be done at the group level. i.e. group.edit_plugin_step
        '''
        self.log.debug(f'edit step: {step_name}')
        if step_name in self.steps:
            step = self.steps[step_name]
            if isinstance(step,Group):
                raise ValueError(f'Only plugins are editable at the Group level, sub groups'
                                'must be edited in their respective groups')
            if plugin_options is not None:
                current_options = step["plugin_options"]
                for option_name, info in plugin_options.items():
                    current_options[option_name] = info
            else:
                step['plugin_options'] = {}
        else:
            raise KeyError(f'The step {step_name} is not a step')
    
    def add_plugin_option(self, step_name:str, option_name:str, option_value):
        self.log.debug(f'add plugin option {option_name} to step {step_name}')
        if step_name in self.steps:
           step = self.steps[step_name]
           if isinstance(step,Group):
                raise ValueError(f'Only plugins are editable at the Group level, sub groups'
                                'must be edited in their respective groups')
           if option_name in  step['plugin_options']:
               print(f'{option_name} already a plugin option. Overwriting value: {step["plugin_options"][option_name]}')
           step["plugin_options"][option_name] = option_value
        else:
            raise KeyError(f'The step {step_name} is not a step')
    
    def remove_plugin_option(self, step_name:str, option_name:str):
        self.log.debug(f'remove plugin option {option_name} to step {step_name}')
        if step_name in self.steps:
            step = self.steps[step_name]
            if isinstance(step,Group):
                raise ValueError(f'Only plugins are editable at the Group level, sub groups'
                                'must be edited in their respective groups')
            step["plugin_options"].pop(option_name)
        else:
            raise KeyError(f'The step {step_name} is not a step')
    
    def _to_dict(self) -> dict:
        # Serialize steps, converting plugin class to its name
        self.log.debug(f'write to dict')
        steps_serialized = {}
        for step_name, step_info in self.steps.items():
            if isinstance(step_info,dict):
                plugin_class = step_info["plugin"]
                plugin_name = plugin_class.__name__
                steps_serialized[step_name] = {
                    "plugin": plugin_name,
                    "plugin_options": step_info["plugin_options"],
                    "plugin_hash": step_info["plugin_hash"],
                    'specification':'plugin'
                }
            else:
                #if its a group instead
                steps_serialized[step_name]=step_info._to_dict()
                steps_serialized[step_name]['specification']='group'
        
        return {
            "steps": steps_serialized,
            "links": self.links,
            "cycle_solver":self.cycle_solver,
            "approx_totals":self._approx_totals,
            "group_name":self.group_name,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> 'Solution':
        group = cls()
        cls.log.debug('solution from dict')
        # Reconstruct steps, mapping plugin name to class if possible
        for step_name, step_data in data["steps"].items():
            if step_data['specification']=='plugin':
                plugin_name = step_data["plugin"]
                subsystem = get_plugin_class(plugin_name)
                subsystem_options = step_data["plugin_options"]
                plugin_hash = step_data["plugin_hash"]
            elif step_data['specification']=='group':
                subsystem=Group._from_dict(step_data)
                subsystem_options={}
                plugin_hash=None


            step_info = group._make_step_info(subsystem, subsystem_options, plugin_hash)
            group.steps[step_name] = step_info

        group.links = data["links"]
        group.cycle_solver=data["cycle_solver"]
        group._approx_totals=data["approx_totals"]
        group.group_name=data["group_name"]
        return group
    
    def generate_plugins(self):
        plugins={}
        for step_name,data in self.steps.items():
            if isinstance(data,dict):
                plugin=data["plugin"]()
                plugin.setup_plugin(data['plugin_options'])
                plugins[step_name]=plugin
            elif isinstance(data,Group):
                group_plugs=data.generate_plugins()
                plugins[step_name]=group_plugs

        return plugins
            
    def _get_all_plugin_names_to_classes(self,head_dict,group_name):
        
        for name,data in self.steps.items():
            if isinstance(data,dict):
                head_dict[f'{group_name}.{name}']=data["plugin"]
            elif isinstance(data,Group):
                data._get_all_plugin_names_to_classes(head_dict,f'{group_name}.{name}')     

        
            
    def _get_all_plugin_names_to_hashes(self,head_dict,group_name):
        
        for name,data in self.steps.items():
            if isinstance(data,dict):
                head_dict[f'{group_name}.{name}']=data["plugin_hash"]
            elif isinstance(data,Group):
                data._get_all_plugin_names_to_hashes(head_dict,f'{group_name}.{name}')  

    def sort_steps(self):
        self._execution_steps={}
        #generate a list of dependent links
        for name,data in self.steps.items():
            if isinstance(data,Group):
                data.sort_steps()
        #may want to handle the case where there is a single plugin/group 
        #within a  group (either a raise or do nothing?)
        reduced_links=[]
        counter=0
        step_to_location={}
        location_to_step={}
        steps=set()
        for independent,dependent in self.links:
            independent=independent.split('.')[0]
            dependent=dependent.split('.')[0]
            steps.add(independent)
            steps.add(dependent)
            pair=[independent,dependent]
            if pair not in reduced_links:
                reduced_links.append(pair)
            if independent not in step_to_location:
                step_to_location[independent]=counter
                location_to_step[counter]=independent
                counter+=1
            if dependent not in step_to_location:
                step_to_location[dependent]=counter
                location_to_step[counter]=dependent
                counter+=1

        adj = [[] for _ in range(len(steps))]
        in_degree = [0]*len(steps) 
        for independent,dependent in reduced_links:
            adj[step_to_location[independent]].append(step_to_location[dependent])
            in_degree[step_to_location[dependent]]+=1
        
        sources=[]
        for loc in step_to_location.values():
            if in_degree[loc]==0:
                sources.append(loc)
        
        sorted_steps=[]
        while sources:
            source=sources.pop()
            sorted_steps.append(source)
            for check_node in adj[source]:
                in_degree[check_node]-=1
                if in_degree[check_node]==0:
                    sources.append(check_node)
            
        if len(sorted_steps) != len(steps):
            print(f"Cycle detected in group {self.group_name}, using declared step order in that group")
            self._execution_steps=copy.deepcopy(self.steps)
            self._is_cycle=True
            if not self.cycle_solver:
                raise ValueError(f'Cycle solver must be selected if cycle is used')
        else:
            for name,data in self.steps.items():
                if name not in list(step_to_location.keys()):
                    self._execution_steps[name]=data
            for step_order in sorted_steps:
                step=location_to_step[step_order]
                if step!='vehicles':
                    self._execution_steps[step]=self.steps[step]
            
        
    def _check_links_to_plugins(self,plugins):
        
        for step_name in self._execution_steps:
            #handle plugin case
            if isinstance(self._execution_steps[step_name],dict):
                input_names,output_names=self.get_plugin_input_and_output_names(step_name)
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
                    
            elif isinstance(self._execution_steps[step_name],Group):
                self._execution_steps[step_name]._check_links_to_plugins(plugins[step_name])
        

    def get_plugin_input_and_output_names(self,step_name):
        input_names=[]
        output_names=[]
        for (independent, dependent) in self.links:
            # remove indices from index based links
            if '[' in  independent:
                independent, _ = handle_brackets(independent)
            if dependent.split('.')[0] == step_name:
                input_names.append(dependent)
            if independent.split('.')[0] == step_name:
                output_names.append(independent)
        return set(input_names), set(output_names)
    
    #def _check_link_types(self,plugins,study):
    #    for name, group_or_plugin_dict in self._execution_steps.items():
    #        if isinstance(group_or_plugin_dict,dict):
    #            
    #        elif isinstance(group_or_plugin_dict,Group):
    #            step._check_link_types(plugins[name],study)
    #        
    #    pass

            
    def get_all_groups(self,name_to_group,parent_name):
        for name,step in self.steps.items():
            if isinstance(step,Group):
                full_name=parent_name+'.'+name
                name_to_group[full_name]=step
                step.get_all_groups(name_to_group,full_name)
        return name_to_group
    
    def get_var_type(self,name,plugins):
        return self.get_var_data(name,plugins)['type']
        
    def get_var_value(self,name,plugins):
        return self.get_var_data(name,plugins)['value']
    
    def get_var_shape(self,name,plugins):
        return self.get_var_data(name,plugins)['shape']
        
    def get_var_data(self,name,plugins):
        split_name=name.split('.')
        step_name=split_name[0]
        rest_name='.'.join(split_name[1:])
        step=plugins[step_name]
        if isinstance(step,Plugin):
            if rest_name in step.plugin_input_names:
                var_type=step.plugin_input_names[rest_name]['type']
                var_value=step.plugin_input_names[rest_name]['value']
                var_shape=step.plugin_input_names[rest_name]['shape']

            elif rest_name in step.plugin_output_names:
                var_type=step.plugin_output_names[rest_name]['type']
                var_value=step.plugin_output_names[rest_name]['value']
                var_shape=step.plugin_output_names[rest_name]['shape']
            else:
                raise KeyError(f'Provided path is not an input/output to the plugin {step_name}')
            var_data={
                'value':var_value,
                'type':var_type,
                'shape':var_shape}
            return var_data
        elif isinstance(step,dict):
            local_plugins=plugins[step_name]
            return self.get_var_data(rest_name,local_plugins)
        
    def _get_global_execution_order_plugins(self,sol_name,plugins,global_name_to_plugin):
        for name,step in self._execution_steps.items():
            if isinstance(step,Group):
                full_name=sol_name+'.'+name
                local_plugs=plugins[name]
                step._get_global_execution_order_plugins(full_name,local_plugs,global_name_to_plugin)
            else:
                global_name_to_plugin[sol_name+'.'+name]=plugins[name]

    def _build_links(self,base_name):
        global_links=[]
        for output,input in self.links:
            if output.split('.')[0]!='vehicles':
                output=base_name+'.'+output
            input=base_name+'.'+input
            global_links.append([output,input])
        for name,step in self._execution_steps.items():
            if isinstance(step,Group):
                new_base_name=base_name+'.'+name
                global_links+=step._build_links(new_base_name)
        return global_links
            

class Solution(Group):
    log = get_log(log, object_name='solution')
    def __init__(self) -> None:
        super().__init__()
        self.tracked_outputs = []
        self.log = Solution.log
        self.log.debug(f'initialize')
        self.step_name='solution'

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


    def _to_dict(self) -> dict:
        # Serialize steps, converting plugin class to its name
        self.log.debug(f'write to dict')
        steps_serialized = {}
        for step_name, step_info in self.steps.items():
            if isinstance(step_info,dict):
                plugin_class = step_info["plugin"]
                plugin_name = plugin_class.__name__
                steps_serialized[step_name] = {
                    "plugin": plugin_name,
                    "plugin_options": step_info["plugin_options"],
                    "plugin_hash": step_info["plugin_hash"],
                    'specification':'plugin'
                }
            else:
                #if its a group instead
                steps_serialized[step_name]=step_info._to_dict()
                steps_serialized[step_name]['specification']='group'
        
        return {
            "steps": steps_serialized,
            "links": self.links,
            'tracked_outputs':self.tracked_outputs
        }

    @classmethod
    def _from_dict(cls, data: dict) -> 'Solution':
        sol = cls()
        cls.log.debug('solution from dict')
        # Reconstruct steps, mapping plugin name to class if possible
        for step_name, step_data in data["steps"].items():
            if step_data['specification']=='plugin':
                plugin_name = step_data["plugin"]
                subsystem = get_plugin_class(plugin_name)
                subsystem_options = step_data["plugin_options"]
                plugin_hash = step_data["plugin_hash"]
            elif step_data['specification']=='group':

                subsystem=Group._from_dict(step_data)
                subsystem_options={}
                plugin_hash=None

            step_info = sol._make_step_info(subsystem, subsystem_options, plugin_hash)
            sol.steps[step_name] = step_info

        sol.tracked_outputs = data["tracked_outputs"]
        sol.links = data["links"]
        return sol

    def get_all_plugin_names_to_classes(self):
        plugin_names_to_class={}
        for name,data in self.steps.items():
            if isinstance(data,dict):
                plugin_names_to_class[name]=data["plugin"]
            elif isinstance(data,Group):
                data._get_all_plugin_names_to_classes(plugin_names_to_class,name)

        return plugin_names_to_class
    
    def get_all_plugin_names_to_hashes(self):
        plugin_names_to_hash={}
        for name,data in self.steps.items():
            if isinstance(data,dict):
                plugin_names_to_hash[name]=data["plugin_hash"]
            elif isinstance(data,Group):
                data._get_all_plugin_names_to_hashes(plugin_names_to_hash,name)

        return plugin_names_to_hash
    

    def get_plugin_input_and_output_names(self,step_name):
        input_names=[]
        output_names=[]
        for (independent, dependent) in self.links:
            # remove indices from index based links
            if '[' in  independent:
                independent, _ = handle_brackets(independent)
            if dependent.split('.')[0] == step_name:
                input_names.append(dependent)
            if independent.split('.')[0] == step_name:
                output_names.append(independent)
        for independent in self.tracked_outputs:
            if independent.split('.')[0] == step_name:
                output_names.append(independent)
        return set(input_names), set(output_names)
    
    def get_all_groups(self):
        name_to_group={'solution':self}
        for name,step in self.steps.items():
            if isinstance(step,Group):
                full_name='solution'+'.'+name
                name_to_group[full_name]=step
                step.get_all_groups(name_to_group,full_name)
        return name_to_group
    
    def get_group(self,dotted_path):
        split_path=dotted_path.split('.')
        groups=self.get_all_groups()
        if split_path[0]!='solution':
            dotted_path='solution.'+dotted_path
        return groups[dotted_path]
    
    def recurse_plugin_dict(self,split_name,plugins):
        #last term in the initial string (is a group)
        #then plugins must be of form last_group:{data}
        if len(split_name)==1:
            return plugins[split_name[0]]
        else:
            start=split_name[0]
            local_plug=plugins[start]
            return self.recurse_plugin_dict(split_name[1:],local_plug)
        
    def _check_link_types(self,plugins,study):
        groups=self.get_all_groups()
        bad_links=[]
        for group_name,group in groups.items():
            split_name=group_name.split('.')
            if split_name[0]=='solution' and len(split_name)==1:
                local_plugins=plugins
            else:
                split_name=split_name[1:]
                local_plugins=self.recurse_plugin_dict(split_name,plugins)
            for output_link,input_link in group.links:
                indexes_specified = False

                if '[' in  output_link:
                    output_link, _ = handle_brackets(output_link)
                    indexes_specified = True
                if output_link.split('.')[0]=='vehicles':
                    output_type=study.get_param_type(output_link)
                else:
                    output_type=self.get_var_type(output_link,local_plugins)
                
                input_type=self.get_var_type(input_link,local_plugins)
                if indexes_specified and output_type != np.ndarray:
                    raise ValueError(f'Indexes may only be specified for arrays. The link: {input_link} is not valid')
                if indexes_specified and  output_type == np.ndarray and  check_continuous(input_type) and input_type != np.ndarray:
                    pass
                elif output_type != input_type:
                    bad_links.append((output_link,output_type, input_link, input_type))

        if bad_links:
            msg = f'Link types must match, the following links have different types:\n' + "\n".join(f"{out_}:{out_type}->{in_}:{in_type}" for  out_,out_type, in_,in_type in  bad_links)
            self.log.error(msg)
            raise TypeError(msg)
        
    def get_all_used_vehicle_params(self):
        groups=self.get_all_groups()
        all_used_vehicle_params=[]
        for _,group in groups.items():
            for independent,_ in group.links:
                independent_group=independent.split('.')[0]
                if independent_group=='vehicles':
                    all_used_vehicle_params.append(independent)
        return all_used_vehicle_params


                
    def _check_optimization_discrete(self,plugins):
        groups=self.get_all_groups()
        bad_types={}
        for group_name,group in groups.items():
            split_name=group_name.split('.')
            if split_name[0]=='solution' and len(split_name)==1:
                local_plugins=plugins
            else:
                split_name=split_name[1:]
                local_plugins=self.recurse_plugin_dict(split_name,plugins)
            for output_link,input_link in group.links:
                if output_link.split('.')[0]=='vehicles':
                    continue

                

                if '[' in  output_link:
                    output_link, _ = handle_brackets(output_link)
                
                output_type=self.get_var_type(output_link,local_plugins)

                if not check_continuous(output_type):
                    bad_types[output_link]=output_type
        if bad_types:
            bad_links = [f"{output_link}: {output_type}" for output_link, output_type in bad_types.items()]
            detail = "\n".join(f"  - {line}" for line in bad_links)
            msg = (
                "Links between optimization components must be continuous (of type float)\n"
                "Discrete variables linked between plugins:\n"
                f"{detail}"
                )
            self.log.error(msg)
            raise TypeError(msg)
        

    def get_global_execution_order_plugins(self,plugins):
        sol_name='solution'
        global_plugin_name_to_plugin={}
        for name,step in self._execution_steps.items():
            if isinstance(step,Group):
                full_name=sol_name+'.'+name
                local_plugs=plugins[name]
                step._get_global_execution_order_plugins(full_name,local_plugs,global_plugin_name_to_plugin)
            else:
                global_plugin_name_to_plugin[sol_name+'.'+name]=plugins[name]

        return global_plugin_name_to_plugin
    
    def _build_global_links(self):
        global_links=[]
        base_name='solution'
        for output,input in self.links:
            if output.split('.')[0]!='vehicles':
                output=base_name+'.'+output
            input=base_name+'.'+input
            global_links.append([output,input])
        for name,step in self._execution_steps.items():
            if isinstance(step,Group):
                child_name=base_name+'.'+name
                global_links+=step._build_links(child_name)
        return global_links
    
    def _check_link_indices(self,plugins,study):
        groups=self.get_all_groups()
        for group_name,group in groups.items():
            split_name=group_name.split('.')
            if split_name[0]=='solution' and len(split_name)==1:
                local_plugins=plugins
            else:
                split_name=split_name[1:]
                local_plugins=self.recurse_plugin_dict(split_name,plugins)
            for output_link,input_link in group.links:
                indexes_specified = False

                if '[' in  output_link:
                    cleaned_output_name, inds = handle_brackets(output_link)
                    if output_link.split('.')[0]=='vehicles':
                        try:
                            a=study.get_value(output_link)
                            src_shape=a.shape
                        except:
                            raise ValueError(f'The indices in {output_link} provided are not a valid means to index\n')
                    else:
                        a=self.get_var_value(cleaned_output_name,local_plugins)
                        a_shape=self.get_var_shape(cleaned_output_name,local_plugins)
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

                    b = self.get_var_value(input_link,local_plugins)
                    b_shape = self.get_var_shape(input_link,local_plugins)
                    b_type = self.get_var_type(input_link,local_plugins)

                    src_size = 1 if src_shape == () else int(np.prod(src_shape))
                    #float case
                    if src_size ==1  and  b_type == float:
                            pass
                    elif b is not None:
                        if src_shape != b.shape:
                            raise ValueError(f'The source indices must match the target shape. The input {output_link} with indices.\n'
                                             f'The size of the source indices is {src_shape} and the destination has shape {b.shape}')                    

                    elif b_shape is  not None:
                        if src_shape != b_shape:
                            raise ValueError(f'The source indices must match the target shape. The input {output_link} with indices.\n'
                                             f'The size of the source indices is {src_shape} and the destination has shape {b_shape}')



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



    def build_partials(self, input_names, output_names,step_name):
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
                    if inp in self.plugin_input_names:
                        input_type=self.plugin_input_names[inp]['type']
                    else:
                        raise KeyError(f'The input {inp} declared as a partial is not an input in the Plugin')
                    if not check_continuous(input_type):
                        raise TypeError(f'Variables with declared partials must be continuous the variable:\n'
                               f'{inp} is not continuous and has type {input_type}')
            if data['of'] != ['*']:
                for out in  data['of']:
                    partial_outs.add(out)
                    if out in self.plugin_output_names:
                        output_type=self.plugin_output_names[out]['type']
                    else:
                        raise KeyError(f'The output {out} declared as a partial is not an output in the Plugin')
                    if not check_continuous(output_type):
                        raise TypeError(f'Variables with declared partials must be continuous the variable:\n'
                               f'{out} is not continuous and has type {output_type}')

        # Collect actual names
        local_output_names = set()
        local_input_names = set()
        start_ind=len(step_name.split('.'))

        out_vio=[]
        for output_name in  output_names:
            local_output_name = '.'.join(output_name.split('.')[start_ind:])
            local_output_names.add(local_output_name)
            output_type=self.plugin_output_names[local_output_name]['type']
            
            if not check_continuous(output_type):
                out_vio.append(output_name)
        in_vio=[]
        for input_name in  input_names:
            local_input_name = '.'.join(input_name.split('.')[start_ind:])
            local_input_names.add(local_input_name)
            input_type=self.plugin_input_names[local_input_name]['type']

            if not check_continuous(input_type):
                in_vio.append(input_name)


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
                if out_vio:
                    raise TypeError(f'Variables with declared partials must be continuous the variables:\n'
                               f'{out_vio} are not continuous but were declared partials')
            else:
                ofs = data['of']

            if data['wrt'] == ['*']:
                wrts = local_input_names
                if in_vio:
                    raise TypeError(f'Variables with declared partials must be continuous the variables:\n'
                        f'{in_vio} are not continuous but were declared partials')
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
            elif value is not None and isinstance(value,np.ndarray) and var_type is np.ndarray:
                shape=value.shape

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

    if is_main_process() and registry_writes_enabled():
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

