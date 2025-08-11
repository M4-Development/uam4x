import json
import os
import sys
import importlib
import uam4x.components
from uam4x.define_vehicles import Vehicle
import copy

class _Overwrite():
    def __init__(self, param_state:dict) -> None:
        self.param_state = param_state
        self.name_to_value = copy.copy(param_state)

    def add_item(self, name, value):
        if name in self.name_to_value:
            print(f"WARNING key already has value: {name}")
        self.name_to_value[name] = value

    def add_items(self, name_to_value):
        for name in name_to_value:
            if name in self.name_to_value:
                print(f"WARNING key already has value: {name}")
            self.name_to_value[name] = name_to_value[name]

    def get_item(self, name):
        return self.name_to_value[name]

    def __contains__(self, item):
        return item in self.name_to_value

    def __eq__(self, other) -> bool:
        same = True
        if len(other.param_state.keys()) == len(self.param_state.keys()):
            for key, value in self.param_state.items():
                if not other.param_state[key] == value:
                    same = False
        else:
            same = False
        return same

    def tracked_outputs_as_dict(self, tracked_outputs):
        results = {}
        for name, value in self.name_to_value.items():
            if (
                name in self.param_state
                or name in tracked_outputs
                or ('des_var_names' in self.param_state and name in self.param_state['des_var_names'])
            ):
                results[name] = value


        return results
class _Optimization():
    def __init__(self, name_to_value) -> None:
        self.des_var_names = name_to_value["des_var_names"]
        self.des_var_lower = name_to_value["des_var_lower"]
        self.des_var_upper = name_to_value["des_var_upper"]
        self.constraint_names = name_to_value["constraint_names"]
        self.constraint_lower = name_to_value["constraint_lower"]
        self.constraint_upper = name_to_value["constraint_upper"]
        self.objective_name = name_to_value["objective_name"]

        assert len(self.des_var_names) == len(self.des_var_lower), "des_var_lower must match length of des_var_names"
        assert len(self.des_var_names) == len(self.des_var_upper), "des_var_upper must match length of des_var_names"

        assert len(self.constraint_names) == len(self.constraint_lower), "constraint_lower must match length of constraint_names"
        assert len(self.constraint_names) == len(self.constraint_upper), "constraint_upper must match length of constraint_names"

    def as_dict(self):
        return {
            "des_var_names": self.des_var_names,
            "des_var_lower": self.des_var_lower,
            "des_var_upper": self.des_var_upper,
            "constraint_names": self.constraint_names,
            "constraint_lower": self.constraint_lower,
            "constraint_upper": self.constraint_upper,
            "objective_name": self.objective_name,
        }


class _ParameterSweep():
    def __init__(self, name_to_value: dict) -> None:
        self.child = None
        self.length = None
        self.param_name_to_values = {}
        for name, value in name_to_value.items():
            if name == "parameter_sweep":
                self.child = _ParameterSweep(value)
            elif name == "optimization":
                self.child = _Optimization(value)
            else:
                assert isinstance(value,list), f"parameter values must be list: {name}: {value}"
                assert not name in self.param_name_to_values, f"Param name already exists: {name}"
                if not self.length:
                    self.length = len(value)
                else:
                    assert self.length == len(value), f"parameter sweep must have same length for value lists"

                self.param_name_to_values[name] = value
    def set_child(self, child):
        assert isinstance(child, _Optimization) or isinstance(child, _ParameterSweep), "Child must be parameter sweep or opimization"
        self.child = child
    def get_states(self, parent_state = {}):
        states = []
        for i in range(self.length):
            state = {}
            for name, values in self.param_name_to_values.items():
                state[name] = values[i]
            if isinstance(self.child, _Optimization):
                state['optimization'] = True
                state.update(self.child.as_dict())
                states.append(state)
            elif isinstance(self.child, _ParameterSweep):
                states.update(self.child.get_states(state))
            else:
                state['optimization'] = False
                states.append(state)
        return states
    def as_dict(self):
        result = self.param_name_to_values
        if isinstance(self.child, _ParameterSweep):
            result["parameter_sweep"] = self.child.as_dict()
        elif isinstance(self.child, _Optimization):
            result["optimization"] = self.child.as_dict()
        return result

class Tradestudy():
    def __init__(self, name) -> None:
        self.name = name
        self.trade = None

    def load_from_dict(self, input_dict):
        for study_type, value_dict in input_dict.items():
            if study_type == "parameter_sweep":
                self.trade = _ParameterSweep(value_dict)
            elif study_type == "optimization":
                self.trade = _Optimization(value_dict)
            else:
                print(f"WARNING: UNKNOWN STUDY TYPE {study_type}")

    def define_parameter_sweep(self, name_to_values_dict, child=None):
        param_sweep = _ParameterSweep(name_to_values_dict)
        if child:
            param_sweep.set_child(child)
        return param_sweep

    def define_optimization(self,
        des_var_names, des_var_lower, des_var_upper,
        constraint_names, constraint_lower, constraint_upper,
        objective_name
    ):
        return _Optimization({
            "des_var_names": des_var_names,
            "des_var_lower": des_var_lower,
            "des_var_upper": des_var_upper,
            "constraint_names": constraint_names,
            "constraint_lower": constraint_lower,
            "constraint_upper": constraint_upper,
            "objective_name": objective_name,
        })

    def as_dict(self):
        result = {}
        if isinstance(self.trade, _ParameterSweep):
            result['parameter_sweep'] = self.trade.as_dict()
        elif isinstance(self.trade, _Optimization):
            result['optimization'] = self.trade.as_dict()
        return result

class Study:
    def __init__(self) -> None:
        self._vehicles = Vehicles(self)
        self._solution = Solution(self)
        self._control = Control(self)
        self._overwrites = []
        self.json_path = None

    @property
    def vehicles(self):
        return self._vehicles

    @vehicles.setter
    def vehicles(self, vehicles):
        self._vehicles = vehicles

    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, solution):
        self._solution = solution

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, control):
        self._control = control

    def get_value(self, info:str):
        split_info = info.split(".")
        if split_info[0].lower() == "vehicles":
            split_info.pop(0)
        if split_info[0] in self.vehicles.get_vehicle_names():
            vehicle = self.vehicles.get_vehicle(split_info[0])
            if split_info[1] in vehicle.get_component_names():
                component = vehicle.get_component(split_info[1])
                if split_info[2] in component.get_mode_names():
                    component_mode = component.get_mode(split_info[2])
                    value = component_mode.get_attribute(split_info[3])
                else:
                    value = component.get_attribute(split_info[2])
            elif split_info[1] in vehicle.get_mode_names():
                vehicle_mode = vehicle.get_mode(split_info[1])
                component = vehicle_mode.get_component(split_info[2])
                value = component.get_attribute(split_info[3])
            else:
                raise RuntimeError("This value does not exist")
        else:
            raise RuntimeError("This value does not exist")
        return value

    def set_value(self, info:str, value):
        split_info = info.split(".")
        if split_info[0].lower() == "vehicles":
            split_info.pop(0)
        if split_info[0] in self.vehicles.get_vehicle_names():
            vehicle = self.vehicles.get_vehicle(split_info[0])
            if split_info[1] in vehicle.get_component_names():
                component = vehicle.get_component(split_info[1])
                if split_info[2] in component.get_mode_names():
                    component_mode = component.get_mode(split_info[2])
                    component_mode.set_attribute(split_info[3], value)
                else:
                    component.set_attribute(split_info[2],value)
            elif split_info[1] in vehicle.get_mode_names():
                vehicle_mode = vehicle.get_mode(split_info[1])
                component = vehicle_mode.get_component(split_info[2])
                component.set_attribute(split_info[3], value)
            else:
                raise RuntimeError("This value does not exist")
        else:
            raise RuntimeError("This value does not exist")

    def save(self,):
        assert self.json_path, "No existing path detected, use save_as"
        self.save_as(self.json_path)

    def save_as(self, filename):
        self.json_path = filename
        f = open(filename, "w")
        data = {}
        control_dict = self._control.save_control()
        vehicles_dict = self._vehicles.save_vehicles()
        solution_dict = self._solution.save_solution()
        results_dict = self._save_results()
        data["control"] = control_dict
        data["vehicles"] = vehicles_dict
        data["solution"] = solution_dict
        data["results"] = results_dict
        f.seek(0)
        json.dump(data, f, indent=4)
        f.close()

    def load(self, filename):
        self.json_path = filename
        f = open(filename, "r")
        data = json.load(f)
        control = data["control"]
        vehicles = data["vehicles"]
        solution = data["solution"]
        self._control.load_control(control)
        self._vehicles.load_vehicles(vehicles)
        self._solution.load_solution(solution)

    def _get_value_with_overwrites(self, name, state):
        temp_over = _Overwrite(state)
        if name in state:
            return temp_over.get_item(name)
        for overwrite in self._overwrites:
            if temp_over == overwrite:
                if name in overwrite:
                    return overwrite.get_item(name)
                break
        return self.get_value(name)

    def _add_overwrite(self, state, name, value):
        temp_over = _Overwrite(state)
        for overwrite in self._overwrites:
            if temp_over == overwrite:
                overwrite.add_item(name, value)
                return
        temp_over.add_item(name, value)
        self._overwrites.append(temp_over)

    def _save_results(self):
        results = []
        for overwrite in self._overwrites:
            results.append(overwrite.tracked_outputs_as_dict(self.solution.tracked_outputs))
        return results
    def get_plugin_inputs(self, step_name, state):
        inputs = {}
        for (independent, dependent) in self.solution.links:
            dependent_split = dependent.split('.')
            if dependent_split[0] == step_name:
                inputs[".".join(dependent_split[1:])] = self._get_value_with_overwrites(independent, state)
        return inputs

    def get_plugin_input_and_output_names(self, step_name):
        input_names = []
        output_names = []
        for (independent, dependent) in self.solution.links:
            if dependent.split('.')[0] == step_name:
                input_names.append(dependent)
            if independent.split('.')[0] == step_name:
                output_names.append(independent)
        for independent in self.solution.tracked_outputs:
            if independent.split('.')[0] == step_name:
                output_names.append(independent)
        return set(input_names), set(output_names)


class Control:
    def __init__(self, study) -> None:
        # defines the trade study that you want to execute (ex. very span)
        # controls what you are going to change within the framework to perform the trade study
        self.study = study
        self._trade_studies = {}
        self._title = "Default Title"
        self._author = "Default Author"
        self._description = "Default Description"

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def author(self):
        return self._author

    @author.setter
    def author(self, name):
        self._author = name

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def trade_studies(self):
        return self._trade_studies

    def save_control(self):
        trade_dict = {}
        for name, trade in self._trade_studies.items():
            trade_dict[name] = trade.as_dict()
        control = {"title": self._title, "author": self._author, "description": self._description, "trade studies": trade_dict}
        return control

    def load_control(self, control:dict):
        self._title = control["title"]
        self._author = control["author"]
        self._description = control["description"]
        for tradestudy_name, trade_data_dict in control["trade studies"].items():
            self._trade_studies[tradestudy_name] = Tradestudy(tradestudy_name)
            self._trade_studies[tradestudy_name].load_from_dict(trade_data_dict)

    def add_tradestudy(self, tradestudy:Tradestudy):
        self._trade_studies[tradestudy.name] = tradestudy

    # def create_nested_trade_study(self, study_name:str, path, values:list):
    #     if study_name not in self.trade_studies:
    #         self._trade_studies[study_name] = {}
    #     trade_study = self._trade_studies[study_name]
    #     existing_paths = []
    #     if "parameter_sweep" not in trade_study and "optimization" not in trade_study:
    #         parm_sweep = trade_study
    #         build_nest = True
    #     else:
    #         [parm_sweep, current_nest_name] = self.get_nest(trade_study, existing_paths)
    #         if current_nest_name == "optimization":
    #             build_nest = False
    #         else:
    #             build_nest = True
    #     if build_nest:
    #         nest = {}
    #         if type(path) is str:
    #             if path not in existing_paths:
    #                 nest[path] = values
    #         else:
    #             build = False
    #             for i in range(len(path)):
    #                 if i == 0:
    #                     if path[i] not in existing_paths:
    #                         nest[path[i]] = values[i]
    #                         build = True
    #                     new_nest = nest
    #                 if i != 0:
    #                     if path[i] not in existing_paths:
    #                         if build:
    #                             new_nest["parameter_sweep"] = {}
    #                             new_nest = new_nest["parameter_sweep"]
    #                         new_nest[path[i]] = values[i]
    #                         build = True
    #         if len(nest) > 0:
    #             parm_sweep["parameter_sweep"] = nest

    # def create_optimization(self, study_name:str, design_vars:list, constraints:dict, objective:str):
    #     if study_name not in self.trade_studies:
    #         self._trade_studies[study_name] = {}
    #     trade_study = self._trade_studies[study_name]
    #     existing_paths = []
    #     if "parameter_sweep" not in trade_study and "optimization" not in trade_study:
    #         parm_sweep = trade_study
    #     else:
    #         [parm_sweep, current_nest_name] = self.get_nest(trade_study, existing_paths)
    #     optimization = {}
    #     if current_nest_name == "optimization":
    #         if isinstance(parm_sweep["optimization"], dict):
    #             old_opt = parm_sweep["optimization"]
    #             build = True
    #             for path in existing_paths:
    #                 if path in design_vars or path in constraints or path in objective:
    #                     build = False
    #             if build:
    #                 new_opt = {"design_vars": design_vars, "constraints": constraints, "objective": objective}
    #                 if new_opt != old_opt:
    #                     optimization = [old_opt, new_opt]
    #         elif isinstance(parm_sweep["optimization"], list):
    #             build = True
    #             for path in existing_paths:
    #                 if path in design_vars or path in constraints or path in objective:
    #                     build = False
    #             if build:
    #                 new_opt = {"design_vars": design_vars, "constraints": constraints, "objective": objective}
    #                 if new_opt not in parm_sweep["optimization"]:
    #                     optimization = parm_sweep["optimization"]
    #                     optimization.append(new_opt)
    #     else:
    #         build = True
    #         for path in existing_paths:
    #             if path in design_vars or path in constraints or path in objective:
    #                 build = False
    #         if build:
    #             optimization["design_vars"] = design_vars
    #             optimization["constraints"] = constraints
    #             optimization["objective"] = objective
    #     if len(optimization) > 0:
    #         parm_sweep["optimization"] = optimization

    # def get_nest(self, trade_study, existing_paths, current_nest_name=None):
    #     for name, value in trade_study.items():
    #         if name.lower() == "parameter_sweep" and isinstance(value, dict):
    #             current_nest_name = name.lower()
    #             [nest, current_nest_name] = self.get_nest(value, existing_paths, current_nest_name)
    #             return [nest, current_nest_name]
    #         elif name.lower() == "optimization":
    #             current_nest_name = name.lower()
    #         elif current_nest_name.lower() == "parameter_sweep":
    #             existing_paths.append(name)
    #     return [trade_study, current_nest_name]



class Vehicles:
    def __init__(self, study) -> None:
        """_summary_
        """
        self._vehicles_data = []
        self.study = study

    @property
    def vehicles_data(self):
        return self._vehicles_data

    @vehicles_data.setter
    def vehicles_data(self, dict:dict):
        self._vehicles_data = dict

    def get_vehicle_names(self):
        vehicle_names = []
        for vehicle in self._vehicles_data:
            vehicle_names.append(vehicle.name)
        return vehicle_names

    def load_vehicles(self, vehicles:dict):
        self._vehicles_data = []
        for k in vehicles:
            vehicle_dict = vehicles[k]
            vehicle_name = k
            components_dict = vehicle_dict["components"]
            components_list = self.load_components_list(components_dict=components_dict)
            new_vehicle = Vehicle(self.study, name=vehicle_name, components=components_list)
            modes_dict = vehicle_dict["modes"]
            for key in modes_dict:
                mode_name = key
                mode_components = modes_dict[key]
                new_vehicle.new_mode(mode_name=mode_name, components=mode_components)
            self._vehicles_data.append(new_vehicle)

    def load_components_list(self, components_dict:dict):
        components_list = []
        for k in components_dict:
            component = components_dict[k]
            component_name = k
            component_mode_name = component["mode_name"]
            component_type = component["type"]
            component_attributes_collector = uam4x.components.AttributeCollector(self.study)
            component_attributes_collector.load_attributes_collector(component["attributes_collector"])
            component_modes = component["modes"]
            if hasattr(uam4x.components, component_type):
                new_component = getattr(uam4x.components, component_type)(self.study, name=component_name, mode_name=component_mode_name, parent_attributes=component_attributes_collector)
            else:
                new_component = self.load_custom_component(component_type=component_type, name=component_name, mode_name=component_mode_name, parent_attributes=component_attributes_collector)
            for mode_k in component_modes:
                mode_name = mode_k
                mode_attributes_collector = uam4x.components.AttributeCollector(self.study)
                mode_attributes_collector.load_attributes_collector(component_modes[mode_k])
                new_component.new_mode(mode_name=mode_name)
                new_component.get_mode(mode_name).attributes_collector = mode_attributes_collector
            components_list.append(new_component)
        return components_list

    def load_custom_component(self, component_type, name, mode_name, parent_attributes):
        work_dir = os.getcwd()
        script_path = os.path.abspath(sys.argv[0])
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        script_dir_relative_path = os.path.relpath(script_dir, work_dir)
        python_files = []
        for f in os.listdir(script_dir_relative_path):
            if f.endswith(".py") and f != "__init__.py":
                python_files.append(f)
        for f in python_files:
            file_name = f[:-3]
            module = importlib.import_module(file_name)
            if hasattr(module, component_type):
                new_component = getattr(module, component_type)(self.study, name=name, mode_name=mode_name, parent_attributes=parent_attributes)
                return new_component

    def save_vehicles(self):
        vehicles = {}
        for vehicle in self._vehicles_data:
            vehicles[vehicle.name] = vehicle.save_vehicle()
        return vehicles

    def new_vehicle(self, name:str) -> Vehicle:
        vehicle_names = self.get_vehicle_names()
        if name not in vehicle_names:
            new_vehicle = Vehicle(self.study, name)
            self._vehicles_data.append(new_vehicle)
            return new_vehicle
        else:
            return self.get_vehicle(name)

    def get_vehicle(self, name:str):
        for v in self._vehicles_data:
            if v.name == name:
                return v



class Solution:
    def __init__(self, study) -> None:
        self.study = study
        self.tracked_outputs = []
        self._links = []
        self._steps = {}

    @property
    def links(self):
        return self._links

    @property
    def steps(self):
        return self._steps

    def save_solution(self):
        solution = {
            "links": self._links,
            "steps": self._steps,
            "tracked_outputs": self.tracked_outputs}
        return solution

    def load_solution(self, solution:dict):
        self._links = solution["links"]
        self._steps = solution["steps"]
        self.tracked_outputs = solution["tracked_outputs"]

    def add_link(self, independent:str, dependent:str):
        link = [independent, dependent]
        if link not in self._links:
            self._links.append(link)

    def remove_link(self, independent:str, dependent:str):
        link = [independent, dependent]
        if link in self._links:
            self._links.remove(link)

    def get_step_names(self):
        return [step_name for step_name in self._steps]

    def add_step(self, plugin_name:str, step_name=None, plugin_options:dict={}):
        if step_name is None:
            step_name = "step" + str(len(self._steps)+1)
        step_info = {"plugin_type": plugin_name, "plugin_options": plugin_options}
        self._steps[step_name] = step_info

    def remove_step(self, step_name):
        if step_name in self._steps:
            self._steps.pop(step_name)

    def edit_step(self, step_name, plugin_options:dict=None):
        if step_name in self._steps:
            step = self._steps[step_name]
            if plugin_options is not None:
                current_options = step["plugin_options"]
                for option_name, info in plugin_options.items():
                    current_options[option_name] = info
            else:
                plugin_options = {}

    def add_tracked_output(self, output:str):
        if output in self.tracked_outputs:
            raise RuntimeError(f"output already being tracked: {output}")
        self.tracked_outputs.append(output)

    def remove_tracked_output(self, output:str):
        self.tracked_outputs.remove(output)

    def add_plugin_option(self, step_name:str, option_name:str, option_value):
        if step_name in self._steps:
            step = self._steps[step_name]
            step["plugin_options"][option_name] = option_value

    def remove_plugin_option(self, step_name:str, option_name:str):
        if step_name in self._steps:
            step = self._steps[step_name]
            step["plugin_options"].pop(option_name)