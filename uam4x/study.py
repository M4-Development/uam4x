from uam4x.solution import Solution
from uam4x.control import Control
from uam4x.vehicle import Vehicle
import json
import os
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

    def tracked_outputs(self, tracked_outputs):
        results = {}
        for name, value in self.name_to_value.items():
            if (
                name in self.param_state
                or name in tracked_outputs
                or ('des_var_names' in self.param_state and name in self.param_state['des_var_names'])
            ):
                results[name] = value
        return results

class Study:
    def __init__(self) -> None:
        self.vehicles = {}
        self.solution = Solution()
        self.control = Control()
        self._overwrites = []

    def add_vehicle(self, vehicle) -> None:
        if vehicle.name in self.vehicles:
            raise ValueError(f"Vehicle '{vehicle.name}' already exists in the study.")
        self.vehicles[vehicle.name] = vehicle

    def get_vehicle(self, vehicle_name: str):
        if vehicle_name not in self.vehicles:
            raise KeyError(f"Vehicle '{vehicle_name}' not found in the study.")
        return self.vehicles[vehicle_name]
    def _to_dict(self) -> dict:
        """Convert the study to a dictionary representation."""
        return {
            "control": self.control._to_dict(),
            "vehicles": {k: v._to_dict() for k, v in self.vehicles.items()},
            "solution": self.solution._to_dict(),
            "results": [overwrite.tracked_outputs(self.solution.tracked_outputs) for overwrite in self._overwrites]
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Study':
        study = Study()
        for vname, vdata in data["vehicles"].items():
            study.add_vehicle(Vehicle._from_dict(vdata))
        study.solution = Solution._from_dict(data["solution"])
        study.control = Control._from_dict(data["control"])
        return study

    def save_to_json(self, json_path: str):
        """Save the study to a JSON file."""
        data = self._to_dict()
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, json_path: str):
        """Load the study from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        study = self._from_dict(data)
        self.control = study.control
        self.vehicles = study.vehicles
        self.solution = study.solution


    @classmethod
    def from_json(cls, json_path: str) -> 'Study':
        """Create a Study instance from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

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

    def get_value(self, dotted_path: str):
        """
        Retrieve a value from the study using a dotted path string, e.g.:
        'vehicles.multi_objective_vehicle.wing1.aspect_ratio'
        """
        parts = dotted_path.split('.')
        obj = self
        for part in parts:
            # Try dict access first (for vehicles, parameters, etc.)
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            # Try attribute access
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            # # Try __getitem__ for custom containers
            # elif hasattr(obj, '__getitem__'):
            #     try:
            #         obj = obj[part]
            #     except Exception:
            #         raise AttributeError(f"Cannot access '{part}' in path '{dotted_path}'")
            else:
                raise AttributeError(f"Cannot access '{part}' in path '{dotted_path}'")
        # If it's a Parameter, return its value
        if hasattr(obj, 'value'):
            return obj.value
        return obj
    def _add_overwrite(self, state, name, value):
        temp_over = _Overwrite(state)
        for overwrite in self._overwrites:
            if temp_over == overwrite:
                overwrite.add_item(name, value)
                return
        temp_over.add_item(name, value)
        self._overwrites.append(temp_over)
    def get_plugin_inputs(self, step_name, state):
        inputs = {}
        for (independent, dependent) in self.solution.links:
            dependent_split = dependent.split('.')
            if dependent_split[0] == step_name:
                inputs[".".join(dependent_split[1:])] = self._get_value_with_overwrites(independent, state)
        return inputs
class Variants:
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.variants = {}

    def add_variant(self, name, func_list:list):
        self.variants[name] = func_list

    def generate_variants(self,):
        studies = []
        for name, func_list in self.variants.items():
            variants_dir = os.path.join(self.dir, "variants")
            if not os.path.exists(variants_dir):
                os.mkdir(variants_dir)

            this_variant_dir = os.path.join(variants_dir, name)
            if not os.path.exists(this_variant_dir):
                os.mkdir(this_variant_dir)

            study = Study()
            for func in func_list:
                func(study)
            json_path = os.path.join(this_variant_dir, 'data.json')
            study.save_to_json(json_path)
            studies.append(study)
        return studies

if __name__ == "__main__":
    from uam4x.vehicle import Vehicle
    from uam4x.components import Wing
    from uam4x.solution import Solution
    from uam4x.control import Control, ParameterSweep, Optimization

    # Define a vehicle with 2 wings
    v = Vehicle("TestVehicle")
    left_wing = Wing("left_wing")
    right_wing = Wing("right_wing")
    v.add_component(left_wing)
    v.add_component(right_wing)

    # Define a solution with at least 2 steps
    solution = Solution()

    # Define a control with a sweep and an optimization child
    opt = Optimization(
        des_var_names=["x", "y"],
        des_var_lower=[0.0, 0.0],
        des_var_upper=[10.0, 10.0],
        constraint_names=["c1"],
        constraint_lower=[0.0],
        constraint_upper=[5.0],
        objective_name="obj"
    )
    sweep = ParameterSweep(
        param_name_to_value_list={
            "param1": [1, 2, 3],
            "param2": [10, 20, 30],
        },
        child=opt
    )
    control = Control(title="Demo Study Control", author="UAM", description="Demo control for study")
    control.add_trade("sweep_with_opt", sweep)

    # Add everything to a Study
    study = Study()
    study.add_vehicle(v)
    study.solution = solution
    study.control = control

    # Print summary
    print("Vehicles:", study.vehicles)
    print("Solution steps:", study.solution.steps)
    print("Control trades:", list(study.control.trade_studies.keys()))
    print("Parameter sweep states:")
    for state in sweep.get_states():
        print(state)