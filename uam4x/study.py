from uam4x.solution import Solution
from uam4x.control import Control
from uam4x.vehicle import Vehicle, VehicleManager
import json
import os
import copy
from uam4x.log import get_log
from uam4x.utils import convert_numpy_to_python,log, is_numpy_array, parse_index, handle_brackets, replace_none_with_empty_quotes
from operator import  attrgetter
import  sys
import platform
from importlib.metadata import distributions
from pathlib import  Path
import subprocess
import pygit2
from uam4x.results import Results
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

class Study:
    log = get_log(log, object_name='study')
    def __init__(self, log_dir = None) -> None:

        if log_dir:
            if log.open_file:
                log.close_file()
            filename_log = os.path.join(log_dir,'log.txt')
            log.set_filename_log(filename_log=filename_log)
        self.log = Study.log
        self.vehicles = VehicleManager()
        self.solution = Solution()
        self.control = Control()
        self.results = Results(self)
        self._runtime_meta_data = defaultdict(dict)


    def _add_runtime_plugin_checksums(self, trade_name,plugin_to_checksum):
        self._runtime_meta_data[trade_name]['step_plugin_hashes'] = plugin_to_checksum

    def _add_runtime_external_executable_data(self, trade_name,external_executable_to_data):
        self._runtime_meta_data[trade_name]['step_external_executable_data'] = external_executable_to_data

    def _capture_environment(self,trade_name):
        '''
        Fully defines environment
        '''

        machine_dict = {
            'os_system': platform.system(),
            'os_release': platform.release(),
            'os_version': platform.version(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        }

        code = "import sys; print(sys.executable)"
        out = subprocess.check_output(["python", "-c", code]).decode().strip()

        python_dict = { 'python_spec': sys.version,
            "python_executable": str(Path(out))}

        environ_dict = {dist.name: dist.version for dist in distributions()}

        dt=datetime.now(tz=ZoneInfo("America/Los_Angeles"))
        message=f'Executed on'
        keywords=['A','B','d','Y','I','M','S','p']
        for keyword in keywords:
            if not keyword in ['M','S']:
                message+=f' {dt.strftime(f"%{keyword}")}'
            else:
                message+=f':{dt.strftime(f"%{keyword}")}'
            if keyword in ['A','d']:
                message+=','
            if keyword=='Y':
                message+=' at'

        message+=' Pacific Standard Time'

        if trade_name in self._runtime_meta_data:
            self._runtime_meta_data[trade_name]['environment'] = {
            'Machine': machine_dict,
            'Python': python_dict,
            'Packages': environ_dict
            }
            self._runtime_meta_data[trade_name]['time_info']={
                'Start_Time':message
                }
        else:
            execution_env={
                'environment':{
                    'Machine': machine_dict,
                    'Python': python_dict,
                    'Packages': environ_dict
            }}
            self._runtime_meta_data[trade_name]=execution_env
            self._runtime_meta_data[trade_name]['time_info']={
                'Start_Time':message
                }

    def add_trade(self, trade_name:str, trade) -> None:
        self.log.debug(f'add trade: {trade_name}')
        self.control.add_tradestudy(trade_name, trade)

    def add_vehicle(self, vehicle) -> None:
        self.log.debug(f'add vehicle: {vehicle.name}')
        return self.vehicles.add_vehicle(vehicle)

    def get_vehicle(self, vehicle_name: str):
        self.log.debug(f'get vehicle: {vehicle_name}')
        return self.vehicles.get_vehicle(vehicle_name)

    def _to_dict(self) -> dict:
        """Convert the study to a dictionary representation."""
        self.log.debug('write to dict')

        study_dict = {
            "control": convert_numpy_to_python(self.control._to_dict()),
            "vehicles": convert_numpy_to_python({v.name: convert_numpy_to_python(v._to_dict()) for v in self.vehicles}),
            "solution": convert_numpy_to_python(self.solution._to_dict()),
            'results': convert_numpy_to_python(self.results._to_dict()),
            "execution_meta_data": convert_numpy_to_python(self._runtime_meta_data),
        }


        return study_dict

    @classmethod
    def _from_dict(cls, data: dict, work_dir = None) -> 'Study':
        cls.log.debug('study from dict')
        study = cls(log_dir=work_dir)
        for vname, vdata in data["vehicles"].items():
            study.add_vehicle(Vehicle._from_dict(vdata))
        study.solution = Solution._from_dict(data["solution"])
        study.control = Control._from_dict(data["control"])
        study.results = Results._from_dict(data["results"],study)
        study._runtime_meta_data=data["execution_meta_data"]

        return study

    def _save_to_json(self, json_path: str, commit=False,commit_message=''):
        """Save the study to a JSON file."""
        self.log.debug('save to json')
        data = replace_none_with_empty_quotes(self._to_dict())

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        if commit:
            self._commit_json_to_branch(json_path, commit_message)

    def save_model_to_json(self, json_path: str, commit=False):

        msg = f'AutoCommit: Created data for variant {self.control.variant_name}'
        self._save_to_json(json_path, commit=commit, commit_message=msg if commit else None)

    def save_results_to_json(self, json_path: str, commit=False):
        msg = f'AutoCommit: Writing results for variant {self.control.variant_name}'
        self._save_to_json(json_path, commit=commit, commit_message=msg if commit else None)


    def _commit_json_to_branch(self, json_path, commit_message):

        if not os.path.isdir(self.control.repo_root):
            raise NotADirectoryError(f'The provided repository root path {self.control.repo_root} is not a directory.\n'
                             f'The repo root must be a directory.')

        assert os.path.isdir(os.path.join(self.control.repo_root, ".git"))

        try:
            repo_path = pygit2.discover_repository(self.control.repo_root)
        except KeyError:
            repo_path = None

        if repo_path is  None:
            raise ValueError('No repository found with provided path')


        expected_git_dir = os.path.join(self.control.repo_root, '.git')
        repo_path_norm = os.path.normcase(os.path.normpath(os.path.realpath(repo_path)))


        expected_norm = os.path.normcase(os.path.normpath(os.path.realpath(expected_git_dir)))

        if repo_path_norm != expected_norm:
            raise ValueError(f'Provided path: {self.control.repo_root} does not have a repo')

        repo = pygit2.Repository(repo_path)
        if repo.head_is_detached:
            raise RuntimeError("Cannot commit: repo is in detached HEAD state.")
        if repo.head_is_unborn:
            parents = []
        else:
            parents = [repo.head.target]

        #stage the JSON file
        index = repo.index

        # rel path
        rel_path = os.path.relpath(json_path, repo.workdir)
        rel_path = rel_path.replace(os.sep, '/')
        #same as git add path
        index.add(rel_path)
        index.write()

        #this is the snapshot of your repo as a hash
        tree = index.write_tree()

        author = pygit2.Signature(self.control.author, 'fake@email')
        committer = pygit2.Signature(self.control.author, 'fake@email')
        repo.create_commit("HEAD",author, committer, commit_message, tree, parents)

    def load_from_json(self, json_path: str):
        """Load the study from a JSON file."""
        self.log.debug('load from json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        study = self._from_dict(data)
        self.control = study.control
        self.vehicles = study.vehicles
        self.solution = study.solution


    @classmethod
    def from_json(cls, json_path: str) -> 'Study':
        """Create a Study instance from a JSON file."""
        cls.log.debug('create study from json')
        work_dir = os.path.dirname(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data, work_dir)

    #def get_plugin_input_and_output_names(self, step_name):
    #    input_names = []
    #    output_names = []
    #    self.log.debug(f'get plugin input and output names for {step_name}')
    #    for (independent, dependent) in self.solution.links:
    #        # remove indices from index based links
    #        if '[' in  independent:
    #            independent, _ = handle_brackets(independent)
    #        if dependent.split('.')[0] == step_name:
    #            input_names.append(dependent)
    #        if independent.split('.')[0] == step_name:
    #            output_names.append(independent)
    #    for independent in self.solution.tracked_outputs:
    #        if independent.split('.')[0] == step_name:
    #            output_names.append(independent)
    #    return set(input_names), set(output_names)


    def get_value(self, dotted_path: str):
        """
        Retrieve a value from the study using a dotted path string, e.g.:
        'vehicles.multi_objective_vehicle.wing1.aspect_ratio'
        """
        self.log.debug(f'get the value of {dotted_path}')
        inds = None
        if '[' in  dotted_path:
            #Functionality exists that checks for brackets in: step name (plugins)
            # vehicles names, component names, mode names, and parameter names. which [ is not allowed
            # are all not allowed.
            #This handles and has checks for the brackets.
            dotted_path, inds = handle_brackets(dotted_path)

        value = attrgetter(dotted_path)(self)
        if inds is not None:
            value = value[inds]
        return value

    def set_value(self,dotted_path:str,value):
        split_up_path=dotted_path.split('.')

        param=split_up_path[-1]
        comp='.'.join(split_up_path[:-1])
        if len(split_up_path) not in [4,5]:
            #vehicles.vehicle.component.mode.parameter
            #vehicles.vehicle.component.parameter
            raise ValueError(f'Invalid vehicle parameter provided {dotted_path}')

        #Will need to add array based logic back in (indexed strings)
        attrgetter(comp)(self).set_parameter(param,value)

    def get_param_type(self,dotted_path:str):
        split_up_path=dotted_path.split('.')

        param=split_up_path[-1]
        #vehicles.vehicle.component.parameter
        if len(split_up_path)==4:
            comp='.'.join(split_up_path[:-1])
        #vehicles.vehicle.component.mode.parameter
        elif len(split_up_path)==5:
            comp='.'.join(split_up_path[:-2])
        else:
            raise ValueError(f'Invalid vehicle parameter provided {dotted_path}')

        #Will need to add array based logic back in (indexed strings)

        param_type=attrgetter(comp)(self).get_parameter_type(param)

        return param_type

    def get_plugin_link_inputs(self, step_name, state):
        self.log.debug(f'get plugin inputs for {step_name}')
        inputs = {}
        for (independent, dependent) in self.solution.links:
            dependent_split = dependent.split('.')
            if dependent_split[0] == step_name:
                inputs[".".join(dependent_split[1:])] = self._get_value_with_overwrites(independent, state)
        return inputs

    def get_all_names_by_category(self):
        """
        Get all possible vehicle names and step input/output names that are links.

        Returns
        -------
        dict
            Dictionary containing:
            - 'vehicle_names': List of all possible vehicle parameter names (e.g., 'vehicles.veh1.wing1.mode.span')
            - 'step_inputs': List of all step input names (e.g., 'step1.weights.wing_weight')
            - 'step_outputs': List of all step output names (e.g., 'step1.weights.wing_weight')
        """
        vehicle_names = []
        step_inputs = []
        step_outputs = []

        # Get all vehicle parameter names
        for vehicle in self.vehicles:
            vehicle_name = vehicle.name
            for component in vehicle.components:
                # Add baseline parameters
                for param_name in component._parameters.keys():
                    vehicle_names.append(f"vehicles.{vehicle_name}.{component.name}.{param_name}")

                # Add mode parameters
                for mode_name, mode in component._modes.items():
                    for param_name in mode._parameters.keys():
                        vehicle_names.append(f"vehicles.{vehicle_name}.{component.name}.{mode_name}.{param_name}")

        # Get step input and output names from links
        for independent, dependent in self.solution.links:
            # Extract step name from dependent (input)
            dependent_parts = dependent.split('.')
            if len(dependent_parts) > 1:
                step_name = dependent_parts[0]
                if step_name.startswith('step'):
                    step_inputs.append(dependent)

            # Extract step name from independent (output)
            independent_parts = independent.split('.')
            if len(independent_parts) > 1:
                step_name = independent_parts[0]
                if step_name.startswith('step'):
                    step_outputs.append(independent)

        # Add tracked outputs as step outputs
        for tracked_output in self.solution.tracked_outputs:
            output_parts = tracked_output.split('.')
            if len(output_parts) > 1:
                step_name = output_parts[0]
                if step_name.startswith('step'):
                    step_outputs.append(tracked_output)

        # Remove duplicates and sort
        vehicle_names = sorted(list(set(vehicle_names)))
        step_inputs = sorted(list(set(step_inputs)))
        step_outputs = sorted(list(set(step_outputs)))

        return {
            'vehicle_names': vehicle_names,
            'step_inputs': step_inputs,
            'step_outputs': step_outputs
        }

    def get_all_names(self):
        """
        Get all possible vehicle names and step input/output names.

        Returns:
            list: List of all possible vehicle names and step input/output names.
        """
        all_names = self.get_all_names_by_category()
        return all_names['vehicle_names'] + all_names['step_inputs'] + all_names['step_outputs']

class Variants:
    log = get_log(log, object_name='variants')
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.variants = {}
        variants_dir = os.path.join(self.dir,'variants')
        if not os.path.exists(variants_dir):
                os.mkdir(variants_dir)

        self.log = Variants.log

    def add_variant(self, name, func_list:list):
        self.log.debug(f"add variant {name}")
        self.variants[name] = func_list

    def generate_variants(self,commit=False):
        self.log.debug('generate variants')
        studies = []

        for name, func_list in self.variants.items():
            variants_dir = os.path.join(self.dir, "variants")

            if not os.path.exists(variants_dir):
                os.mkdir(variants_dir)

            this_variant_dir = os.path.join(variants_dir, name)
            if not os.path.exists(this_variant_dir):
                os.mkdir(this_variant_dir)

            study = Study(log_dir=variants_dir)
            for func in func_list:
                func(study)

            study.control._set_variant_name(name)
            json_path = os.path.join(this_variant_dir, 'data.json')
            study.save_model_to_json(json_path,commit=commit)
            study._save_to_json(json_path)
            studies.append(study)
        return studies

if __name__ == "__main__":
    from uam4x.vehicle import Vehicle
    from uam4x.subsystems import Wing
    from uam4x.solution import Solution
    from uam4x.control import Control, ParameterSweep, Optimization

    # Define a vehicle with 2 wings
    v = Vehicle("TestVehicle")
    left_wing = Wing("left_wing")
    left_wing.span = 5.
    left_wing.add_mode("happy")
    left_wing.happy.span = 10.
    right_wing = Wing("right_wing")
    v.add_component(left_wing)
    v.add_component(right_wing)

    # Define a solution with at least 2 steps
    solution = Solution()

    # Add some example links and tracked outputs for testing
    solution.add_link("vehicles.TestVehicle.left_wing.span", "step1.inputs.wing_span")
    solution.add_link("vehicles.TestVehicle.right_wing.span", "step1.inputs.wing_span_right")
    solution.add_link("step1.outputs.wing_weight", "step2.inputs.weight")
    solution.add_tracked_output("step1.outputs.wing_weight")
    solution.add_tracked_output("step2.outputs.total_weight")

    #TODO: Update this to use the new Optimization class
    # Define a control with a sweep and an optimization child
    opt = Optimization()
    opt.add_design_var(name='x', lower=0.0, upper=10.0)
    opt.add_design_var(name='y', lower=0.0, upper=10.0)
    opt.add_constraint(name='c1', lower=0.0, upper=5.0)
    opt.add_objective('obj')

    sweep = ParameterSweep(
        param_name_to_value_list={
            "param1": [1, 2, 3],
            "param2": [10, 20, 30],
            },
        child=opt
    )
    control = Control(title="Demo Study Control", author="UAM", description="Demo control for study")
    control.add_tradestudy("sweep_with_opt", sweep)

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

    # Test the new get_all_names method
    print("\n=== Testing get_all_names method ===")
    all_names = study.get_all_names_by_category()
    print("Vehicle names:")
    for name in all_names['vehicle_names']:
        print(f"  {name}")
    print("Step inputs:")
    for name in all_names['step_inputs']:
        print(f"  {name}")
    print("Step outputs:")
    for name in all_names['step_outputs']:
        print(f"  {name}")


    #new tom checks
    s = study
    sv = study.vehicles
    svt = study.vehicles.TestVehicle
    svtl = study.vehicles.TestVehicle.left_wing
    svtls = study.vehicles.TestVehicle.left_wing.span
    svtlhs = study.vehicles.TestVehicle.left_wing.happy.span

    svt1 = sv.TestVehicle
    svtl1 = sv.TestVehicle.left_wing
    svtls1 = sv.TestVehicle.left_wing.span
    svtlhs1 = sv.TestVehicle.left_wing.happy.span

    svtl2 = svt.left_wing
    svtls2 = svt.left_wing.span
    svtlhs2 = svt.left_wing.happy.span

    svtls3 = svtl.span
    svtlhs3 = svtl.happy.span

    assert svt is svt1
    assert svtl is svtl1
    assert svtl is svtl2
    assert svtls is svtls1
    assert svtls is svtls2
    assert svtls is svtls3
    assert svtlhs is svtlhs1
    assert svtlhs is svtlhs2
    assert svtlhs is svtlhs3

    from uam4x.execution_engine import Engine
    engine = Engine(study, "test_work")
    print(study.get_param_type("vehicles.TestVehicle.left_wing.happy.span"))
    print(study.get_param_type("vehicles.TestVehicle.left_wing.span"))



