from uam4x.solution import Solution
from uam4x.control import Control
from uam4x.vehicle import Vehicle, VehicleManager
import json
import os
import copy
from uam4x.log import get_log
from uam4x.utils import (
    convert_numpy_to_python,
    log,
    split_dotted_path_and_indices,
    replace_none_with_empty_quotes,
)
from operator import attrgetter
import sys
import platform
from importlib.metadata import distributions
from pathlib import Path
import subprocess
import pygit2
from uam4x.results import Results
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib


class Study:
    """Represent a UAM4X study and its associated data.

    A study owns the vehicles, solution, control configuration, results, and
    runtime metadata for a UAM4X workflow. It also provides serialization,
    persistence, and dotted-path access helpers for study data.

    Parameters
    ----------
    log_dir : str or path-like, optional
        Directory where the study log file is written. If omitted, the existing
        logging configuration is used.

    Notes
    -----
    The class initializes empty vehicle, solution, control, and result
    containers on construction.

    """

    log = get_log(log, object_name="study")

    def __init__(self, log_dir=None) -> None:
        """Initialize a study.

        Parameters
        ----------
        log_dir : str or path-like, optional
            Directory where the study log file is written. If omitted, the
            existing logging configuration is used.

        Notes
        -----
        Construction initializes empty vehicle, solution, control, result, and
        runtime metadata containers.

        """
        if log_dir:
            if log.open_file:
                log.close_file()
            os.makedirs(log_dir, exist_ok=True)
            filename_log = os.path.join(log_dir, "log.txt")
            log.set_filename_log(filename_log=filename_log)
        self.log = Study.log
        self.vehicles = VehicleManager()
        self.solution = Solution()
        self.control = Control()
        self.results = Results(self)
        self._runtime_meta_data = defaultdict(dict)
        self._model_dict_hash=''
        self._results_dict_hash=''
        self._post_data={}

    def _add_runtime_plugin_checksums(self, trade_name, plugin_to_checksum):
        """Store plugin checksum metadata for a trade execution.

        Parameters
        ----------
        trade_name : str
            Name of the trade study associated with the recorded runtime
            metadata.
        plugin_to_checksum : dict
            Mapping from plugin identifiers to checksum strings captured during
            execution.

        Returns
        -------
        None

        """
        self._runtime_meta_data[trade_name]["step_plugin_hashes"] = plugin_to_checksum

    def _add_runtime_external_executable_data(
        self, trade_name, external_executable_to_data
    ):
        """Store external executable metadata for a trade execution.

        Parameters
        ----------
        trade_name : str
            Name of the trade study associated with the recorded runtime
            metadata.
        external_executable_to_data : dict
            Mapping from external executable identifiers to execution metadata
            captured during the run.

        Returns
        -------
        None

        """
        self._runtime_meta_data[trade_name][
            "step_external_executable_data"
        ] = external_executable_to_data

    def _capture_environment(self, trade_name):
        """Capture runtime environment metadata for a trade execution.

        Parameters
        ----------
        trade_name : str
            Name of the trade study associated with the recorded runtime
            metadata.

        Returns
        -------
        None

        Notes
        -----
        The captured metadata includes machine information, Python interpreter
        details, installed package versions, and a formatted start-time string
        in the America/Los_Angeles time zone.

        """

        machine_dict = {
            "os_system": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        code = "import sys; print(sys.executable)"
        out = subprocess.check_output(["python", "-c", code]).decode().strip()

        python_dict = {"python_spec": sys.version, "python_executable": str(Path(out))}

        environ_dict = {dist.name: dist.version for dist in distributions()}

        dt = datetime.now(tz=ZoneInfo("America/Los_Angeles"))
        message = f"Executed on"
        keywords = ["A", "B", "d", "Y", "I", "M", "S", "p"]
        for keyword in keywords:
            if not keyword in ["M", "S"]:
                message += f' {dt.strftime(f"%{keyword}")}'
            else:
                message += f':{dt.strftime(f"%{keyword}")}'
            if keyword in ["A", "d"]:
                message += ","
            if keyword == "Y":
                message += " at"

        message += " Pacific Time"

        if trade_name in self._runtime_meta_data:
            self._runtime_meta_data[trade_name]["environment"] = {
                "Machine": machine_dict,
                "Python": python_dict,
                "Packages": environ_dict,
            }
            self._runtime_meta_data[trade_name]["time_info"] = {"Start_Time": message}
        else:
            execution_env = {
                "environment": {
                    "Machine": machine_dict,
                    "Python": python_dict,
                    "Packages": environ_dict,
                }
            }
            self._runtime_meta_data[trade_name] = execution_env
            self._runtime_meta_data[trade_name]["time_info"] = {"Start_Time": message}

    def add_trade(self, trade_name: str, trade) -> None:
        """Register a trade study with the study control object.

        Parameters
        ----------
        trade_name : str
            Name used to register the trade study.
        trade : object
            Trade-study workflow object to register with :attr:`control`.

        Returns
        -------
        None

        """
        self.log.debug(f"add trade: {trade_name}")
        self.control.add_tradestudy(trade_name, trade)

    def add_vehicle(self, vehicle) -> None:
        """Add a vehicle definition to the study.

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle object to add to the study vehicle manager.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a vehicle with the same name is already registered.

        """
        self.log.debug(f"add vehicle: {vehicle.name}")
        return self.vehicles.add_vehicle(vehicle)

    def get_vehicle(self, vehicle_name: str):
        """Return a vehicle definition by name.

        Parameters
        ----------
        vehicle_name : str
            Name of the vehicle to retrieve.

        Returns
        -------
        Vehicle
            Vehicle object registered under ``vehicle_name``.

        Raises
        ------
        ValueError
            If no vehicle with name ``vehicle_name`` is registered.

        """
        self.log.debug(f"get vehicle: {vehicle_name}")
        return self.vehicles.get_vehicle(vehicle_name)

    def _to_dict(self) -> dict:
        """Serialize the study to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the study control data,
            vehicles, solution data, results data, and execution metadata.

        """
        self.log.debug("write to dict")

        study_dict = {
            "control": convert_numpy_to_python(self.control._to_dict()),
            "vehicles": convert_numpy_to_python(
                {v.name: convert_numpy_to_python(v._to_dict()) for v in self.vehicles}
            ),
            "solution": convert_numpy_to_python(self.solution._to_dict()),
            "results": convert_numpy_to_python(self.results._to_dict()),
            "execution_meta_data": convert_numpy_to_python(self._runtime_meta_data),
        }

        return study_dict

    @classmethod
    def _from_dict(cls, data: dict, work_dir=None) -> "Study":
        """Reconstruct a study from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized study dictionary, typically produced by :meth:`_to_dict`.
        work_dir : str or path-like, optional
            Directory used for study logging during reconstruction. If omitted,
            the existing logging configuration is used.

        Returns
        -------
        Study
            Reconstructed study containing vehicles, solution, control,
            results, and execution metadata.

        """
        cls.log.debug("study from dict")
        study = cls(log_dir=work_dir)
        for vname, vdata in data["vehicles"].items():
            study.add_vehicle(Vehicle._from_dict(vdata))
        study.solution = Solution._from_dict(data["solution"])
        study.control = Control._from_dict(data["control"])
        if "results" in data:
            study.results = Results._from_dict(data["results"], study)
            study._runtime_meta_data = data["execution_meta_data"]

        return study

    def _save_to_json(self,data, json_path: str, commit=False, commit_message=""):
        """Write serializable data to a JSON file.

        Parameters
        ----------
        data : dict
            JSON-serializable data payload to write.
        json_path : str or path-like
            Output path for the JSON file.
        commit : bool, optional
            If True, commit the written file to the current git branch.
        commit_message : str, optional
            Commit message used when ``commit`` is True.

        Returns
        -------
        None

        """
        self.log.debug("save to json")

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_to_json(self,variant_dir:str,commit=False):
        """Write study model data and, when available, results data to JSON files.

        Parameters
        ----------
        variant_dir : str or path-like
            Variant directory where ``model.json`` and, if results exist,
            ``results.json`` are written.
        commit : bool, optional
            If True, commit the written JSON file or files to the current git
            branch.

        Returns
        -------
        None

        Notes
        -----
        If the current model hash differs from the loaded model hash, only
        ``model.json`` is rewritten and existing results are discarded rather
        than saved.
        """

        #Paths
        model_path=os.path.join(variant_dir,'model.json')
        result_path=os.path.join(variant_dir,'results.json')

        #Full study dictionary
        study_dict=replace_none_with_empty_quotes(self._to_dict())
        model_dict={
            "control":study_dict["control"],
            "vehicles":study_dict["vehicles"],
            "solution":study_dict["solution"],
        }

        model_hash=Study._generate_data_hash(model_dict)
        new_control_dict=Study._update_control(model_dict,model_hash)
        model_dict['control']=new_control_dict

        msg = f"AutoCommit: Writing study to json for variant {self.control.variant_name}"
        if self._model_dict_hash:
            if self._model_dict_hash!=model_hash:
                self._save_to_json(model_dict,model_path)
                print(f'New model does not match loaded model, discarding results and saving')
                if commit:
                    self._commit_json_to_branch(json_paths=[model_path],commit_message=msg)
                return

        self._save_to_json(model_dict,model_path)
        written=[model_path]
        #save the results
        if study_dict['results']:
            results_dict={
                "results":study_dict["results"],
                "execution_meta_data":study_dict["execution_meta_data"]
            }
            results_hash=Study._generate_data_hash(results_dict)
            header_data=Study._generate_header_dict(study_dict["control"],
                                                model_hash,results_hash)
            result_dict={
                "header":header_data,
                'results':study_dict['results'],
                'execution_meta_data':study_dict['execution_meta_data']
                }
            self._save_to_json(result_dict,result_path)
            written.append(result_path)
        if commit:
            self._commit_json_to_branch(written,commit_message=msg)

    @staticmethod
    def _update_control(model_dict,model_hash):
        """Insert the model hash into serialized control data.

        Parameters
        ----------
        model_dict : dict
            Serialized model dictionary containing a ``"control"`` entry.
        model_hash : str
            Hash string to insert into the serialized control data.

        Returns
        -------
        dict
            Updated control dictionary including ``"model_hash"``.

        Notes
        -----
        The method preserves the original control key order and inserts
        ``"model_hash"`` after the fourth existing control entry.

        """
        new_control_dict={}
        for i,(name,data) in enumerate(model_dict["control"].items()):
            new_control_dict[name]=data
            if i==3:
                new_control_dict["model_hash"]=model_hash
        return new_control_dict

    def save_model_to_json(self, variant_dir: str, commit=False):
        """Write model configuration data to ``model.json``.

        Parameters
        ----------
        variant_dir : str or path-like
            Variant directory where ``model.json`` is written.
        commit : bool, optional
            If True, commit the generated JSON file to the current git branch.

        Returns
        -------
        None

        """
        msg = f"AutoCommit: Created data for variant {self.control.variant_name}"
        study_dict=replace_none_with_empty_quotes(self._to_dict())
        model_dict={
            "control":study_dict["control"],
            "vehicles":study_dict["vehicles"],
            "solution":study_dict["solution"],
        }

        model_hash=Study._generate_data_hash(model_dict)
        new_control_dict=Study._update_control(model_dict,model_hash)
        model_dict['control']=new_control_dict

        model_path=os.path.join(variant_dir,'model.json')
        self._save_to_json(model_dict,model_path)
        if commit:
            self._commit_json_to_branch(json_paths=[model_path],commit_message=msg)

    @staticmethod
    def _generate_header_dict(control_dict,model_hash,results_hash):
        """Build header metadata for serialized results or post-processing data.

        Parameters
        ----------
        control_dict : dict
            Serialized control dictionary containing study metadata such as
            title, author, description, repository root, and variant name.
        model_hash : str
            Hash of the serialized model data.
        results_hash : str
            Hash of the serialized results data.

        Returns
        -------
        dict
            Header dictionary containing study metadata together with the model
            and results hashes.

        """
        return {
            "title":control_dict['title'],
            "author":control_dict["author"],
            "description":control_dict["description"],
            "model_hash":model_hash,
            "results_hash":results_hash,
            "repo_root":control_dict["repo_root"],
            "variant_name":control_dict["variant_name"]
        }


    def save_results_to_json(self, variant_path: str, commit=False):
        """Write current study results to ``results.json``.

        Parameters
        ----------
        variant_path : str or path-like
            Variant directory where ``results.json`` is written.
        commit : bool, optional
            If True, commit the generated JSON file to the current git branch.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no results exist, if ``model.json`` does not exist, or if the
            current model hash does not match the loaded model hash.
        AttributeError
            If the study was not loaded from JSON before attempting to save
            results.

        """
        msg = f"AutoCommit: Writing results for variant {self.control.variant_name}"
        results_path=os.path.join(variant_path,'results.json')
        study_dict=replace_none_with_empty_quotes(self._to_dict())
        model_path=os.path.join(variant_path,'model.json')
        if not study_dict['results']:
            raise ValueError(f'Attempted to save results to json but no results exist for study')
        if not os.path.exists(model_path):
            raise ValueError(f'model.json file must exist before saving results to json')

        model_dict={
            "control":study_dict["control"],
            "vehicles":study_dict["vehicles"],
            "solution":study_dict["solution"],
        }
        model_hash= Study._generate_data_hash(model_dict)

        temp_results_dict={
            "results":study_dict["results"],
            "execution_meta_data":study_dict["execution_meta_data"]
        }

        results_hash=Study._generate_data_hash(temp_results_dict)
        header_data=Study._generate_header_dict(study_dict["control"],
                                                model_hash,results_hash)
        results_dict={
            "header":header_data,
            "results":study_dict["results"],
            "execution_meta_data":study_dict["execution_meta_data"]
        }
        if not self._model_dict_hash:
            raise AttributeError(f'Study must be loaded from json before run '
                                 'if results are to be saved')
        else:
            if self._model_dict_hash!=model_hash:
                raise ValueError(f"Study's loaded in hash does not match current model"
                                 "hash.")

        self._save_to_json(results_dict,results_path)

        if commit:
            self._commit_json_to_branch([results_path],commit_message=msg)

    def save_post_to_json(self,variant_path,commit=False):
        """Write post-processing data to ``post.json``.

        Parameters
        ----------
        variant_path : str or path-like
            Variant directory where ``post.json`` is written.
        commit : bool, optional
            If True, commit the generated JSON file to the current git branch.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If results do not exist, if required ``model.json`` or
            ``results.json`` files are missing, or if the current results hash
            does not match the loaded results hash.

        """
        msg=f"AutoCommit: Writing post processing data for variant {self.control.variant_name}"
        post_processing_path=os.path.join(variant_path,'post.json')
        study_dict=replace_none_with_empty_quotes(self._to_dict())
        results_path=os.path.join(variant_path,'results.json')
        #i dont think these checks actually matter because to process the
        #results, you actually need the results already but this
        #is just a safeguard i guess
        if not study_dict['results']:
            raise ValueError(f'Attempted to save post to json but no results exist for study')
        if not os.path.exists(results_path):
            raise ValueError(f'results.json file must exist before saving post to json')
        if not os.path.exists(os.path.join(variant_path,'model.json')):
            raise ValueError(f'model.json file must exist before saving post to json')


        model_dict={
            "control":study_dict["control"],
            "vehicles":study_dict["vehicles"],
            "solution":study_dict["solution"],
        }
        model_hash= Study._generate_data_hash(model_dict)



        result_dict={
            "results":study_dict["results"],
            "execution_meta_data":study_dict["execution_meta_data"]
                     }
        results_hash=Study._generate_data_hash(result_dict)

        if self._results_dict_hash and self._results_dict_hash!=results_hash:
            raise ValueError(f"Study's loaded in hash does not match current results"
                             "hash.")
        header_data=Study._generate_header_dict(study_dict["control"],
                                                model_hash,results_hash)
        header_data["result_hash"]=results_hash
        post_dict={
            "header":header_data,
            "post_storage":self._post_data
                   }
        self._save_to_json(post_dict,post_processing_path)
        if commit:
            self._commit_json_to_branch([post_processing_path],msg)

    def _commit_json_to_branch(self, json_paths, commit_message):
        """Commit generated JSON artifacts to the configured git repository.

        Parameters
        ----------
        json_paths : iterable of str or path-like
            Paths to JSON files to stage and commit.
        commit_message : str
            Git commit message used for the commit.

        Returns
        -------
        None

        Raises
        ------
        NotADirectoryError
            If :attr:`control.repo_root` is not a directory.
        ValueError
            If no git repository can be found at :attr:`control.repo_root`.
        RuntimeError
            If the repository is in a detached HEAD state.

        Notes
        -----
        If staging the requested files produces no tree changes relative to
        HEAD, no commit is created.

        """
        if not os.path.isdir(self.control.repo_root):
            raise NotADirectoryError(
                f"The provided repository root path {self.control.repo_root} is not a directory.\n"
                f"The repo root must be a directory."
            )

        assert os.path.isdir(os.path.join(self.control.repo_root, ".git"))

        try:
            repo_path = pygit2.discover_repository(self.control.repo_root)
        except KeyError:
            repo_path = None

        if repo_path is None:
            raise ValueError("No repository found with provided path")

        expected_git_dir = os.path.join(self.control.repo_root, ".git")
        repo_path_norm = os.path.normcase(os.path.normpath(os.path.realpath(repo_path)))

        expected_norm = os.path.normcase(
            os.path.normpath(os.path.realpath(expected_git_dir))
        )

        if repo_path_norm != expected_norm:
            raise ValueError(
                f"Provided path: {self.control.repo_root} does not have a repo"
            )

        repo = pygit2.Repository(repo_path)
        if repo.head_is_detached:
            raise RuntimeError("Cannot commit: repo is in detached HEAD state.")
        if repo.head_is_unborn:
            parents = []
            unborn=True
        else:
            parents = [repo.head.target]
            unborn=False
        index = repo.index

        for json_path in json_paths:
            # stage the JSON file

            # rel path
            rel_path = os.path.relpath(json_path, repo.workdir)
            rel_path = rel_path.replace(os.sep, "/")
            # same as git add path
            index.add(rel_path)
        index.write()

        # this is the snapshot of your repo as a hash
        tree = index.write_tree()
        if not unborn:
            head_commit=repo[repo.head.target]
            if tree==head_commit.tree_id:
                self.log.debug("No JSON changes detected; skipping commit.")
                print("No JSON changes detected; skipping commit.")
                return


        author = pygit2.Signature(self.control.author, "fake@email")
        committer = pygit2.Signature(self.control.author, "fake@email")
        repo.create_commit("HEAD", author, committer, commit_message, tree, parents)


    @classmethod
    def from_json(cls, variant_dir: str) -> "Study":
        """Create a study from JSON artifacts in a variant directory.

        Parameters
        ----------
        variant_dir : str or path-like
            Variant directory containing ``model.json`` and, optionally,
            ``results.json`` and ``post.json``.

        Returns
        -------
        Study
            Study reconstructed from the JSON artifacts in ``variant_dir``.

        """
        cls.log.debug("create study from json")
        model_path=os.path.join(variant_dir,'model.json')
        with open(model_path, "r") as f:
            model_data = json.load(f)

        loaded_model_hash=model_data["control"]["model_hash"]
        del model_data['control']['model_hash']
        #initially loads in model
        total_dict={
            "control":model_data["control"],
            "vehicles":model_data["vehicles"],
            "solution":model_data["solution"],
        }


        model_hash=Study._generate_data_hash(total_dict)

        if loaded_model_hash!=model_hash:
            log.warning(
            f'Warning, loaded model hash differs from model checksum at save point'
            )

        results_path=os.path.join(variant_dir,'results.json')
        results_hash=None
        if os.path.exists(results_path):
            with open(results_path,'r') as f:
                result_data=json.load(f)

            total_dict["results"]=result_data["results"]
            total_dict["execution_meta_data"]=result_data["execution_meta_data"]
            result_dict={
                "results":result_data["results"],
                "execution_meta_data":result_data["execution_meta_data"]
            }
            results_hash=Study._generate_data_hash(result_dict)

        study=cls._from_dict(total_dict,work_dir=variant_dir)
        study._model_dict_hash=model_hash
        if results_hash:
            study._results_dict_hash=results_hash

        for name in study.control.trade_studies.keys():
            study._post_data[name]={}

        return study


    @staticmethod
    def _generate_data_hash(model_dict):
        """Generate a deterministic hash for serialized study data.

        Parameters
        ----------
        model_dict : dict
            JSON-serializable dictionary to hash.

        Returns
        -------
        str
            SHA-256 hash of the normalized serialized data.

        """
        payload = json.dumps(
        model_dict,
        sort_keys=True,
        separators=(",", ":"),   # remove whitespace for stable bytes
        ).encode("utf-8")
        model_hash=hashlib.sha256(payload).hexdigest()
        return model_hash


    def get_value(self, dotted_path: str):
        """Retrieve a study value from a dotted path.

        Parameters
        ----------
        dotted_path : str
            Dotted attribute path to resolve from the study. Paths may include a
            trailing bracketed index or slice for array-like values, for
            example ``"vehicles.my_vehicle.wing.span[0]"``.

        Returns
        -------
        object
            Value resolved from ``dotted_path``.

        Notes
        -----
        Dotted-path resolution uses :func:`operator.attrgetter`. If bracketed
        indices are present, they are applied after the base attribute path is
        resolved.

        """
        self.log.debug(f"get the value of {dotted_path}")
        inds = None
        if "[" in dotted_path:
            # Functionality exists that checks for brackets in: step name (plugins)
            # vehicles names, component names, mode names, and parameter names. which [ is not allowed
            # are all not allowed.
            # This handles and has checks for the brackets.
            dotted_path, inds = split_dotted_path_and_indices(dotted_path)

        value = attrgetter(dotted_path)(self)
        if inds is not None:
            value = value[inds]
        return value

    def set_value(self, dotted_path: str, value):
        """Set a vehicle parameter value from a dotted path.

        Parameters
        ----------
        dotted_path : str
            Vehicle parameter path in one of these forms:
            ``"vehicles.vehicle.component.parameter"`` or
            ``"vehicles.vehicle.component.mode.parameter"``.
        value : object
            New value to assign.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``dotted_path`` is not a supported vehicle parameter path.

        Notes
        -----
        Indexed paths are not currently supported by this method.

        """
        split_up_path = dotted_path.split(".")

        param = split_up_path[-1]
        comp = ".".join(split_up_path[:-1])
        if len(split_up_path) not in [4, 5]:
            # vehicles.vehicle.component.mode.parameter
            # vehicles.vehicle.component.parameter
            raise ValueError(f"Invalid vehicle parameter provided {dotted_path}")

        # Will need to add array based logic back in (indexed strings)
        attrgetter(comp)(self).set_parameter(param, value)

    def get_param_type(self, dotted_path: str):
        """Return the declared type of a vehicle parameter.

        Parameters
        ----------
        dotted_path : str
            Vehicle parameter path in one of these forms:
            ``"vehicles.vehicle.component.parameter"`` or
            ``"vehicles.vehicle.component.mode.parameter"``.

        Returns
        -------
        type
            Parameter type returned by the underlying component or mode object.

        Raises
        ------
        ValueError
            If ``dotted_path`` is not a supported vehicle parameter path.

        Notes
        -----
        Indexed paths are not currently supported by this method.

        """
        split_up_path = dotted_path.split(".")

        param = split_up_path[-1]
        # vehicles.vehicle.component.parameter
        if len(split_up_path) == 4:
            comp = ".".join(split_up_path[:-1])
        # vehicles.vehicle.component.mode.parameter
        elif len(split_up_path) == 5:
            comp = ".".join(split_up_path[:-2])
        else:
            raise ValueError(f"Invalid vehicle parameter provided {dotted_path}")

        # Will need to add array based logic back in (indexed strings)

        param_type = attrgetter(comp)(self).get_parameter_type(param)

        return param_type

    def get_all_vehicle_param_names(self):
        """Return all available vehicle parameter paths.

        Returns
        -------
        list of str
            Sorted list of dotted vehicle parameter paths, including both
            baseline component parameters and mode-specific parameters.

        Notes
        -----
        Returned paths use the same naming convention accepted by
        :meth:`get_value`, :meth:`set_value`, and :meth:`get_param_type`.

        """
        vehicle_names = []

        # Get all vehicle parameter names
        for vehicle in self.vehicles:
            vehicle_name = vehicle.name
            for component in vehicle.components:
                # Add baseline parameters
                for param_name in component._parameters.keys():
                    vehicle_names.append(
                        f"vehicles.{vehicle_name}.{component.name}.{param_name}"
                    )

                # Add mode parameters
                for mode_name, mode in component._modes.items():
                    for param_name in mode._parameters.keys():
                        vehicle_names.append(
                            f"vehicles.{vehicle_name}.{component.name}.{mode_name}.{param_name}"
                        )

        # Remove duplicates and sort
        vehicle_names = sorted(list(set(vehicle_names)))

        return vehicle_names


class Variants:
    """Manage generation of multiple study variants.

    A ``Variants`` object stores named variant definitions, where each variant
    is described by a sequence of functions that modify a :class:`Study`
    instance before it is saved to a variant directory.

    Parameters
    ----------
    dir : str or path-like
        Repository or project directory where the ``variants`` subdirectory is
        created and variant artifacts are written.

    """

    log = get_log(log, object_name="variants")

    def __init__(self, dir: str) -> None:
        """Initialize a variant manager.

        Parameters
        ----------
        dir : str or path-like
            Repository or project directory where the ``variants``
            subdirectory is created.

        Notes
        -----
        Construction creates the ``variants`` subdirectory if it does not
        already exist.

        """
        self.dir = dir
        self.variants = {}
        variants_dir = os.path.join(self.dir, "variants")
        if not os.path.exists(variants_dir):
            os.mkdir(variants_dir)

        self.log = Variants.log

    def set_repo_root(self, repo_dir: str) -> None:
        """Set the repository root used to generate variant artifacts.

        Parameters
        ----------
        repo_dir : str or path-like
            Path to the repository root. Relative paths are expanded and stored
            as an absolute path.

        Returns
        -------
        None

        """
        self.dir = os.path.abspath(os.path.expanduser(repo_dir))

    def add_variant(self, name, func_list: list,options:dict=None):
        """Register a named study variant.

        Parameters
        ----------
        name : str
            Variant name.
        func_list : list of callable
            Sequence of functions applied to a new :class:`Study` instance to
            configure the variant.
        options : dict, optional
            Optional dictionary passed as the second argument to each function
            in ``func_list``.

        Returns
        -------
        None

        """
        self.log.debug(f"add variant {name}")
        self.variants[name] = (func_list, options)

    def generate_variants(self, commit=False):
        """Generate, save, and return all configured study variants.

        Parameters
        ----------
        commit : bool, optional
            If True, also commit generated JSON artifacts to git.

        Returns
        -------
        list of Study
            Generated study objects, one per configured variant.

        Notes
        -----
        Each generated study is written to ``<repo>/variants/<variant_name>``
        using :meth:`Study.save_model_to_json`.

        """
        self.log.debug("generate variants")
        studies = []

        for name, (func_list, options) in self.variants.items():
            variants_dir = os.path.join(self.dir, "variants")

            if not os.path.exists(variants_dir):
                os.mkdir(variants_dir)

            this_variant_dir = os.path.join(variants_dir, name)
            if not os.path.exists(this_variant_dir):
                os.mkdir(this_variant_dir)

            study = Study(log_dir=variants_dir)
            for func in func_list:
                if options is not None:
                    func(study,options)
                else:
                    func(study)

            study.control._set_repo_root(self.dir)
            study.control._set_variant_name(name)
            study.save_model_to_json(this_variant_dir, commit=commit)
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
    left_wing.span = 5.0
    left_wing.add_mode("happy")
    left_wing.happy.span = 10.0
    right_wing = Wing("right_wing")
    v.add_component(left_wing)
    v.add_component(right_wing)

    # Define a solution with at least 2 steps
    solution = Solution()

    # Add some example links and tracked outputs for testing
    solution.add_link("vehicles.TestVehicle.left_wing.span", "step1.inputs.wing_span")
    solution.add_link(
        "vehicles.TestVehicle.right_wing.span", "step1.inputs.wing_span_right"
    )
    solution.add_link("step1.outputs.wing_weight", "step2.inputs.weight")
    solution.add_tracked_output("step1.outputs.wing_weight")
    solution.add_tracked_output("step2.outputs.total_weight")

    # TODO: Update this to use the new Optimization class
    # Define a control with a sweep and an optimization child
    opt = Optimization()
    opt.add_design_var(name="x", lower=0.0, upper=10.0)
    opt.add_design_var(name="y", lower=0.0, upper=10.0)
    opt.add_constraint(name="c1", lower=0.0, upper=5.0)
    opt.add_objective("obj")

    sweep = ParameterSweep(
        param_name_to_value_list={
            "param1": [1, 2, 3],
            "param2": [10, 20, 30],
        },
        child=opt,
    )
    control = Control(
        title="Demo Study Control", author="UAM", description="Demo control for study"
    )
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
    vehicle_names = study.get_all_vehicle_param_names()
    print("Vehicle names:")
    for name in vehicle_names:
        print(f"  {name}")

    # new tom checks
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
