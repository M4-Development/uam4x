from uam4x_general.plugins import group_step_path
from uam4x.solution import Group
#Standard libraries
import os
import copy
import rcotools.ndarc.ndarc_core as nc
import rcotools.ndarc.ndarc_parse_design as ncdesign
import rcotools.ndarc.ndarc_parse_output as ncout
import rcotools.ndarc.ndarc_parse_perf as ncperf

#uam4x
from uam4x.log import get_log
from uam4x.utils import log, REVERSE_TYPE_MAP, declare_standard_io
from uam4x.custom_errors import NdarcExternalException
from uam4x.solution import Plugin, register_plugin



def is_ndarc_converged(out_file):

        """Check the NDARC output file for case convergence failures."""
        line=True
        in_case_convergence=False
        found_case_convergence=False
        with open(out_file,'r') as f:
            while line:
                line=f.readline()
                if 'Case Convergence' in line:
                    in_case_convergence=True
                    found_case_convergence=True
                if in_case_convergence:
                    if '*** not converged ***' in line or 'NaN' in line:
                        return False
                if line.strip() and set(line.strip())=={'#'}:
                    in_case_convergence=False

        if found_case_convergence:
            return True
        return False

@register_plugin
class NdarcPlugin(Plugin):
    """Run an NDARC job and retrieve configured outputs."""
    def setup_plugin(self, plugin_options:dict):

        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``njob_path``, ``ndarc_exe``, input/output declarations, and optional output-file and derivative settings.

        Returns
        -------
        None
        """
        assert "njob_path" in plugin_options, "NdarcPlugin Requires njob_path option to be set"
        assert "ndarc_exe" in plugin_options, "NdarcPlugin Requires ndarc_exe option to be set"
        assert "output_dict" in plugin_options, "NdarcPlugin requires output_dict to be set"
        assert "input_dict" in plugin_options, "NdarcPlugin Requires input_dict option to be set"

        self.njob_path = plugin_options["njob_path"]
        self.ndarc_exe = plugin_options["ndarc_exe"]
        self.output_dict = plugin_options["output_dict"]
        self.input_dict = plugin_options["input_dict"]
        base_filename = ".".join(os.path.basename(self.njob_path).split(".")[:-1])
        self.log = get_log(log, object_name=f'ndarc plugin: {base_filename}')
        self.log.debug(f'setup')
        self.convergence_strategies=plugin_options.get('convergence_strategies',[])

        #Hash the input ndarc file
        self.declare_template(self.njob_path)

        if 'out_files' in plugin_options:
            self.out_files = plugin_options['out_files'].copy()
        else:
            self.out_files = {}

        self.parse_soln = False
        self.parse_design = False
        self.parse_perf = False

        # check that all the necessary output files are being generated.
        # For example if an output from the solution file is requested the solution file must be generated
        for key in self.output_dict:

            # the first component of the link is the output file the requested output comes from
            outfile = key.split('.')[0]

            # check if the soltuion file is needed
            if outfile == 'soln' and not self.parse_soln:
                self.parse_soln = True
                if self.out_files.get('OUT_solution') == 0:
                    self.log.warning('outputs from solution file were requested but solution file not specified as an NDARC output. \n Adding solution file as an NDARC output. \n')
                self.out_files["OUT_solution"] = 1

            # check if the design file is needed
            elif outfile == 'design' and not self.parse_design:
                self.parse_design = True
                if self.out_files.get('OUT_design') == 0:
                    self.log.warning('outputs from design file were requested but design file not specified as an NDARC output. \n Adding design file as an NDARC output. \n')
                self.out_files["OUT_design"] = 1

            # check if the performance file is needed
            elif outfile == 'perf' and not self.parse_perf:
                self.parse_perf = True
                if self.out_files.get('OUT_perf') == 0:
                    self.log.warning('outputs from performance file were requested but performance file not specified as an NDARC output. \n Adding performance file as an NDARC output. \n')
                self.out_files["OUT_perf"] = 1


        continuous_inputs, continuous_outputs = declare_standard_io(
            self, plugin_options
        )

        if 'step_size' in plugin_options:
            self.declare_partials(continuous_outputs,continuous_inputs, method = 'fd', step=plugin_options['step_size'])
        else:
            self.declare_partials(continuous_outputs,continuous_inputs, method = 'fd')

        if 'partial_pairs' in plugin_options:
            for of,data in plugin_options['partial_pairs'].items():
                if data['wrt']=='*':
                    wrt=continuous_inputs
                else:
                    wrt=data['wrt']
                self.declare_partials(of,wrt,step=data['step_size'],method='fd')


    def run_plugin(self, links:list, work_dir:str):
        """Apply linked NDARC inputs, run the job, and read outputs.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            Working directory for generated files and solver results.

        Returns
        -------
        dict
            Contains the requested NDARC solution, design, or performance values.
        """
        self.log.debug('run')
        # dictionary to hold updates to the ndarc file
        update_dict = {}

        # sample link GroupName.VariableName
        for name, value in links.items():

            # get group name and variable name
            rco_name = name.replace('.','%')

            # turn into dictionary input for rcotools input
            update_dict[rco_name] = value

        # use rcotools to write these updates into the ndarc file
        ndarc_out = self.use_rcotools(update_dict, work_dir)

        # results dictionary
        results = {}

        if self.parse_soln:
            soln = ndarc_out['soln']

        if self.parse_design:
            design = ndarc_out['design']

        if self.parse_perf:
            perf = ndarc_out['perf']

        # search for the output files for the tracked outputs
        for item in self.output_dict:
            outfile, link = item.split('.', 1)
            query = link.replace('.','%')
            output=None

            if outfile == 'soln':
                output = soln.retrieve(query)

            if outfile == 'design':
                output = design.getByAccesstext(query)

            if outfile == 'perf':
                output = perf.getByAccesstext(query)

            if output is not None:
                # add to results dictionary
                results[item] = output
            else:
                err = NdarcExternalException(f'Could not find {link} in NDARC results')
                self.log.error(f'Could not find {link} in NDARC results')
                raise err

        return results

    def use_rcotools(self, update_dict, work_dir):
            # filepaths for ndarc inputs and outputs
            """Apply input updates, execute the external job, and load its results."""
            file_dir = os.path.dirname(self.njob_path)
            base_filename = ".".join(os.path.basename(self.njob_path).split(".")[:-1])
            out_path = os.path.join(work_dir, base_filename + ".out")
            njob_path = os.path.join(work_dir, base_filename + '.njob')

            # update case dictionary with filepaths
            case_mods = {}
            case_mods["FILE_solution"] = os.path.join(work_dir, base_filename + ".soln")
            case_mods["FILE_design"] = os.path.join(work_dir, base_filename + '.design')
            case_mods["FILE_perf"] = os.path.join(work_dir, base_filename + '.perf')
            case_mods["FILE_geometry"] = os.path.join(work_dir, base_filename + '.geom')
            case_mods["FILE_aircraft"] = os.path.join(work_dir, base_filename + '.acd')
            case_mods["FILE_sketch"] = os.path.join(work_dir, base_filename + '.dxf')
            case_mods["FILE_engine"] = os.path.join(work_dir, base_filename + '.eng')
            case_mods["FILE_aero"] = os.path.join(work_dir, base_filename + '.aero')
            case_mods["FILE_error"] = os.path.join(work_dir,  base_filename + '.err')

            # load njob file
            #njob = nc.NdarcJob(self.njob_path, execdir=file_dir, nowarnings=True)
            # update the last cases namelist group with the case modifications then write an updated input file
            valid_out_files = ['OUT_design', 'OUT_perf', 'OUT_geometry', 'OUT_aircraft', 'OUT_solution', 'OUT_sketch', 'OUT_error']
            for key, value in self.out_files.items():
                if key in valid_out_files:
                    case_mods[key] = value
                else:
                    self.log.warning(f'NDARC plugin out file ({key}) specified in plugin_options is not a valid out file. Ignoring {key}.')
            for file in valid_out_files:
                case_mods[file]=1

            converged=False
            convergence_strategies=copy.deepcopy(self.convergence_strategies)
            if not convergence_strategies:
                #just make the list of nonzero length for one iteration
                convergence_strategies=[{}]
            while not converged and convergence_strategies:
                njob=nc.NdarcJob(self.njob_path, execdir=file_dir, nowarnings=True)
                temp_update_dict=copy.deepcopy(update_dict)
                for name, value in convergence_strategies.pop(0).items():
                    temp_update_dict[name]=value
                njob.applyModifications(temp_update_dict)
                njob.updatecaseparams(case_mods)
                njob.write(njob_path)

                # execute njob
                exitonerror = njob.execute(out_path, self.ndarc_exe)
                if not exitonerror:
                    converged=is_ndarc_converged(out_file=out_path)
                else:
                     self.log.error("Error executing ndarc job")
                     raise NdarcExternalException('Error executing NDARC job')
                # read results if there is no error

                if converged:
                    ndarc_out = {}

                    if self.parse_soln:
                        soln = nc.NdarcSolution(case_mods["FILE_solution"], dolazy=False)
                        ndarc_out['soln'] = soln

                    if self.parse_design:
                        design = ncdesign.parse_design_file( case_mods["FILE_design"], verbose=False)
                        ndarc_out['design'] = design

                    if self.parse_perf:
                        perf = ncperf.parse_perf_file(case_mods["FILE_perf"])
                        ndarc_out['perf'] = perf



                #check convergence here
            if converged:
                return ndarc_out
            else:
                self.log.error("Ndarc Job Did Not Converge")
                raise NdarcExternalException("Ndarc Job Did Not Converge")

class NdarcPluginGenerator:

    """Build an NDARC plugin with input, output, and file requests."""
    def __init__(self,name,solution,njob_path,ndarc_exe,group=None):
        """Initialize the instance with its configuration and owning solution."""
        if group is not None and not isinstance(group, Group):
            raise TypeError("group must be a Group instance or None")
        self.group=group
        self.added_to_solution=False
        self.added_to_group=False
        self.name=name
        #uam4x solution
        self.solution=solution

        #ndarc paths
        self.njob_path=njob_path
        self.ndarc_exe=ndarc_exe

        self._input_dict={}
        self._output_dict={}
        self._out_files={}
        self._plugin_options={
            'njob_path':self.njob_path,
            'ndarc_exe':self.ndarc_exe,
            'output_dict':self._output_dict,
            'input_dict':self._input_dict,
            "out_files":self._out_files
                              }
        self._convergence_strategies=[]


    def add_input(self,name,var_type,default_value):
        """Declare an input variable for the generated plugin."""
        self._input_dict[name]={'type':REVERSE_TYPE_MAP[var_type],'default_value':default_value}

    def add_float_input(self, name, default_value=0.0):
        """Declare a floating-point input variable."""
        self.add_input(name, float, default_value)

    def add_int_input(self, name, default_value=1):
        """Declare an integer input variable."""
        self.add_input(name, int, default_value)

    def add_output(self,name,var_type,default_value,tracked=False):
        """Declare an output variable for the generated plugin."""
        self._output_dict[name]={'type':REVERSE_TYPE_MAP[var_type],'default_value':default_value}
        if tracked:
            self.solution.add_tracked_output(group_step_path(self.solution, self.group, f'{self.name}.{name}'))

    def add_float_output(self, name, default_value=0.0, tracked=False):
        """Declare a floating-point output variable."""
        self.add_output(name, float, default_value, tracked=tracked)

    def add_int_output(self, name, default_value=0, tracked=False):
        """Declare an integer output variable."""
        self.add_output(name, int, default_value, tracked=tracked)

    def add_design_output(self,name,var_type,default_value,tracked=False):
        """Declare an output parsed from the NDARC design file."""
        name=f'design.{name}'
        self.add_output(name,var_type,default_value,tracked)

    def add_design_float_output(self, name, default_value=0.0, tracked=False):
        """Declare a floating-point NDARC design output."""
        self.add_design_output(name, float, default_value, tracked=tracked)

    def add_design_int_output(self, name, default_value=0, tracked=False):
        """Declare an integer NDARC design output."""
        self.add_design_output(name, int, default_value, tracked=tracked)

    def add_solution_output(self,name,var_type,default_value,tracked=False):
        """Declare an output parsed from the NDARC solution file."""
        name=f'soln.{name}'
        self.add_output(name,var_type,default_value,tracked)

    def add_solution_float_output(self, name, default_value=0.0, tracked=False):
        """Declare a floating-point NDARC solution output."""
        self.add_solution_output(name, float, default_value, tracked=tracked)

    def add_solution_int_output(self, name, default_value=0, tracked=False):
        """Declare an integer NDARC solution output."""
        self.add_solution_output(name, int, default_value, tracked=tracked)

    def add_performance_output(self,name,var_type,default_value,tracked=False):
        """Declare an output parsed from the NDARC performance file."""
        name=f'perf.{name}'
        self.add_output(name,var_type,default_value,tracked)

    def add_performance_float_output(self, name, default_value=0.0, tracked=False):
        """Declare a floating-point NDARC performance output."""
        self.add_performance_output(name, float, default_value, tracked=tracked)

    def add_performance_int_output(self, name, default_value=0, tracked=False):
        """Declare an integer NDARC performance output."""
        self.add_performance_output(name, int, default_value, tracked=tracked)

    def _check_registration(self, target):
        """Reject duplicate or conflicting step registration."""
        if self.added_to_solution or self.added_to_group:
            raise RuntimeError(f"Plugin generator {self.name!r} is already added")
        if self.name in target.steps:
            raise ValueError(f"Step {self.name!r} already exists in the target container")

    def add_to_study(self):
        """Register the configured plugin as a solution step."""
        if self.group is not None:
            raise ValueError("Generator has a group; call add_to_group()")
        self._check_registration(self.solution)
        self._plugin_options['convergence_strategies']=self._convergence_strategies
        self.solution.add_step(NdarcPlugin, self.name, plugin_options=self._plugin_options)
        self.added_to_solution=True

    def add_to_group(self):
        """Register the configured plugin in its parent group."""
        if self.group is None:
            raise ValueError("Generator has no group; call add_to_study()")
        self._check_registration(self.group)
        self._plugin_options['convergence_strategies']=self._convergence_strategies
        self.group.add_step(NdarcPlugin, self.name, plugin_options=self._plugin_options)
        self.added_to_group=True

    def input(self, name: str) -> str:
        """Return the qualified path for a declared input."""
        if '[' in name:
            test_name=name.split('[')[0]
        else:
            test_name=name
        if test_name not in self._input_dict:
            raise KeyError(
                f"Input {name!r} is not defined for step {self.name!r}"
            )

        return f"{self.name}.{name}"

    def output(self,name:str)->str:
        """Return the qualified path for a declared output."""
        if '[' in name:
            test_name=name.split('[')[0]
        else:
            test_name=name
        if test_name not in self._output_dict:
            raise KeyError(
                f"Output {name!r} is not defined for step {self.name!r}"
            )

        return f"{self.name}.{name}"

    def design_output(self,name:str):
        """Return the qualified path for a declared design output."""
        if '[' in name:
            test_name=name.split('[')[0]
        else:
            test_name=name
        if test_name not in self._output_dict:
            raise KeyError(
                f"Output {name!r} is not defined for step {self.name!r}"
            )

        return f"{self.name}.design.{name}"

    def solution_output(self,name:str):
        """Return the solution-level path for a generated output."""
        if '[' in name:
            test_name=name.split('[')[0]
        else:
            test_name=name
        if test_name not in self._output_dict:
            raise KeyError(
                f"Output {name!r} is not defined for step {self.name!r}"
            )
        return f"{self.name}.soln.{name}"

    def add_convergence_strategy(self, strategy):
        """Append an NDARC convergence strategy."""
        convergence_strategy = {}

        for path, value in strategy.items():
            if "%" not in path:
                raise ValueError(
                    f"Convergence path must be fully qualified: {path!r}"
                )
            convergence_strategy[path] = value

        self._convergence_strategies.append(convergence_strategy)

    def request_solution_file(self):
        """Request the NDARC solution file as an output artifact."""
        self._out_files['OUT_solution']=1

    def request_design_file(self):
        """Request the NDARC design file as an output artifact."""
        self._out_files['OUT_design']=1


    def request_perf_file(self):
        """Request the NDARC performance file as an output artifact."""
        self._out_files['OUT_perf']=1

    def request_geom_file(self):
        """Request the NDARC geometry file as an output artifact."""
        self._out_files['OUT_geometry']=1

    def request_aircraft_file(self):
        """Request the NDARC aircraft file as an output artifact."""
        self._out_files['OUT_aircraft']=1

    def request_sketch_file(self):
        """Request the NDARC sketch file as an output artifact."""
        self._out_files['OUT_sketch']=1

    def request_error_file(self):
        """Request the NDARC error file as an output artifact."""
        self._out_files['OUT_error']=1

    def request_all_output_files(self):
        """Request all supported NDARC output files."""
        self.request_solution_file()
        self.request_design_file()
        self.request_perf_file()
        self.request_geom_file()
        self.request_aircraft_file()
        self.request_sketch_file()
        self.request_error_file()
