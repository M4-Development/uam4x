import os

from uam4x import Plugin, register_plugin
from rcotools.ixgen.ixgen_core import IxgenJob, IxgenSolution, ixgendefs
from uam4x.log import get_log
from uam4x.utils import declare_standard_io, log
from uam4x.custom_errors import IxgenExternalException

@register_plugin
class IxgenPlugin(Plugin):

    """Run an IXGEN job and retrieve configured results."""
    def setup_plugin(self, plugin_options:dict):

        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``ixgen_path``, ``ixgen_exe``, and input/output declarations.

        Returns
        -------
        None
        """
        assert "ixgen_path" in plugin_options, "IxgenPlugin Requires ixgen_path option to be set"
        assert "ixgen_exe" in plugin_options, "IxgenPlugin Requires ixgen_exe option to be set"
        assert "output_dict" in plugin_options, "IxgenPlugin requires output_dict to be set"
        assert "input_dict" in plugin_options, "IxgenPlugin Requires input_dict option to be set"

        self.ixgen_path = plugin_options["ixgen_path"]
        self.ixgen_exe = plugin_options["ixgen_exe"]
        self.output_dict = plugin_options["output_dict"]
        self.input_dict = plugin_options["input_dict"]
        base_filename = ".".join(os.path.basename(self.ixgen_path).split(".")[:-1])
        self.log = get_log(log, object_name=f'ixgen plugin: {base_filename}')
        self.log.debug(f'setup')

        continuous_inputs, continuous_outputs = declare_standard_io(
            self, plugin_options
        )

        if 'step_size' in plugin_options:
            self.declare_partials(continuous_outputs, continuous_inputs, step=plugin_options['step_size'])
        else:
            self.declare_partials(continuous_outputs, continuous_inputs)


    def run_plugin(self, links, work_dir:str):

        # use rcotools to write updates (contained in links dict) to the new ixgen file
        """Update the IXGEN input file, run IXGEN, and read outputs.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            Working directory for generated files and solver results.

        Returns
        -------
        dict
            Contains the requested IXGEN result values.
        """
        soln = self.use_rcotools(links, work_dir)

        # preallocate results dictionary
        results = {}

        # search in the solution for the tracked outputs
        for query in self.output_dict:

            try:
                # search through output file for specified value
                output = soln.retrieve(query)
                results[query] = output
            except:
                self.log.error(f'Key error in {query}.')
                raise IxgenExternalException(f'Key error in {query}.')

            if output is not None:
                # add to results dictionary
                results[query] = output
            else:
                self.log.error(f'Could not find {query} in IXGEN results')
                raise IxgenExternalException(f'Could not find {query} in IXGEN results')

        return results

    def use_rcotools(self, update_dict, work_dir):

        # output directory
        """Apply input updates, execute the external job, and load its results."""
        base_filename = ".".join(os.path.basename(self.ixgen_path).split(".")[:-1])
        file_dir = os.path.dirname(self.ixgen_path)

        # load input file
        job = IxgenJob(self.ixgen_path,
                       outdir = work_dir,
                       jobname = base_filename,
                       nowarnings = True)

        # Replace missing paths, if any
        job.replaceMissingFilePaths()

        # add updates to input file
        for key, value in update_dict.items():
            job.setByAccesstext(key,value)

        # switch the current directory to locate any reference files (for example airfoil dat files)
        original_cwd = os.getcwd()
        os.chdir(file_dir)

        # execute job
        success = job.execute(IXGENcmd=self.ixgen_exe,
                              outdir=work_dir,
                              nowarnings=True)

        # return current directory to original
        os.chdir(original_cwd)

        # read results if there is no error
        if success:
            soln = IxgenSolution(job.outdir)
        else:
            self.log.error(f'IXGEN failed.')
            raise IxgenExternalException(f'IXGEN failed')


        # return the solution
        return soln
