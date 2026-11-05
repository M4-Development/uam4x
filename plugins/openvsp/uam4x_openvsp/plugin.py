import openvsp as vsp
from uam4x.solution import Plugin, register_plugin, Solution, Group
import os
from uam4x.log import get_log
from uam4x.utils import declare_standard_io, log, type_string_to_type
import numpy as np

from uam4x.solution import get_plugin_class
from uam4x_openvsp.constants import TWO_INDEX_MESH_TYPES, THREE_INDEX_MESH_TYPES, MESH_TYPES
from uam4x_general.plugins import PluginGenerator
from PIL import Image, ImageDraw,ImageFont
@register_plugin
class VSPUpdatePlugin(Plugin):

    """Update OpenVSP parameters and save the resulting model."""
    def setup_plugin(self, plugin_options:dict=None):
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``base_vsp3_path``, input/output declarations, and screenshot preferences.

        Returns
        -------
        None
        """
        assert "base_vsp3_path" in plugin_options, "OpenvspPlugin Requires base_vsp3_path option to be set"
        assert "output_dict" in plugin_options, "OpenvspPlugin requires output_dict to be set"
        assert "input_dict" in plugin_options, "OpenvspPlugin Requires input_dict option to be set"

        self.base_vsp3_path = plugin_options['base_vsp3_path']
        self.output_dict = plugin_options["output_dict"]
        self.input_dict = plugin_options["input_dict"]
        base_filename = ".".join(os.path.basename(self.base_vsp3_path).split(".")[:-1])
        self.log = get_log(log, object_name=f'openvsp plugin: {base_filename}')
        self.log.debug(f'setup')

        continuous_inputs, continuous_outputs = declare_standard_io(
            self, plugin_options
        )

        # declare template file
        self.declare_template(plugin_options['base_vsp3_path'])


        # screenshotting option:
        self.screenshot_preferences=plugin_options['screenshot_preferences']
        self.screenshot_bool=self.screenshot_preferences['screenshot']

        self.declare_partials(of=continuous_outputs,wrt=continuous_inputs,method='fd')
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(self.base_vsp3_path)

    def run_plugin(self, links:dict, work_dir:str):
        # Setup of other OpenVSP plugins may have cleared the shared VSP session.
        """Update OpenVSP parameters and save the resulting model.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            Working directory for generated files and solver results.

        Returns
        -------
        dict
            Contains the requested outputs from the updated OpenVSP model.
        """
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(self.base_vsp3_path)
        vsp.SetAllViews(vsp.CAM_LEFT_ISO)
        vsp.FitAllViews()

        updated = False
        for name, value in links.items():
            name_split = name.split(".")
            if name_split[0].lower() == "parm_id" or name_split[0].lower() == "parmid":
                parm_id = name_split[-1]
            #handle variables defined for the dynamic watermark
            elif name_split[0].lower()=='wm':
                continue
            else:
                parm_id = self.find_parm_id(name_split)
            valid_id = vsp.ValidParm(parm_id)

            if valid_id:
                vsp.SetParmVal(parm_id, value)
                updated = True
            else:
                if name_split[0].lower() == "parm_id" or name_split[0].lower() == "parmid":
                    self.log.error("Invalid Parm ID (" + name_split[-1] + ")")
                    raise ValueError("Invalid Parm ID (" + name_split[-1] + ")")
                else:
                    self.log.error("Unable to Find Parm ID From Link (" + name + ")")
                    raise ValueError("Unable to Find Parm ID From Link (" + name + ")")


        if self.screenshot_bool:
            #Name of the attribute for text
            #text_attr="_AttrWMText"
            text=''
            if 'watermark_text' in self.screenshot_preferences:
                for name,_ in self.input_dict.items():
                    split_name=name.split('.')
                    if split_name[0].lower()=='wm':

                        text+=f"{self.screenshot_preferences['watermark_text'][name]}: {links[name]}\n"
            save_path=os.path.join(work_dir,self.screenshot_preferences["save_name"])
            screenshot_width=self.screenshot_preferences['width']
            screenshot_height=self.screenshot_preferences['height']
            vsp.ScreenGrab(save_path,w=screenshot_width,h=screenshot_height,transparentBG=False)
            #annotate image:
            img=Image.open(save_path).convert("RGB")
            margin_w=screenshot_width//2
            pad=30
            canvas = Image.new("RGB", (img.width + margin_w, img.height), "white")
            canvas.paste(img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            x0 = img.width + pad
            y0 = pad
            font = ImageFont.truetype("arial.ttf", screenshot_height//40)
            draw.text((x0, y0), text, fill="black",font=font)
            canvas.save(save_path)


        vsp.Update()
        vsp.UpdateGUI()
        vsp.WriteVSPFile(os.path.join(work_dir, os.path.basename(self.base_vsp3_path)))
        results = {}
        results['vsp3_path'] = os.path.join(work_dir, os.path.basename(self.base_vsp3_path))
        if 'update_con' in self.output_dict:
            results['update_con'] = self.output_dict['update_con']['default_value']
        return results

    def find_parm_id(self, name_split):
        """Resolve an OpenVSP parameter name to its parameter ID."""
        parm_location = name_split[0]
        if parm_location.lower() == "geom":
            container_id = vsp.FindGeom(name_split[1], 0)
            parm_id = vsp.FindParm(parm_container_id=container_id, group_name=name_split[2], parm_name=name_split[3])
        elif parm_location.lower() == "userparm" or parm_location.lower() == "user_parm":
            container_id = vsp.FindContainer("UserParms", 0)
            parm_id = vsp.FindParm(parm_container_id=container_id, group_name=name_split[1], parm_name=name_split[2])
        else:
            parm_id = ""
        return parm_id

@register_plugin
class VSPAnalysisPlugin(Plugin):
    """Run an OpenVSP analysis on the updated model and return requested results."""

    def setup_plugin(self, plugin_options):
        """Declare requested results and an input for execution order."""
        self.base_vsp3_path = plugin_options['base_vsp3_path']
        self.base_vsp_filename = os.path.basename(self.base_vsp3_path)
        self.analysis_name = plugin_options['analysis_name']
        self.analysis_options = plugin_options.get('analysis_options', {})
        self.output_dict = plugin_options['output_dict']
        self.parent_update_vsp_name = plugin_options.get('parent_update_vsp_name')
        self.analysis_step_name = plugin_options.get('analysis_step_name')

        self.declare_input('geometry_updated', float, value=0.0)
        for name, spec in self.output_dict.items():
            var_type = type_string_to_type(spec['type'])
            self.declare_output(name, var_type, value=spec.get('default_value'))

    def run_plugin(self, links, work_dir):
        """Read the updated model, execute the analysis, and return declared results."""
        vsp.ClearVSPModel()
        model_path = self.base_vsp3_path
        if self.parent_update_vsp_name is not None:
            step_dir = os.path.basename(os.path.normpath(work_dir))
            if self.analysis_step_name and (
                step_dir == self.analysis_step_name or
                step_dir.endswith('.' + self.analysis_step_name)
            ):
                group_prefix, dot, _ = step_dir.rpartition('.')
                parent_dir = self.parent_update_vsp_name
                if dot:
                    parent_dir = f'{group_prefix}.{parent_dir}'
                model_path = os.path.join(
                    work_dir, '..', parent_dir, self.base_vsp_filename
                )
            else:
                model_path = os.path.join(work_dir, self.base_vsp_filename)
        model_path = os.path.normpath(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'OpenVSP analysis model not found: {model_path}')
        vsp.ReadVSPFile(model_path)

        for name, value in self.analysis_options.items():
            value_type = vsp.GetAnalysisInputType(self.analysis_name, name)
            if value_type == vsp.INT_DATA:
                vsp.SetIntAnalysisInput(self.analysis_name, name, value)
            elif value_type == vsp.DOUBLE_DATA:
                vsp.SetDoubleAnalysisInput(self.analysis_name, name, value)
            elif value_type == vsp.STRING_DATA:
                vsp.SetStringAnalysisInput(self.analysis_name, name, value)
            else:
                raise ValueError(f'Unsupported OpenVSP analysis option: {name}')

        results_id = vsp.ExecAnalysis(self.analysis_name)
        available = set(vsp.GetAllDataNames(results_id))
        results = {}
        for name, spec in self.output_dict.items():
            if name not in available:
                raise ValueError(f'{self.analysis_name} did not return {name!r}')
            result_type = vsp.GetResultsType(results_id, name)
            if result_type == vsp.INT_DATA:
                values = vsp.GetIntResults(results_id, name)
            elif result_type == vsp.DOUBLE_DATA:
                values = vsp.GetDoubleResults(results_id, name)
            elif result_type == vsp.STRING_DATA:
                values = vsp.GetStringResults(results_id, name)
            else:
                raise ValueError(f'Unsupported OpenVSP result type for {name!r}: {result_type}')
            values = list(values)
            declared_type = type_string_to_type(spec['type'])
            if declared_type is np.ndarray:
                results[name] = np.asarray(values)
            else:
                if len(values) != 1:
                    raise ValueError(f'{name!r} has {len(values)} values; declare it as an ndarray')
                results[name] = declared_type(values[0])
        return results

@register_plugin
class VSPMeshPlugin(Plugin):
    """Generate and query OpenVSP meshes."""
    def setup_plugin(self, plugin_options:dict=None):
        """Configure this plugin and declare its UAM4X variables.

        Parameters
        ----------
        plugin_options : dict
            Contains ``base_vsp3_path`` and the configured mesh queries and variable declarations.

        Returns
        -------
        None
        """
        if not isinstance(plugin_options, dict):
            raise ValueError("OpenvspMeshPlugin requires a plugin_options dictionary")
        for key in ("base_vsp3_path", "var_name_to_query"):
            if key not in plugin_options:
                raise ValueError(f"OpenvspMeshPlugin requires option {key!r}")
        queries = plugin_options["var_name_to_query"]
        if not isinstance(queries, dict):
            raise ValueError("OpenvspMeshPlugin option 'var_name_to_query' must be a dictionary")

        # Output names use component_name.query_label; the label is not a parm ID.
        # Keep normalized arrays separate from the caller's serialized options.
        self.var_name_to_query = {}
        for name, data in queries.items():
            if not isinstance(data, dict):
                raise ValueError(f"Mesh query {name!r} must be a dictionary")
            for key in ("surf_index", "mesh_type", "point_query"):
                if key not in data:
                    raise ValueError(f"Mesh query {name!r} requires {key!r}")
            mesh_type = data["mesh_type"]
            if mesh_type not in MESH_TYPES:
                raise ValueError(f"Mesh query {name!r}: invalid mesh type {mesh_type!r}; expected one of {MESH_TYPES}")
            surf_index = data["surf_index"]
            if isinstance(surf_index, (bool, np.bool_)) or not isinstance(surf_index, (int, np.integer)) or surf_index < 0:
                raise ValueError(f"Mesh query {name!r}: surf_index must be a nonnegative integer")
            try:
                point_query = np.asarray(data["point_query"], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Mesh query {name!r}: point_query must be a numeric matrix") from error
            columns = 2 if mesh_type in TWO_INDEX_MESH_TYPES else 3
            if point_query.ndim != 2 or point_query.shape[0] == 0 or point_query.shape[1] != columns:
                raise ValueError(f"Mesh query {name!r}: {mesh_type} point_query must have shape (N, {columns}), with N > 0")
            if not np.isfinite(point_query).all():
                raise ValueError(f"Mesh query {name!r}: point_query must contain finite coordinates")
            num_pts = data.get("num_pts", point_query.shape[0])
            if (
                isinstance(num_pts, (bool, np.bool_))
                or not isinstance(num_pts, (int, np.integer))
                or num_pts != point_query.shape[0]
            ):
                raise ValueError(f"Mesh query {name!r}: num_pts must equal the number of query points ({point_query.shape[0]})")
            self.var_name_to_query[name] = dict(
                data, point_query=point_query, num_pts=int(num_pts)
            )

        # This input establishes execution order with the geometry update step.
        self.declare_input("geometry_updated", float, value=0.0)
        for name, data in self.var_name_to_query.items():
            self.declare_output(
                name, np.ndarray, shape=(data["num_pts"], 3)
            )

        #This allows the user to use the parent update vsp file
        self.parent_update_vsp_name=plugin_options.get("parent_update_vsp_name",None)
        self.mesh_step_name=plugin_options.get("mesh_step_name")
        self.base_vsp3_path=plugin_options['base_vsp3_path']
        self.base_vsp_filename = os.path.basename(self.base_vsp3_path)

        vsp.ClearVSPModel()
        try:
            vsp.ReadVSPFile(plugin_options['base_vsp3_path'])
            self.var_name_to_geom_id = {}
            for name in self.var_name_to_query:
                comp_name = name.split('.')[0]
                geom_id = vsp.FindGeom(comp_name, 0)
                if not geom_id:
                    raise ValueError(f"Mesh query {name!r}: geometry {comp_name!r} was not found")
                self.var_name_to_geom_id[name] = geom_id
        finally:
            vsp.ClearVSPModel()

    def run_plugin(self, links:dict, work_dir:str):
        """Evaluate configured surface mesh point queries.

        Parameters
        ----------
        links : dict
            Input names mapped to values for this evaluation.
        work_dir : str or path-like
            Working directory for generated files and solver results.

        Returns
        -------
        dict
            Contains mesh query names mapped to arrays of XYZ coordinates.
        """
        vsp.ClearVSPModel()

        model_path = self.base_vsp3_path
        if self.parent_update_vsp_name is not None:
            step_dir = os.path.basename(os.path.normpath(work_dir))
            if self.mesh_step_name and (step_dir == self.mesh_step_name or step_dir.endswith('.' + self.mesh_step_name)):
                group_prefix, dot, _ = step_dir.rpartition('.')
                parent_dir = f"{group_prefix}.{self.parent_update_vsp_name}" if dot else self.parent_update_vsp_name
                model_path = os.path.normpath(
                    os.path.join(work_dir, '..', parent_dir, self.base_vsp_filename)
                )
            else:
                # With shared work directories, the update writes into work_dir itself.
                model_path = os.path.join(work_dir, self.base_vsp_filename)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"OpenVSP mesh model not found: {model_path}")
        vsp.ReadVSPFile(model_path)
        results = {}
        for name,data in self.var_name_to_query.items():
            component_name=name.split('.', 1)[0]
            geom_id=vsp.FindGeom(component_name, 0)
            if not geom_id:
                raise ValueError(f"Mesh query {name!r}: geometry {component_name!r} was not found in {model_path}")
            surf_index=data["surf_index"]
            mesh_type=data["mesh_type"]
            point_query=data["point_query"]
            if mesh_type == "UV":
                vec3d_vec = vsp.CompVecPnt01(geom_id, surf_index, point_query[:,0], point_query[:,1])

            elif mesh_type == "RST":
                vec3d_vec = vsp.CompVecPntRST(geom_id, surf_index, point_query[:,0], point_query[:,1], point_query[:,2])

            elif mesh_type == "LMN":
                r_vec, s_vec, t_vec = vsp.ConvertLMNtoRSTVec(
                    geom_id, surf_index,
                    point_query[:, 0], point_query[:, 1], point_query[:, 2]
                )
                vec3d_vec = vsp.CompVecPntRST(geom_id, surf_index, r_vec, s_vec, t_vec)
            elif mesh_type == "EtaV":
                u_vec = np.empty(len(point_query))
                for i, (eta,v) in enumerate(point_query):
                    u = vsp.ConvertEtatoU(geom_id, eta)
                    u_vec[i] = u
                vec3d_vec = vsp.CompVecPnt01(geom_id, surf_index, u_vec, point_query[:,1])
            else:
                raise ValueError(f"Mesh query {name!r}: invalid mesh type {mesh_type!r}")
            results[name] = np.array(vsp.convert_vec3d_array_to_list_matrix(vec3d_vec))
        return results

def vsp_parm_type_to_uam4x_type(vsp_type):
    """Map an OpenVSP parameter type to a UAM4X variable type."""
    if vsp_type == vsp.PARM_INT_TYPE:
        return int
    elif vsp_type == vsp.PARM_DOUBLE_TYPE:
        return float
    else:
        raise ValueError(f"Invalid VSP Parm Type ({vsp_type})")

def vsp_data_type_to_uam4x_type(vsp_type):
    """Map an OpenVSP data type to a UAM4X variable type."""
    if vsp_type == vsp.INT_DATA:
        return int
    elif vsp_type == vsp.DOUBLE_DATA:
        return float
    elif vsp_type == vsp.STRING_DATA:
        return str
    else:
        raise ValueError(f"Invalid VSP Data Type ({vsp_type})")


#Base generators for openvsp:
#Update parm, run analysis, mesh analysis,
#only update parm needs the screenshot capability
class VSPUpdatePluginGenerator(PluginGenerator):
    """Build an OpenVSP update step with screenshot options."""
    plugin_class=VSPUpdatePlugin
    def __init__(self,name,solution,vsp3_path,group=None):
        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name,solution,group=group)

        self.vsp3_path=vsp3_path
        self._screenshot_preferences={
            'screenshot':False,
            'save_name':None,
            'width':2000,
            'height':2000,
            'watermark_text':{}
                                      }
        self._plugin_options['base_vsp3_path']=self.vsp3_path
        self._plugin_options['screenshot_preferences']=self._screenshot_preferences

    def set_screenshot(self,save_name,width=2000,height=2000):
        """Enable an OpenVSP screenshot with the requested dimensions."""
        self._screenshot_preferences['screenshot']=True
        self._screenshot_preferences['save_name']=save_name
        self._screenshot_preferences['width']=width
        self._screenshot_preferences['height']=height

    def add_screenshot_watermark_var(self,var_name,display_name):
        """Add a dynamic value to the screenshot watermark."""
        self._screenshot_preferences['watermark_text'][var_name]=display_name

class VSPAnalysisPluginGenerator(PluginGenerator):
    """Build an OpenVSP analysis step and declare its requested results."""
    plugin_class = VSPAnalysisPlugin

    def __init__(self, name, solution, vsp3_path, analysis_name,
                 analysis_options=None, group=None):
        super().__init__(name, solution, group=group)
        self._plugin_options['base_vsp3_path'] = vsp3_path
        self._plugin_options['analysis_name'] = analysis_name
        self._plugin_options['analysis_options'] = dict(analysis_options or {})
        self._plugin_options['analysis_step_name'] = name
        self.add_float_input('geometry_updated')

    def set_analysis_option(self, name, value):
        """Set an OpenVSP analysis input used on every evaluation."""
        self._plugin_options['analysis_options'][name] = value


class VSPMeshPluginGenerator(PluginGenerator):
    """Build an OpenVSP mesh step and configure mesh queries."""
    plugin_class=VSPMeshPlugin
    def __init__(self,name,solution,vsp3_path,group=None):
        """Initialize the instance with its configuration and owning solution."""
        super().__init__(name,solution,group=group)

        self.vsp3_path=vsp3_path
        self._plugin_options['base_vsp3_path']=vsp3_path
        self._plugin_options['mesh_step_name']=name
        self.input_var_name="geometry_updated"
        self.add_float_input(self.input_var_name)


        self._plugin_options['var_name_to_query']={}


    def add_mesh_query(self,component_name,descriptive_name,mesh_type,point_query,surf_index=0,tracked=True):
        """Declare a mesh point query and its output name."""
        var_name=component_name+'.'+descriptive_name
        if mesh_type in TWO_INDEX_MESH_TYPES and point_query.shape[1]!=2:
            raise ValueError
        if mesh_type in THREE_INDEX_MESH_TYPES and point_query.shape[1]!=3:
            raise ValueError
        if mesh_type not in TWO_INDEX_MESH_TYPES+THREE_INDEX_MESH_TYPES:
            raise ValueError(f'The mesh type bust be in {TWO_INDEX_MESH_TYPES} or {THREE_INDEX_MESH_TYPES}'
                             f'The mesh type {mesh_type} is invalid')

        self.add_output(name=var_name,var_type=np.ndarray,default_value=np.zeros((point_query.shape[0], 3)),tracked=tracked)

        self._plugin_options['var_name_to_query'][var_name]={
            'surf_index':surf_index,
            'mesh_type':mesh_type,
            'point_query':point_query,
            'num_pts':point_query.shape[0]
            }

        return var_name

class VSPGroupGenerator:
    """Create a group for related OpenVSP steps."""
    def __init__(self,name,solution:Solution,
                 vsp3_path,group=None,analysis_name='CompGeom',
                 analysis_options=None):
        """Initialize the instance with its configuration and owning solution."""
        self.name=name
        self.solution=solution
        self.vsp3_path=vsp3_path
        #add groups
        self.group=Group()

        #add the different steps to the group
        if group is not None and not isinstance(group, Group):
            raise TypeError("group must be a Group instance or None")
        (group if group is not None else self.solution).add_step(self.group,name)
        # Update
        self.update_generator=VSPUpdatePluginGenerator(
            name="update",solution=solution,vsp3_path=vsp3_path,group=self.group
        )
        #update connection to define path in
        self.update_generator.add_float_output('update_con',default_value=1.0)
        self.update_generator.add_to_group()
        # Analysis
        self.analysis_generator = VSPAnalysisPluginGenerator(
            name='analysis', solution=solution, vsp3_path=vsp3_path,
            analysis_name=analysis_name, analysis_options=analysis_options,
            group=self.group,
        )
        self.analysis_generator._plugin_options['parent_update_vsp_name'] = self.update_generator.name
        self.analysis_generator.add_to_group()
        self.group.add_link(
            self.update_generator.output('update_con'),
            self.analysis_generator.input('geometry_updated'),
        )

        # Mesh query
        self.mesh_query_generator = VSPMeshPluginGenerator(
            name='mesh_query', solution=solution, vsp3_path=vsp3_path,
            group=self.group,
        )
        self.mesh_query_generator._plugin_options['parent_update_vsp_name'] = self.update_generator.name
        self.mesh_query_generator.add_to_group()
        self.group.add_link(
            self.update_generator.output('update_con'),
            self.mesh_query_generator.input('geometry_updated'),
        )

    def group_input(self, step_generator, name):
        """Return a child step input path within this group."""
        if step_generator.group is not self.group:
            raise ValueError("Generator is not a child of this group")
        return self.name+'.'+step_generator.input(name)

    def group_output(self, step_generator, name):
        """Return a child step output path within this group."""
        if step_generator.group is not self.group:
            raise ValueError("Generator is not a child of this group")
        return self.name+'.'+step_generator.output(name)

    def add_link(self,input_name,output_name):
        self.group.add_link(input_name,output_name)

    #Update parms


    #Mesh query parms