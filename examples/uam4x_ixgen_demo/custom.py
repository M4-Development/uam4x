from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x import utils
from uam4x.utils import log
from uam4x.log import get_log
import numpy as np

@register_subsystem
class NdarcTechFactorsIxgenDemo(Subsystem):
    def setup(self):
        self.log = get_log(log, object_name='NdarcTechFactorsIxgenDemo')
        self.log.debug('setup')
        self.add_parameter("Tech1",  1., float)
        self.add_parameter("Tech2",  1., float)
        self.add_parameter("Tech3",  1., float)
        self.add_parameter("Tech4",  1., float)
        self.add_parameter("Tech5",  1., float)
        self.add_parameter("Tech6",  1., float)
        self.add_parameter("Tech7",  1., float)
        self.add_parameter("Tech8",  1., float)
        self.add_parameter("Tech9",  1., float)
        self.add_parameter("Tech10", 1., float)

@register_plugin
class ResidualIxgenDemo(Plugin):
    def setup_plugin(self, plugin_options: dict):
        self.log = get_log(log, object_name='ResidualIxgen')
        self.log.debug('setup')
        assert "input_dict" in plugin_options, "NdarcPlugin Requires input_dict option to be set"
        self.input_dict = plugin_options["input_dict"]

        # declare inputs
        for key in self.input_dict:
            var_type = utils.type_string_to_type(self.input_dict[key]['type'])
            self.declare_input(key, var_type, self.input_dict[key]['default_value'])

        # declare outputs
        self.declare_output("R", float, 0.0)

        # declare partials
        self.declare_partials('*','*')

        self.first = True
        self.scale = 1.
        return

    def run_plugin(self, links: dict, work_dir: str):
        R = 0
        i = 1
        while i < 1000:

            try:
                R += (links[f"W{i}a"] - links[f"W{i}b"])**2
                i += 1
            except KeyError:
                break
        if i == 1:
            raise RuntimeError("Residual plugin found no correctly formatted inputs")
        if self.first:
            self.scale = R/10
            self.first = False
        results = {"R": R/self.scale}

        return results

@register_plugin
class Ft2Meters(Plugin):
    '''
    Convert feet --> meters
    '''
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='Ft2Meters')
        self.log.debug('setup')
        assert "input_dict" in plugin_options, "NdarcPlugin Requires input_dict option to be set"
        self.input_dict = plugin_options["input_dict"]

        # declare inputs and outputs
        for key in self.input_dict:
            value_type  = utils.type_string_to_type(self.input_dict[key]['type'])
            self.declare_input(key, value_type, self.input_dict[key]['default_value'])
            self.declare_output(key + '.meters', value_type , self.input_dict[key]['default_value'])

        # declare partials
        self.declare_partials('*','*')

    def run_plugin(self, links, work_dir):

        #  preallocate dictionary
        results = {}

        # convert from ft to meters
        for key, value in links.items():
            results[key + '.meters'] = value * 0.3048

        return results

@register_plugin
class CalcBladeChord(Plugin):
    '''
    Calculate the blade chord along the span
    '''
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='CalcBladeChord')
        self.log.debug('setup')
        # declare inputs
        self.declare_input("chord", float, 0.0)
        self.declare_input("taper", float, 0.0)

        # declare outputs
        self.declare_output("root_chord",float, 0.0)
        self.declare_output("tip_chord",float, 0.0)

        # declare partials
        self.declare_partials('*','*')


    def run_plugin(self, links, work_dir):
        # chord from ndarc is the chord at 75% span
        # assume linear taper to calculate root and tip chords

        # prealllocate results dictionary
        results = {}

        # calculate root and tip chord
        taper = links['taper']
        chord75 = links['chord']

        root_chord = chord75/(0.75*taper + 0.25)
        tip_chord = taper * root_chord
        results['root_chord'] = root_chord
        results['tip_chord'] = tip_chord

        return results


@register_plugin
class CalcCrossSectionTwist(Plugin):
    '''
    Calculate the twist at each cross section assuming a linear twist
    '''
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='CalcCrossSectionTwist')
        self.log.debug('setup')
        # declare inputs
        self.declare_input("twist", float, 0.0)

        # declare outputs
        self.declare_output("root_twist", float, 0.0)
        self.declare_output("tip_twist", float, 0.0)

        # declare partials
        self.declare_partials('*','*')

        return

    def run_plugin(self, links, work_dir):
        # use linear twist from ndarc to get twist at cross sections

        # prealllocate results dictionary
        results = {}

        # calculate root and tip chord
        twist = links['twist']

        root_twist = 0
        tip_twist = twist
        results['root_twist'] = root_twist
        results['tip_twist'] = tip_twist

        return results

@register_plugin
class kg_m2lbs_ft(Plugin):
    '''
    Convert units from kg/m --> lbs/ft
    '''
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='kg_m2lbs_ft')
        self.log.debug('setup')
        assert "input_dict" in plugin_options, "NdarcPlugin Requires input_dict option to be set"
        self.input_dict = plugin_options["input_dict"]

        # declare inputs and outputs
        for key in self.input_dict:
            value_type  = utils.type_string_to_type(self.input_dict[key]['type'])
            self.declare_input(key, value_type, self.input_dict[key]['default_value'])
            self.declare_output(key + '.lbs_ft', value_type , self.input_dict[key]['default_value'])

        # declare partials
        self.declare_partials('*','*')

        return

    def run_plugin(self, links, work_dir):
        results = {}

        for key, value in links.items():
            results[key + '.lbs_ft'] = value * 2.20462 / 3.28084

        return results

@register_plugin
class CalcTotalMassFromCrossSection(Plugin):
    '''
    Calculate total blade mass from the cross sectional masses from IXGEN
    '''
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='CalcTotalMassFromCrossSection')
        self.log.debug('setup')
        # declare inputs
        self.declare_input("mass_inboard1", float, 0.0)
        self.declare_input("mass_inboard2", float, 0.0)
        self.declare_input("mass_outboard1", float, 0.0)
        self.declare_input("mass_outboard2", float, 0.0)
        self.declare_input("span", float, 0.0)
        self.declare_input("start1", float, 0.0)
        self.declare_input("start2", float, 0.0)
        self.declare_input("stop1", float, 0.0)
        self.declare_input("stop2", float, 0.0)

        # declare outputs
        self.declare_output("total_mass", float, 0.0)

        # declare partials
        self.declare_partials('*','*')

        return

    def run_plugin(self, links, work_dir):
        results = {}

        # mass at cross sections [lb/ft]
        mass_in1 = links['mass_inboard1']
        mass_out1 = links['mass_outboard1']
        mass_in2 = links['mass_inboard2']
        mass_out2 = links['mass_outboard2']

        # radius of rotor blades and section start and stop locations
        span = links['span'] # [ft]
        start1 = links['start1']
        stop1 = links['stop1']
        start2 = links['start2']
        stop2 = links['stop2']

        #  numeric integration variables
        N = 1000
        x1 = np.linspace(start1, stop1, N)
        x2 = np.linspace(start2, stop2, N)

        # mass with given mass in [lb/ft]
        mx1 = mass_in1 + (mass_out1 - mass_in1) * x1
        mass1 =  np.trapezoid(mx1, x1) * span * (stop1 - start1)

        mx2 = mass_in2 + (mass_out2 - mass_in2) * x2
        mass2 =  np.trapezoid(mx2, x2) * span * (stop2 - start2)

        results['total_mass'] = mass1 + mass2

        return results

@register_plugin
class CalcSingleBladeWeightNdarc(Plugin):
    '''
    Calculate the weight of each blade from NDARC
    '''
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='CalcSingleBladeWeightNdarc')
        self.log.debug('setup')
        # declare inputs
        self.declare_input("rotors_weight", float, 0.0)
        self.declare_input("nblade", float, 2.0)

        # declare outputs
        self.declare_output("blade_weight", float, 0.0)

        # declare partials
        self.declare_partials('*','*')

        return

    def run_plugin(self, links, work_dir):
        results = {}

        results['blade_weight'] = links['rotors_weight'] / links['nblade']

        return results

@register_plugin
class FakeFastIxgen(Plugin):
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='FakeFastIxgen')
        self.log.debug('setup')

        # declare inputs
        self.declare_input("region.1.spanwise.start", float, 0.0)
        self.declare_input("region.1.spanwise.stop", float, 0.0)
        self.declare_input("region.1.sweep", float, 0.0)
        self.declare_input("region.2.spanwise.start", float, 0.0)
        self.declare_input("region.2.spanwise.stop", float, 0.0)
        self.declare_input("region.2.sweep", float, 0.0)
        self.declare_input("region.2.mainsparweb.layer_1.thickness", float, 0.0)
        self.declare_input("region.2.mainsparweb.layer_2.thickness", float, 0.0)
        self.declare_input("region.2.mainsparweb.layer_3.thickness", float, 0.0)
        self.declare_input("span", float, 0.0)
        self.declare_input("region.1.chord.start", float, 0.0)
        self.declare_input("region.1.chord.stop", float, 0.0)
        self.declare_input("region.2.chord.start", float, 0.0)
        self.declare_input("region.2.chord.stop", float, 0.0)
        self.declare_input("region.1.twist.start", float, 0.0)
        self.declare_input("region.1.twist.stop", float, 0.0)
        self.declare_input("region.2.twist.start", float, 0.0)
        self.declare_input("region.2.twist.stop", float, 0.0)

        # declare outputs
        self.declare_output('vabs.region.1.inboard.mass', float, 0.0)
        self.declare_output('vabs.region.1.outboard.mass', float, 0.0)
        self.declare_output('vabs.region.2.inboard.mass', float, 0.0)
        self.declare_output('vabs.region.2.outboard.mass', float, 0.0)

        self.declare_partials('*','*')

        return

    def run_plugin(self, links, work_dir):
        results = {'vabs.region.1.inboard.mass': 4.2771162775264875,
                   'vabs.region.1.outboard.mass':4.2771162775264875,
                   'vabs.region.2.inboard.mass': 1.6940357492959117,
                   'vabs.region.2.outboard.mass':1.6104395510357101}

        return results

