from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem,register_subsystem
import numpy as np

@register_subsystem
class Borehole(Subsystem):

    def setup(self):
        self.add_parameter('rw',0.0,float)
        self.add_parameter('r',0.0,float)
        self.add_parameter('Tu',0.0,float)
        self.add_parameter('Hu',0.0,float)
        self.add_parameter('Tl',0.0,float)
        self.add_parameter('Hl',0.0,float)
        self.add_parameter('L',0.0,float)
        self.add_parameter('Kw',0.0,float)

@register_plugin
class BoreholeAnalysis(Plugin):

    def setup_plugin(self,plugin_options):

        self.declare_input('rw',float,0.0)
        self.declare_input('r',float,0.0)
        self.declare_input('Tu',float,0.0)
        self.declare_input('Hu',float,0.0)
        self.declare_input('Tl',float,0.0)
        self.declare_input('Hl',float,0.0)
        self.declare_input('L',float,0.0)
        self.declare_input('Kw',float,0.0)

        self.declare_output('flow_rate',float,0.0)


    def run_plugin(self,links,work_dir):
        
        rw=links['rw']
        r=links['r']
        Tu=links['Tu']
        Hu=links['Hu']
        Tl=links['Tl']
        Hl=links['Hl']
        L=links['L']
        Kw=links['Kw']

        numerator=2*np.pi*Tu*(Hu-Hl)
        denominator=np.log(r/rw)*(1+(2*L*Tu)/(np.log(r/rw)*rw**2*Kw)+Tu/Tl)
        flow_rate=numerator/denominator
        return {'flow_rate':flow_rate}
    