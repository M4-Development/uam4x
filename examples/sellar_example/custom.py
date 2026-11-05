from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem,register_subsystem
import numpy as np

@register_subsystem
class SellarBase(Subsystem):

    def setup(self):
        self.add_parameter('x',1.0,float)
        self.add_parameter('z',np.array([5.0,2.0]),np.ndarray)


@register_plugin
class SellarDis1(Plugin):

    def setup_plugin(self,plugin_options):

        #global dv
        self.declare_input("z",np.ndarray,np.zeros(2))

        #local dv
        self.declare_input("x",float,0.0)

        #coupling param
        self.declare_input("y2",float,1.0)

        #coupling output
        self.declare_output("y1",float,1.0)

        self.declare_partials('*','*',method='fd')

    def run_plugin(self,links,work_dir):

        z1=links['z'][0]
        z2=links['z'][1]

        x1=links['x']
        y2=links['y2']

        y1=z1**2+z2+x1-0.2*y2

        return {'y1':y1}

@register_plugin
class SellarDis2(Plugin):

    def setup_plugin(self,plugin_options):

        self.declare_input('z',np.ndarray,np.zeros(2))

        self.declare_input('y1',float,1.0)

        self.declare_output('y2',float,1.0)

        self.declare_partials('*','*',method='fd')


    def run_plugin(self,links,work_dir):
        z1=links['z'][0]
        z2=links['z'][1]

        y1=links['y1']

        if y1.real<0.0:
            y1*=-1

        y2=y1**0.5+z1+z2
        return {'y2':y2}

@register_plugin
class SellarObjective(Plugin):

    def setup_plugin(self, plugin_options):
        self.declare_input('z',np.ndarray,np.zeros(2))
        self.declare_input('x',float,0.0)
        self.declare_input('y1',float,1.0)
        self.declare_input('y2',float,1.0)
        self.declare_output('obj',float)
        self.declare_output('con1',float)
        self.declare_output('con2',float)
        self.declare_partials('*','*',method='fd')

    def run_plugin(self, links, work_dir):

        x=links['x']
        y1=links['y1']
        y2=links['y2']
        z1=links['z'][1]

        obj=x**2+z1+y1+np.exp(-y2)
        con1=3.16-y1
        con2=y2-24.0
        return {'obj':obj,
                'con1':con1,
                'con2':con2}