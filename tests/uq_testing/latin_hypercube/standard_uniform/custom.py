from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem,register_subsystem
import numpy as np

@register_subsystem
class Point(Subsystem):

    def setup(self):
        self.add_parameter('x',0.0,float)

@register_plugin
class PointAnalysis(Plugin):

    def setup_plugin(self,plugin_options):

        self.declare_input('x',float)
        self.declare_output('x_out',float)


    def run_plugin(self,links,work_dir):
        x=links['x']
        return {'x_out':x}

