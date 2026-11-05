from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem,register_subsystem

@register_subsystem
class CantileverBase(Subsystem):

    def setup(self):
        self.add_parameter('M',1.0,float)
        self.add_parameter('Y',1.0,float)
        self.add_parameter('T',1.0,float)
        self.add_parameter("b",1.3,float)
        self.add_parameter("h",2.0,float)
        self.add_parameter("P",500.,float)

@register_plugin
class CantiliverDeflection(Plugin):

    def setup_plugin(self,plugin_options):

        #inputs
        self.declare_input("M",float)
        self.declare_input("Y",float)
        self.declare_input("T",float)
        self.declare_input("b",float)
        self.declare_input("h",float)
        self.declare_input("P",float)
        #outputs
        self.declare_output("g",float)

    def run_plugin(self,links,work_dir):

        M=links["M"]
        Y=links["Y"]
        T=links["T"]
        b=links["b"]
        h=links["h"]
        P=links["P"]

        g=1-((4*M)/(b*h**2*Y)-(P/(b*h*Y))**2)*T

        return {'g':g}
