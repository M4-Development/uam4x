from uam4x.subsystems import Subsystem, register_subsystem

@register_subsystem
class Ishigami(Subsystem):
    def setup(self):
        self.add_parameter('a', 7.0, float)
        self.add_parameter('b', 0.1, float)
        self.add_parameter('X1', 0.0, float)
        self.add_parameter('X2', 0.0, float)
        self.add_parameter('X3', 0.0, float)
