from uam4x.subsystems import Subsystem, register_subsystem


@register_subsystem
class BeamProblem(Subsystem):
    def setup(self):
        self.add_parameter("width",1.3,float, description="Height of the beam in inches")
        self.add_parameter("height",2.0,float, description="Width of the beam in inches")
        self.add_parameter("axial_load",500.0,float, description="Axial load on the beam in pounds")
        self.add_parameter("bending_moment",1000.0,float, description="Bending moment on the beam in inch-pounds")
        self.add_parameter("yield_strength", 1000., float, description="Yield strength of the beam material in psi")
