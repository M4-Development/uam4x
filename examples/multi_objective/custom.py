from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x.units import UnitType
from uam4x.custom_errors import PluginException
from uam4x.utils import log
from uam4x.log import get_log
import math

@register_subsystem
class WeightingFactor(Subsystem):
    def setup(self):
        self.log = get_log(log, object_name='component: WeightingFactor')
        self.log.debug('setup')
        self.add_parameter("weighting_factor", 0., float, unit_type=UnitType.LENGTH, description="Wing span")

@register_plugin
class CalcObjective(Plugin):
    def setup_plugin(self, plugin_options):
        self.log = get_log(log, object_name='CalcObjective')
        self.log.debug('setup')
        # declare inputs
        self.declare_input("aspect_ratio", float, 1.0)
        self.declare_input("span", float, 1.0)
        self.declare_input("area", float, 1.0)
        self.declare_input("weighting_factor", float, 1.0)

        # declare outputs
        self.declare_output("chord", float, 1.0)
        self.declare_output("CD", float, 1.0)
        self.declare_output("wing_weight", float, 1.0)
        self.declare_output("objective", float, 1.0)

        # declare partials
        self.declare_partials('*','*',method='exact')
        # constants
        self.e = 0.8 # efficiency
        self.Cd_0 = 0.02 # parasite drag
        self.L = 20000 # lift [lbs]
        self.mach = 0.4 # mach number
        self.a = 343 # speed of sound at sea level [m/s]
        self.v = self.mach * self.a # air speed [m/s]
        self.rho = 1.225 # density at sea level [kg/m^3]
        self.q = (0.5 * self.rho * self.v**2) / 6894.76 # dynamic pressure [psi]
        self.sigma = 40000 # stress allow [psi]
        self.h_box = 15 # height of wing box [in]
        self.rho_mat = 0.1 # material density [psi]

        return

    def run_plugin(self, links, work_dir):
        if links['weighting_factor'] < 0:
            msg = 'plugin failed'
            self.log.error(msg)
            raise PluginException(msg)


        # Calc CD
        CL = self.L / (self.q * links["area"])
        CD_i = CL**2 / (math.pi * self.e * links["aspect_ratio"])
        self.CD = self.Cd_0 + CD_i

        # Calc wing weight
        M_root = self.L * (links["span"] / 4) # root bending moment [lbs * in]
        chord = links["span"] / links["aspect_ratio"] # chord [in]
        F_skin = M_root / self.h_box # force in skin [lbs]
        t_box = F_skin / (self.sigma * chord) # thickness of wing box [in]
        P_box = (2 * chord) + (2 * self.h_box) # perimeter of wing box [in]
        A_box = t_box * (P_box) # area of wing box [in^2]
        V_box = links["span"] * A_box # volume of wing box [in^3]
        W_box = V_box * self.rho_mat # weight of wing box [lbs]

        # Calc objective
        objective = self.CD * links["weighting_factor"] + W_box # objective

        return {"chord": chord, "CD" : self.CD, "wing_weight": W_box, "objective": objective}

    def compute_partials(self, links):
            '''
            Define analytical partials of the objective wrt every
            des var this is a 4 input (weighting factor, area, span,
            AR) 4 output system (chord, objective, wing weight, CD)
            thus there are 16 analytical derivatives
            '''



            # Now do chord=span/AR
            dchord_dweighting_factor = 0.0
            dchord_dspan = 1 / links['aspect_ratio']
            dchord_daspect_ratio = -links['span'] / links['aspect_ratio'] ** 2
            dchord_darea = 0.0

            # Now do Cd
            dCd_dspan = 0.0
            dCd_daspect_ratio = -1 * self.L ** 2 / (self.q ** 2 * links['area'] ** 2 * math.pi * self.e * links['aspect_ratio'] ** 2)
            dCd_darea = -2 * self.L ** 2 / (self.q ** 2 * links['area'] ** 3 * math.pi * self.e * links['aspect_ratio'])
            dCd_dweighting_factor = 0.0
            # Now do W_box

            c1 = self.rho_mat * self.L / (2 * self.h_box * self.sigma)
            dW_box_dspan = c1 * (2 * links["span"] + self.h_box * links["aspect_ratio"])
            dW_box_daspect_ratio = c1 * self.h_box * links["span"]
            dW_box_darea = 0.0
            dW_box_dweighting_factor = 0.0

            # Now do objective
            df_dspan = links['weighting_factor'] * dCd_dspan + dW_box_dspan
            df_daspect_ratio = links['weighting_factor'] * dCd_daspect_ratio + dW_box_daspect_ratio
            df_darea = links['weighting_factor'] * dCd_darea + dW_box_darea
            df_dwf = self.CD

            partial_to_val = {("objective", "weighting_factor") : df_dwf,
            ("objective", "aspect_ratio"): df_daspect_ratio,
            ("objective", "span"): df_dspan,
            ("objective", "area"): df_darea,
            ("CD", "weighting_factor"): dCd_dweighting_factor,
            ("CD", "aspect_ratio"): dCd_daspect_ratio,
            ("CD", "span"): dCd_dspan,
            ("CD", "area"): dCd_darea,
            ("chord", "weighting_factor"): dchord_dweighting_factor,
            ("chord", "aspect_ratio"): dchord_daspect_ratio,
            ("chord", "span"): dchord_dspan,
            ("chord", "area"): dchord_darea,
            ("wing_weight","weighting_factor"): dW_box_dweighting_factor,
            ("wing_weight", "aspect_ratio"): dW_box_daspect_ratio,
            ("wing_weight", "span"): dW_box_dspan,
            ("wing_weight", "area"): dW_box_darea,
            }

            return partial_to_val