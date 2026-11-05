from uam4x.solution import Solution
from uam4x.control import ParameterSweep, Optimization
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Rotor, RotorBladeRegion
from uam4x.study import Study, Variants
from custom import *

import os

from uam4x_ndarc.plugin import NdarcPlugin
from uam4x_ixgen.plugin import IxgenPlugin


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# EDIT THESE
NDARC_EXE_PATH = r'C:\Program Files\NDARC_v1_17c_x64\NDARC_v1_17c_x64.exe'
IXGEN_EXE_PATH = r'D:\IXGEN\IXGEN.exe'

# additional varaibles
NJOB_PATH = os.path.join(THIS_DIR, 'input_files', 'ndarc', 'UAM-N-quad-5-turbo-xshaft-coll-flap-550.njob')
IXGEN_INPUT_PATH = os.path.join(THIS_DIR, 'input_files', 'ixgen', 'ATR_steel.ixgen')


def setup_quad_model(study:Study):

    # add vehicle
    quad = Vehicle("quad")

    # add rotors
    rotor1 = Rotor('rotor')
    rotor1.set_parameter("radius", 8.0)

    # add rotor blade regions
    region1 = RotorBladeRegion("region1")
    region1.set_parameter("spanwise_start",0.01)
    region1.set_parameter("spanwise_stop", 0.05)
    region1.set_parameter("twist_start", 0.0)
    region1.set_parameter("twist_stop", 0.0)
    region1.set_parameter("sweep", 0.0)

    region2 = RotorBladeRegion("region2")
    region2.set_parameter("spanwise_start",0.06)
    region2.set_parameter("spanwise_stop", 1.0)
    region2.set_parameter("twist_start", 0.0)
    region2.set_parameter("twist_stop", 0.0)
    region2.set_parameter("sweep", 0.0)
    region2.set_parameter("mainsparweb_thickness", 0.002)

    # add tech factors
    tech_factors = NdarcTechFactorsIxgenDemo("tech")

    # add rotors to vehicle
    quad.add_component(rotor1)
    quad.add_component(region1)
    quad.add_component(region2)
    quad.add_component(tech_factors)

    # add vehicle to study
    study.add_vehicle(quad)


def setup_solution(study:Study):
    """
    sets up the solution sequence for updating IXGEN blade informtion with NDARC outputs to caluclate new mass

    Parameters
    ----------
    study : Study
        The study to add the solution to
    """
    solution = Solution("basic")


    # Define and add the NDARC step
    solution.add_step(
        NdarcPlugin,
        "ndarc",
        {
            "njob_path": NJOB_PATH,
            "ndarc_exe": NDARC_EXE_PATH,
            "step_size": 0.1,
            "out_files": {"OUT_geometry": 0,
                          "OUT_design": 1,
                          "OUT_aircraft": 0,
                          "OUT_perf":0,
                          "OUT_solution": 1},
            "input_dict":
            {
                "Rotor(1).TECH_blade" : {"type": 'float', 'default_value': 0.0},
                "Rotor(2).TECH_blade" : {"type": 'float', 'default_value': 0.0},
                "Rotor(3).TECH_blade" : {"type": 'float', 'default_value': 0.0},
                "Rotor(4).TECH_blade" : {"type": 'float', 'default_value': 0.0},
                "Rotor(1).Vtip_ref" : {"type": 'float', 'default_value': 0.0},
                "Rotor(2).Vtip_ref" : {"type": 'float', 'default_value': 0.0},
                "Rotor(3).Vtip_ref" : {"type": 'float', 'default_value': 0.0},
                "Rotor(4).Vtip_ref" : {"type": 'float', 'default_value': 0.0},
            },
            "output_dict": {
                "soln.Rotor(1).CHORD": {"type": 'float', "default_value": 0.0},
                "soln.Rotor(1).TAPER": {"type": 'float', "default_value": 0.0},
                "soln.Rotor(1).RADIUS": {"type": 'float', "default_value": 0.0},
                "soln.Rotor(1).TWISTL": {"type": 'float', "default_value": 0.0},
                "soln.Rotor(1).WEIGHT.W_ROTOR_BLADE": {"type": 'float', "default_value": 0.0},
                "soln.Rotor(1).NBLADE": {"type": 'float', "default_value": 2.0},
            },
        }
    )
    # ndarc links
    solution.add_link("vehicles.quad.tech.Tech1", "ndarc.Rotor(1).TECH_blade")
    solution.add_link("vehicles.quad.tech.Tech1", "ndarc.Rotor(2).TECH_blade")
    solution.add_link("vehicles.quad.tech.Tech1", "ndarc.Rotor(3).TECH_blade")
    solution.add_link("vehicles.quad.tech.Tech1", "ndarc.Rotor(4).TECH_blade")

    # link rotor tip speed
    solution.add_link("vehicles.quad.rotor.vtip", "ndarc.Rotor(1).Vtip_ref")
    solution.add_link("vehicles.quad.rotor.vtip", "ndarc.Rotor(2).Vtip_ref")
    solution.add_link("vehicles.quad.rotor.vtip", "ndarc.Rotor(3).Vtip_ref")
    solution.add_link("vehicles.quad.rotor.vtip", "ndarc.Rotor(4).Vtip_ref")

    # convert ft --> m
    solution.add_step(
        Ft2Meters,
        'lengthMETERS',
        {
            "input_dict":
            {
                "Rotor(1).radius": {"type":'float', "default_value":0.0},
                "Rotor(1).chord" : {"type": 'float', "default_value": 0.0}
            }

        }
    )
    solution.add_link('ndarc.soln.Rotor(1).RADIUS','lengthMETERS.Rotor(1).radius')
    solution.add_link('ndarc.soln.Rotor(1).CHORD','lengthMETERS.Rotor(1).chord')

    # calculate the root and tip chord
    solution.add_step(CalcBladeChord,'CrossSectionChord', {})
    solution.add_link('lengthMETERS.Rotor(1).chord.meters', 'CrossSectionChord.chord')
    solution.add_link('ndarc.soln.Rotor(1).TAPER', 'CrossSectionChord.taper')

    # calculate the twist at the cross section
    solution.add_step(CalcCrossSectionTwist,'CrossSectionTwist', {})
    solution.add_link('ndarc.soln.Rotor(1).TWISTL', 'CrossSectionTwist.twist')

    # ndarc gives the total weight of all the rotors. get the weight of a single rotor [lbs]
    solution.add_step(CalcSingleBladeWeightNdarc,'Rotor1BladeWeight',{})
    solution.add_link('ndarc.soln.Rotor(1).WEIGHT.W_ROTOR_BLADE', 'Rotor1BladeWeight.rotors_weight')
    solution.add_link('ndarc.soln.Rotor(1).NBLADE', 'Rotor1BladeWeight.nblade')

    ## fake fast ixgen step
    #solution.add_step(
        #FakeFastIxgen,
        #"ixgen",
        #{
            #"input_dict":{},
            #"output_request_dict":{}
        #}
    #)

    # Define and add the IXGEN step
    solution.add_step(
        IxgenPlugin,
        "ixgen",
        {
            "ixgen_path": IXGEN_INPUT_PATH,
            "ixgen_exe": IXGEN_EXE_PATH,
            "step_size": 0.1,
            "input_dict":
            {
                "region.1.spanwise.start": {"type": 'float', "default_value": 0.0},
                "region.1.spanwise.stop": {"type": 'float', "default_value": 0.0},
                "region.1.sweep": {"type": 'float', "default_value": 0.0},
                "region.2.spanwise.start": {"type": 'float', "default_value": 0.0},
                "region.2.spanwise.stop": {"type": 'float', "default_value": 0.0},
                "region.2.sweep": {"type": 'float', "default_value": 0.0},
                "region.2.mainsparweb.layer_1.thickness": {"type": 'float', "default_value": 0.0},
                "region.2.mainsparweb.layer_2.thickness": {"type": 'float', "default_value": 0.0},
                "region.2.mainsparweb.layer_3.thickness": {"type": 'float', "default_value": 0.0},
                "span": {"type": 'float', "default_value": 0.0},
                "region.1.chord.start": {"type": 'float', "default_value": 0.0},
                "region.1.chord.stop": {"type": 'float', "default_value": 0.0},
                "region.2.chord.start": {"type": 'float', "default_value": 0.0},
                "region.2.chord.stop": {"type": 'float', "default_value": 0.0},
                "region.1.twist.start": {"type": 'float', "default_value": 0.0},
                "region.1.twist.stop": {"type": 'float', "default_value": 0.0},
                "region.2.twist.start": {"type": 'float', "default_value": 0.0},
                "region.2.twist.stop": {"type": 'float', "default_value": 0.0},
            },
            "output_dict":
            {
                "vabs.region.1.inboard.mass": {"type": 'float', "default_value": 0.0},
                "vabs.region.1.outboard.mass": {"type": 'float', "default_value": 0.0},
                "vabs.region.2.inboard.mass": {"type": 'float', "default_value": 0.0},
                "vabs.region.2.outboard.mass": {"type": 'float', "default_value": 0.0}
            }
            }
    )

    # ixgen links
    solution.add_link("vehicles.quad.region1.spanwise_start", "ixgen.region.1.spanwise.start")
    solution.add_link("vehicles.quad.region1.spanwise_stop", "ixgen.region.1.spanwise.stop")
    solution.add_link("vehicles.quad.region1.sweep", "ixgen.region.1.sweep")

    solution.add_link("vehicles.quad.region2.spanwise_start", "ixgen.region.2.spanwise.start")
    solution.add_link("vehicles.quad.region2.spanwise_stop", "ixgen.region.2.spanwise.stop")
    solution.add_link("vehicles.quad.region2.sweep", "ixgen.region.2.sweep")
    solution.add_link("vehicles.quad.region2.mainsparweb_thickness", "ixgen.region.2.mainsparweb.layer_1.thickness")
    solution.add_link("vehicles.quad.region2.mainsparweb_thickness", "ixgen.region.2.mainsparweb.layer_2.thickness")
    solution.add_link("vehicles.quad.region2.mainsparweb_thickness", "ixgen.region.2.mainsparweb.layer_3.thickness")

    solution.add_link("lengthMETERS.Rotor(1).radius.meters", "ixgen.span")
    solution.add_link("CrossSectionChord.root_chord", "ixgen.region.1.chord.start")
    solution.add_link("CrossSectionChord.root_chord", "ixgen.region.1.chord.stop")
    solution.add_link("CrossSectionChord.root_chord", "ixgen.region.2.chord.start")
    solution.add_link("CrossSectionChord.tip_chord", "ixgen.region.2.chord.stop")
    solution.add_link("CrossSectionTwist.root_twist", "ixgen.region.1.twist.start")
    solution.add_link("CrossSectionTwist.root_twist", "ixgen.region.1.twist.stop")
    solution.add_link("CrossSectionTwist.root_twist", "ixgen.region.2.twist.start")
    solution.add_link("CrossSectionTwist.tip_twist", "ixgen.region.2.twist.stop")

    # convert from kg (ixgen) --> lbs (ndarc)
    solution.add_step(
        kg_m2lbs_ft,
          'massLBS',
            {
                "input_dict":
                {
                    "region.1.inboard.mass": {"type":'float',"default_value":0.0},
                    "region.2.inboard.mass": {"type":'float',"default_value":0.0},
                    "region.1.outboard.mass": {"type":'float',"default_value":0.0},
                    "region.2.outboard.mass": {"type":'float',"default_value":0.0},
                }
            }
            )
    solution.add_link('ixgen.vabs.region.1.inboard.mass', 'massLBS.region.1.inboard.mass')
    solution.add_link('ixgen.vabs.region.1.outboard.mass', 'massLBS.region.1.outboard.mass')
    solution.add_link('ixgen.vabs.region.2.inboard.mass', 'massLBS.region.2.inboard.mass')
    solution.add_link('ixgen.vabs.region.2.outboard.mass', 'massLBS.region.2.outboard.mass')

    # ixgen gives the mass of each inboard and outboard cross section
    #   - integrate to get the total mass of each section
    #   - assume cross section mass given in mass/unit length
    solution.add_step(CalcTotalMassFromCrossSection,'massTotallbs',{})
    solution.add_link('massLBS.region.1.inboard.mass.lbs_ft', 'massTotallbs.mass_inboard1')
    solution.add_link('massLBS.region.2.inboard.mass.lbs_ft', 'massTotallbs.mass_inboard2')
    solution.add_link('massLBS.region.1.outboard.mass.lbs_ft', 'massTotallbs.mass_outboard1')
    solution.add_link('massLBS.region.2.outboard.mass.lbs_ft', 'massTotallbs.mass_outboard2')
    solution.add_link('ndarc.soln.Rotor(1).RADIUS', 'massTotallbs.span')
    solution.add_link('vehicles.quad.region1.spanwise_start', 'massTotallbs.start1')
    solution.add_link('vehicles.quad.region2.spanwise_start', 'massTotallbs.start2')
    solution.add_link('vehicles.quad.region1.spanwise_stop', 'massTotallbs.stop1')
    solution.add_link('vehicles.quad.region2.spanwise_stop', 'massTotallbs.stop2')

    # calculate residuals
    solution.add_step(
        ResidualIxgenDemo,
          'residual',
            {
                "input_dict":{
                    "W1a": {"type": 'float', "default_value": 0.0},
                    "W1b": {"type": 'float', "default_value": 0.0},
                }
            }
        )
    solution.add_link('massTotallbs.total_mass', 'residual.W1a')
    solution.add_link('Rotor1BladeWeight.blade_weight', 'residual.W1b')

    # add tracked outputs
    solution.add_tracked_output("Rotor1BladeWeight.blade_weight")
    solution.add_tracked_output("residual.R")
    solution.add_tracked_output('massTotallbs.total_mass')
    # solution.add_tracked_output('vehicles.quad.tech.Tech1')
    solution.add_tracked_output('ndarc.soln.Rotor(1).RADIUS')
    study.add_solution(solution)


def setup_tradestudy(study:Study):

# Initialize an optimization
    optimization = Optimization()
    optimization.add_design_var("vehicles.quad.tech.Tech1", lower=0.6, upper=1.0)
    optimization.add_objective("residual.R")

    # define a parameter sweep
    vel_sweep = ParameterSweep(
        # parameters to sweep
        {
            "vehicles.quad.rotor.vtip": [500., 550., 600., 650. ]
        },

        optimization
    )
    # add passtrade to study
    study.control.add_tradestudy("vel_trade", vel_sweep)

if __name__ == "__main__":
    # define variants
    quad_variant = [setup_quad_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('quad', quad_variant)
    studies = framework.generate_variants()
