from uam4x.control import ParameterSweep
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Rotor
from uam4x.study import Study, Variants
from uam4x.solution import Solution
from uam4x_ndarc.plugin import NdarcPluginGenerator
from uam4x_openvsp.plugin import VSPGroupGenerator
from conf import (TURBO_ELECTRIC, TURBO_ELECTRIC_AT_SEA,
                ELECTRIC,ELECTRIC_AT_SEA,VEH_VTIP,
                VARIANT_OPTIONS,NDARC_EXE_PATH)
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))



def setup_model(study:Study,variant_options):
    mtr = Vehicle("mtr")
    rotor = Rotor("rotor")
    rotor.set_parameter("radius", 18.22473/2)
    rotor.set_parameter("vtip", 550.)
    mtr.add_component(rotor)
    study.add_vehicle(mtr)


def setup_vtip_tradestudy(study:Study,variant_options):
    # initialize vehicle topology trade study
    vtip_trade = ParameterSweep(
        {
            "vehicles.mtr.rotor.vtip": [450, 500, 550, 600, 650, 700]
        }
    )
    # add passtrade to study
    study.control.add_tradestudy("vtip_trade", vtip_trade)

def setup_solution(study:Study,variant_options):
    # replace ndarc filepath
    # replace exe key-value pair with ndarc executable path
    solution=Solution('basic')
    ndarc_plug=NdarcPluginGenerator(
        name='ndarc',
        solution=solution,
        njob_path=variant_options['njob_path'],
        ndarc_exe=NDARC_EXE_PATH
    )
    for i in range(1,5):
        ndarc_plug.add_input(f"ROTOR({i}).Vtip_ref",var_type=float,default_value=0.0)
        solution.add_link(VEH_VTIP,ndarc_plug.input(f"ROTOR({i}).Vtip_ref"))
    ndarc_plug.add_output("soln.Rotor(1).RADIUS",var_type=float, default_value=0.0,tracked=True)
    ndarc_plug.add_to_study()

    vsp_group=VSPGroupGenerator(name='openvsp',
                            solution=solution,
                            vsp3_path=variant_options['vsp_path'],
                                    )
    analysis_gen=vsp_group.analysis_generator
    update_gen=vsp_group.update_generator
    update_gen.add_float_input(f'parm_id.{variant_options["radius_parm_id"]}')
    analysis_gen.add_float_output('Total_Wet_Area',tracked=True)
    analysis_gen.add_float_output('Total_Wet_Vol',tracked=True)

    solution.add_link(ndarc_plug.output("soln.Rotor(1).RADIUS"),
                      vsp_group.group_input(step_generator=update_gen,
                                            name=f'parm_id.{variant_options["radius_parm_id"]}'))

    study.add_solution(solution=solution)

if __name__ == "__main__":
    func_list = [setup_vtip_tradestudy, setup_solution,setup_model ]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant(TURBO_ELECTRIC,
                          func_list,
                          options=VARIANT_OPTIONS[TURBO_ELECTRIC])
    framework.add_variant(ELECTRIC,
                          func_list,
                          options=VARIANT_OPTIONS[ELECTRIC])
    framework.add_variant(TURBO_ELECTRIC_AT_SEA,
                          func_list,
                          options=VARIANT_OPTIONS[TURBO_ELECTRIC_AT_SEA])
    framework.add_variant(ELECTRIC_AT_SEA,
                          func_list,
                          options=VARIANT_OPTIONS[ELECTRIC_AT_SEA])
    studies = framework.generate_variants()