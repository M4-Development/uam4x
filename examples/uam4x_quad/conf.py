import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VSP_FILE_PATH = os.path.join(THIS_DIR, "quad_rotor.vsp3")
VSP_AT_SEA_FILE_PATH = os.path.join(THIS_DIR, "quad_rotor_at_sea.vsp3")
NJOB_PATH = os.path.join(THIS_DIR, r'UAM-N-quad-5-turbo-xshaft-coll-flap-550.njob')
ELECTRIC_NJOB_PATH = os.path.join(THIS_DIR, r'UAM-N-quad-5-batt-xshaft-coll-flap-550.njob')


TURBO_ELECTRIC='turbo_electric'
ELECTRIC='electric'
TURBO_ELECTRIC_AT_SEA='turbo_electric_at_sea'
ELECTRIC_AT_SEA='electric_at_sea'


VARIANT_OPTIONS={
    TURBO_ELECTRIC:{
        'radius_parm_id':'FXNTZVGGXQV',
        'vsp_path':VSP_FILE_PATH,
        'njob_path':NJOB_PATH
    },
    ELECTRIC:{
        'radius_parm_id':'FXNTZVGGXQV',
        'vsp_path':VSP_FILE_PATH,
        'njob_path':ELECTRIC_NJOB_PATH
    },
    TURBO_ELECTRIC_AT_SEA:{
        'radius_parm_id':'VSSEDJJZYAE',
        'vsp_path':VSP_AT_SEA_FILE_PATH,
        'njob_path':NJOB_PATH
    },
    ELECTRIC_AT_SEA:{
        'radius_parm_id':'VSSEDJJZYAE',
        'vsp_path':VSP_AT_SEA_FILE_PATH,
        'njob_path':ELECTRIC_NJOB_PATH
    },
}

VEH_VTIP="vehicles.mtr.rotor.vtip"



# NDARC_EXE_PATH = r'C:\dev\NDARC_v1_17c_x64.exe'
#NDARC_EXE_PATH=r'E:\NASA_UAM\NDARC_v1_17c_x64\NDARC_v1_17c_x64.exe'
NDARC_EXE_PATH = r'C:\Program Files\NDARC_v1_17c_x64\NDARC_v1_17c_x64.exe'

