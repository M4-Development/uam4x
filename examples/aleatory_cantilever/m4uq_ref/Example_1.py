# =======================================================================
# Purpose: Objective function computes deflection of a short column.
#
# Variables:
# ----------
# P   = Axial Force     (input)
# b   = Width of Beam   (input)
# h   = Depth of Beam   (input)
# M   = Bending Moment  (input uncertainty variable)
# Y   = Yield Stress    (input uncertainty variable)
# OV1 = Beam Deflection (output variable)
# =======================================================================

def beamModel(samps):
    b=1.3
    h=2.0
    P=500.0
    M=samps[0]
    Y=samps[1]
    OV1 = 1. - ((4.*M)/(b*Y*(h**2))) - ((P**2)/((b*h*Y)**2))
    return [OV1]