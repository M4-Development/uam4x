# UnitType as class with string attributes
class UnitType:
    """Define supported unit categories for subsystem parameters."""

    NONE = "none"
    LENGTH = "length"
    ANGLE = "angle"
    MASS = "mass"
    TIME = "time"
    FORCE = "force"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    ENERGY = "energy"
    POWER = "power"
    VOLUME = "volume"
    AREA = "area"
    SPEED = "speed"
    ACCELERATION = "acceleration"


class LengthUnit:
    """Define supported length unit strings."""

    METER = "m"
    CENTIMETER = "cm"
    MILLIMETER = "mm"
    MICROMETER = "um"
    NANOMETER = "nm"
    INCH = "in"
    FOOT = "ft"
    YARD = "yd"
    MILE = "mi"
    KILOMETER = "km"


class AngleUnit:
    """Define supported angle unit strings."""

    DEGREE = "deg"
    RADIAN = "rad"
    GRADIAN = "gon"
    ARCSECOND = "arcsec"
    ARCMINUTE = "arcmin"


class MassUnit:
    """Define supported mass unit strings."""

    KILOGRAM = "kg"
    GRAM = "g"
    MILLIGRAM = "mg"
    MICROGRAM = "ug"
    TONNE = "t"
    POUND = "lb"
    OUNCE = "oz"


class TimeUnit:
    """Define supported time unit strings."""

    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    MINUTE = "min"
    HOUR = "h"
    DAY = "d"


class ForceUnit:
    """Define supported force unit strings."""

    NEWTON = "N"
    KILONEWTON = "kN"
    POUND_FORCE = "lbf"
    DYNE = "dyn"


class TemperatureUnit:
    """Define supported temperature unit strings."""

    KELVIN = "K"
    CELSIUS = "C"
    FAHRENHEIT = "F"
    RANKINE = "R"


class PressureUnit:
    """Define supported pressure unit strings."""

    PASCAL = "Pa"
    KILOPASCAL = "kPa"
    BAR = "bar"
    ATMOSPHERE = "atm"
    TORR = "Torr"
    PSI = "psi"


class EnergyUnit:
    """Define supported energy unit strings."""

    JOULE = "J"
    KILOJOULE = "kJ"
    CALORIE = "cal"
    KILOCALORIE = "kcal"
    WATT_HOUR = "Wh"
    KILOWATT_HOUR = "kWh"
    ELECTRONVOLT = "eV"


class PowerUnit:
    """Define supported power unit strings."""

    WATT = "W"
    KILOWATT = "kW"
    MEGAWATT = "MW"
    HORSEPOWER = "hp"


class VolumeUnit:
    """Define supported volume unit strings."""

    LITER = "L"
    MILLILITER = "mL"
    CUBIC_METER = "m3"
    CUBIC_CENTIMETER = "cm3"
    CUBIC_INCH = "in3"
    CUBIC_FOOT = "ft3"
    GALLON = "gal"
    QUART = "qt"
    PINT = "pt"
    FLUID_OUNCE = "fl_oz"


class AreaUnit:
    """Define supported area unit strings."""

    SQUARE_METER = "m2"
    SQUARE_CENTIMETER = "cm2"
    SQUARE_MILLIMETER = "mm2"
    SQUARE_KILOMETER = "km2"
    SQUARE_INCH = "in2"
    SQUARE_FOOT = "ft2"
    SQUARE_YARD = "yd2"
    SQUARE_MILE = "mi2"


class SpeedUnit:
    """Define supported speed unit strings."""

    METER_PER_SECOND = "m/s"
    KILOMETER_PER_HOUR = "km/h"
    MILE_PER_HOUR = "mph"
    KNOT = "kt"
    FOOT_PER_SECOND = "ft/s"


class AccelerationUnit:
    """Define supported acceleration unit strings."""

    METER_PER_SECOND_SQUARED = "m/s2"
    FOOT_PER_SECOND_SQUARED = "ft/s2"
    G_FORCE = "g"


# Mapping from UnitType to corresponding unit class
UNIT_TYPE_TO_UNIT_CLASS = {
    UnitType.LENGTH: LengthUnit,
    UnitType.ANGLE: AngleUnit,
    UnitType.MASS: MassUnit,
    UnitType.TIME: TimeUnit,
    UnitType.FORCE: ForceUnit,
    UnitType.TEMPERATURE: TemperatureUnit,
    UnitType.PRESSURE: PressureUnit,
    UnitType.ENERGY: EnergyUnit,
    UnitType.POWER: PowerUnit,
    UnitType.VOLUME: VolumeUnit,
    UnitType.AREA: AreaUnit,
    UnitType.SPEED: SpeedUnit,
    UnitType.ACCELERATION: AccelerationUnit,
}
