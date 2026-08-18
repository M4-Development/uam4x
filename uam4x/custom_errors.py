class PluginException(Exception):
    """Base exception for plugin execution errors."""

    pass


class PluginExternalException(PluginException):
    """Base exception for plugin errors caused by external tools."""

    pass


class NdarcExternalException(PluginExternalException):
    """Exception raised for NDARC-related external execution errors."""

    pass


class IxgenExternalException(PluginExternalException):
    """Exception raised for IXGEN-related external execution errors."""

    pass


class OpenvspExternalException(PluginExternalException):
    """Exception raised for OpenVSP-related external execution errors."""

    pass


class M4ssExternalException(PluginExternalException):
    """Exception raised for M4SS-related external execution errors."""

    pass


class OpenvspM4ssExternalException(PluginExternalException):
    """Exception raised for combined OpenVSP and M4SS execution errors."""

    pass

class CamradExternalException(PluginExternalException):
    """Exception raised for CAMRADII-related external execution errors."""

    pass
