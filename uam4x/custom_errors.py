class PluginException(Exception):
    pass

class PluginExternalException(PluginException):
    pass

class NdarcExternalException(PluginExternalException):
    pass

class IxgenExternalException(PluginExternalException):
    pass

class OpenvspExternalException(PluginExternalException):
    pass

class M4ssExternalException(PluginExternalException):
    pass

class OpenvspM4ssExternalException(PluginExternalException):
    pass


