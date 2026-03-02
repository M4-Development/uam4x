import os
import json
import multiprocessing as mp
import numpy as  np
import re
from uam4x.log import FileLogger
from numbers import Real
from operator import  itemgetter, attrgetter
import hashlib
import sys

DEV = True
if hasattr(sys, '_MEIPASS'):  # this indicates we are in the pyinstaller exe
    DEV = False

log = FileLogger(filename_log=None)

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "ndarray": np.ndarray,
}

def is_main_process():
    return mp.current_process().name == 'MainProcess'

def registry_writes_enabled():
    return os.getenv('UAM4X_DISABLE_REGISTRY_WRITES') != '1'

def check_and_get_config(conf_path):
    if not os.path.exists(conf_path):
        if is_main_process() and registry_writes_enabled():
            log.info(f'No config found at {conf_path}. Creating new empty config file.')
            with open(conf_path, 'w') as configfile:
                json.dump({},configfile)
        else:
            return {"SUBSYSTEMS": {}, "PLUGINS": {}}

    with open(conf_path, 'r') as configfile:
        try:
            _config = json.load(configfile)
        except json.decoder.JSONDecodeError:
            log.error(f"Could not load parse json at {conf_path}. Please correct or delete the file.")
            raise IOError(f"Could not load parse json at {conf_path}. Please correct or delete the file.")

    if 'SUBSYSTEMS' not in _config:
        _config['SUBSYSTEMS'] = {}
        if is_main_process() and registry_writes_enabled():
            log.info(f'No SUBSYSTEMS found in config at {conf_path}. Creating empty SUBSYSTEMS section.')
            with open(conf_path, 'w') as configfile:
                json.dump(_config, configfile, indent=4)

    if 'PLUGINS' not in _config:
        _config['PLUGINS'] = {}
        if is_main_process() and registry_writes_enabled() :
            log.info(f'No PLUGINS found in config at {conf_path}. Creating empty PLUGINS section.')
            with open(conf_path, 'w') as configfile:
                json.dump(_config, configfile, indent=4)

    return _config


def check_continuous(val_or_type):
    #Handle values (e.g 1, 2.0, np.array([0.0,0.0,...]),np.array([0,0,...]))
    if is_float(val_or_type):
        return True
    elif is_numpy_array(val_or_type):
        return np.issubdtype(val_or_type.dtype, np.floating)
    #Handle types
    if isinstance(val_or_type, type) and (val_or_type == float or issubclass(val_or_type, (np.floating, np.ndarray))):
        return True

    return False

def check_if_scalar(val_or_type):
    '''
    Checks if a provided value or type is a scalar
    to be used primarily for checking upper and
    lower bounds for constraints and des vars
    '''
    if isinstance(val_or_type, (Real)) and not is_bool(val_or_type):
        return True
    if isinstance(val_or_type, type) and (issubclass(val_or_type, (Real, np.integer, np.floating)) and val_or_type not in (np.bool_,bool)):
        return True

    return False

def type_string_to_type(type_name:str):
    return TYPE_MAP[type_name.lower()]


#def get_param_type(dotted_path: str, study, plugins= None):
#    '''
#    Get a the type of a given parameter
#    assumes general form vehicles. ... .attribute
#    '''
#    if '[' in  dotted_path:
#        dotted_path, _ = handle_brackets(dotted_path)
#    parts = dotted_path.split('.')
#    obj = study
#    if parts[0] == 'vehicles':
#        for part in parts:
#            # Try dict access first (for vehicles, parameters, etc.)
#            if isinstance(obj, dict) and part in obj:
#                obj = obj[part]
#            # Try attribute access
#            elif hasattr(obj, part):
#                if hasattr(obj,'_modes'):
#                    modes = list(obj._modes.keys())
#                else:
#                    modes = []
#                if hasattr(obj, 'get_parameter_type') :
#                        if part not in modes:
#                            return obj.get_parameter_type(part)
#                else:
#                    obj = getattr(obj, part)
#            else:
#                raise AttributeError(f"Cannot access '{part}' in path '{dotted_path}'")
#    elif plugins:
#        plugin_name = dotted_path.split('.')[0]
#        part = '.'.join(dotted_path.split('.')[1:])
#        plugin = plugins[plugin_name]
#        if part in  plugin.plugin_input_names:
#            obj = plugin.plugin_input_names[part]['type']
#        elif part in  plugin.plugin_output_names:
#            obj = plugin.plugin_output_names[part]['type']
#        else:
#            raise KeyError(f'Provided path is not an input/output to the plugin {plugin_name}')
#
#    return obj


def convert_numpy_to_python(value):
    '''
    Converts numpy type instances (np.ndarray,np.floating,np.int)
    to python types for Json serialization
    '''
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [convert_numpy_to_python(x) for x in value.tolist()]
    if isinstance(value, list):
        return [convert_numpy_to_python(x) for x in value]
    if isinstance(value, dict):
        return {k: convert_numpy_to_python(x) for  k, x in value.items()}

    return value


def check_numpy_shape(shape):
    #Integer shapes are allowed
    if is_bool(shape):
        return False
    elif is_integer(shape):
        return shape >= 0
    elif is_tuple(shape):
        for dim in  shape:
            if is_bool(dim):
                return False
            if not is_integer(dim):
                return False
            if dim < 0:
                return False
        return True


    return False

def normalize_shape(shape):
    '''
    Given a shape intended for a numpy array
    if the provided shape is a tuple, then
    the returns original value, if it is an int
    then it returns the canonical (N,) form shown
    by numpy when shape is requested

    NOTE: ASSUMES check_numpy_shape has been run on the shape
    already.
    '''
    if is_tuple(shape):
        return shape

    return (int(shape), )
# Nicer type checkers

def is_bool(value):
    return isinstance(value, (bool, np.bool_))

def is_integer(value):

    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

def is_float(value):
    return isinstance(value, (float, np.floating))

def is_tuple(value):
    return isinstance(value, tuple)

def is_numpy_array(value):
    return isinstance(value, np.ndarray)

#Get type name from a value
def value_to_type_name(value):

    value_type = type(value)
    if hasattr(value_type, '__name__'):
        return value_type.__name__
    elif str(value_type) is  not None:
        return str(value_type)
    else:
        return None


def canonicalize_state(x):
    '''
    Given a state dictionary, sorts for hashing
    '''
    if isinstance(x, dict):
        return {k: canonicalize_state(x[k]) for k in sorted(x.keys())}

    if isinstance(x, list):
        return [canonicalize_state(v) for v in x]

    if isinstance(x, np.ndarray):
        return canonicalize_state(x.tolist())

    if isinstance(x, float):
        return round(x, 12)

    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    return x

def hash_state(state):
    '''
    hashes an input state
    '''
    canon = canonicalize_state(state)

    data = json.dumps(
        canon,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False).encode("utf-8")


    return hashlib.sha256(data).hexdigest()


def is_valid_square_brackets(string):
    '''
    Checks if the string has the proper square brackets
    '''
    stack = []
    for letter in  string:
        if letter == "[":
            stack.append(letter)
        if letter == "]" and not stack:
            raise ValueError(f'Brackets are invalid in indexing string: {string}')
        elif letter == "]":
            stack.pop()
    if not stack:
        return True
    else:
        raise ValueError(f'Brackets are inalid in indexing string: {string}')

def is_valid_index(string: str):
    '''
    Checks list for valid characters and order run after is_valid_square_brackets.

    Note, the string should be the index, not the full variable name.

    examples of valid indexing:
    simple: inds=[2,5,7,11,3], inds=[-2,5,-1,3]
    slicing: inds=[::-1], inds=[4:3:2]
    listed SUBSYSTEMS: inds=[[1,2],[6,7]]
    any combination: inds=[[6,7],3:4:2,4]
    '''

    valid_characters = [',', '-', ':', '[', ']']
    # At most two consectutive colons
    if ':::' in string:
        raise ValueError('At most 2 consecutive colons allowed when specifying indices. The string:\n'
                         f'{string} has invalid indexes.')

    if not (string.startswith('[') and  string.endswith(']')):
        raise ValueError(f'Indexes must be surrounded with [] the string: {string} does not have this')

    for i, letter in  enumerate(string[1:-1]):
        loc = i + 1
        # letter must be in the valid characters or an integer
        if not (letter in valid_characters or letter.isdigit()):
            raise ValueError(f'The string: {string} is not a valid array index')
        # number must come after minus sign
        if letter == '-' and  not string[loc+1].isdigit():
            raise ValueError(f'A minus sign must be proceeded with an integer index. The string: {string} is not valid.')
        # commas must be proceeded by -, : or int or opening bracket
        if letter == ',' and  not (string[loc+1].isdigit() or  string[loc+1] in valid_characters[1:-1]):
            raise ValueError(f'A comma must be proceeded with a colon, integer, or opening bracket. The string: {string}\n'
                             f'is not valid. ')
        # check for leading zeros which occur after commas, colons or open brackets
        if letter == '0' and  string[loc-1] in itemgetter(0, 2, 3)(valid_characters) and  string[loc+1].isdigit():
            raise ValueError(f'Leading zeros are not allowed, the string {string} is not valid. ')
        # handle [
        if letter == '[' and not (string[loc+1].isdigit() or  string[loc+1] == ':'):
            raise ValueError(f'Internal brackets must be proceeded with a colon or an integer slice. The string {string} is not valid')
        # handle ]
        if letter == ']' and loc != len(string)-2 and string[loc+1] != ',':
            raise ValueError(f'Internal closing brackets must be proceeded with commas. The string {string} is not valid')

    return True


def is_simple(string):
    '''
    Run after is_valid_square_brackets to characterize
    the slice type. Check if a declared index is simple (aka just a list
    no slices or fancy indexing)
    '''
    for letter in  string:
        if letter == '[':
            return False
        if letter == ':':
            return False
    return True

def is_slice(string):
    '''
    Ran only when index is grouped,
    so run on: 2:3:4 or [1,2,3] or 1
    '''
    if ':' in  string:
        return True
    return False

def is_fancy(string):
    '''
    Ran only when index is grouped,
    so run on: 2:3:4 or [1,2,3] or 1
    '''
    if '[' in  string and ']' in string:
        if ':' in  string:
            raise ValueError(f'Slicing not allowed in fancy indexing')
        else:
            return True
    return False


def parse_simple(string):
    '''

    '''
    indices = re.findall(r'[-]?\d+', string)
    indices = [int(num) for  num in  indices]
    if len(indices) == 1:
        return indices[0]

    return tuple(indices)

def parse_fancy(string):

    indices = re.findall(r'[-]?\d+', string)
    return [int(num) for  num in  indices]

def parse_slice(string):

    if string.count(':') > 2:
        raise ValueError(f'At most 2 colons for slicing within category {string} in violation')

    s = string.split(':')

    while len(s) < 3:
        s.append('')

    def empty_to_none(ex_string):
        if ex_string == '':
            return None
        return int(ex_string)

    start, stop, step = [empty_to_none(letter) for  letter in  s]

    if step == 0:
        raise ValueError(f'Step size cannot be zero. Slice: {string} in violation')

    return slice(start, stop, step)


def parse_complicated(string):
    '''
    parse complex indexing string
    form is assumed as [...]
    '''
    comma_inds = []
    depth = 0
    for i, letter in  enumerate(string[1:-1]):
        loc = i + 1
        if letter == '[':
            depth +=1
        if letter == ']':
            depth -= 1
        if letter == ',' and depth == 0:
            comma_inds.append(loc)

    lower_list = [1] + [comma_ind + 1 for  comma_ind in  comma_inds]
    upper_list = [comma_ind for comma_ind in comma_inds] + [-1]

    slice_list = []
    for lower, upper in  zip(lower_list, upper_list):
        slice_list.append(slice(lower, upper))

    index_groups = []
    for slice_inds in slice_list:
        index_groups.append(string[slice_inds])

    full_index = []
    for group in  index_groups:
        if is_simple(group):
            full_index.append(parse_simple(group))
        elif is_fancy(group):
            full_index.append(parse_fancy(group))
        elif is_slice(group):
            full_index.append(parse_slice(group))
        else:
            raise ValueError(f'Issue')

    return tuple(full_index)


def parse_index(string):
    '''
    Parse the slices out of a string.
    Acceptable forms:
    1.) N-dimensional simple index call out e.g [1,4,6]
    2.) N-dimensional advanced call out e.g [[1,5,3],[1,2],[8]]
    3.) N-dimensional slicing e.g [1:,2:-1,3].

    Assumes the string passed in is the list of indexes in string form
    '''
    is_valid_square_brackets(string)
    is_valid_index(string)
    #Try first form:
    if is_simple(string[1:-1]):
        return parse_simple(string)
    else:
        return parse_complicated(string)


def handle_brackets(dotted_path):
    bracket_ind = dotted_path.find('[')
    inds = parse_index(dotted_path[bracket_ind:])
    dotted_path = dotted_path[:bracket_ind]

    return dotted_path, inds

