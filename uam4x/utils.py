from dataclasses import replace
import os
import json
import multiprocessing as mp
import numpy as np
import re
from uam4x.log import FileLogger
from numbers import Real
from operator import itemgetter, attrgetter
import hashlib
import sys
import importlib.util
from pathlib import Path

DEV = True
if hasattr(sys, "_MEIPASS"):  # this indicates we are in the pyinstaller exe
    DEV = False

log = FileLogger(filename_log=None)

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "ndarray": np.ndarray,
}

REVERSE_TYPE_MAP={
    type_:name for name,type_ in TYPE_MAP.items()
}

def is_main_process():
    """Return whether the current process is the Python main process.

    Returns
    -------
    bool
        True if the current multiprocessing process name is ``"MainProcess"``,
        otherwise False.

    """
    return mp.current_process().name == "MainProcess"


def registry_writes_enabled():
    """Return whether runtime registry writes are enabled.

    Returns
    -------
    bool
        True unless the environment variable
        ``UAM4X_DISABLE_REGISTRY_WRITES`` is set to ``"1"``.

    """
    return os.getenv("UAM4X_DISABLE_REGISTRY_WRITES") != "1"


def check_and_get_config(conf_path):
    """Load a UAM4X registry configuration file, creating defaults if needed.

    Parameters
    ----------
    conf_path : str or path-like
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Configuration dictionary containing at least ``"SUBSYSTEMS"`` and
        ``"PLUGINS"`` sections.

    Raises
    ------
    IOError
        If the configuration file exists but cannot be parsed as JSON.

    Notes
    -----
    When registry writes are enabled and the current process is the main
    process, missing config files or missing top-level sections are created on
    disk automatically.

    """
    if not os.path.exists(conf_path):
        if is_main_process() and registry_writes_enabled():
            log.info(f"No config found at {conf_path}. Creating new empty config file.")
            with open(conf_path, "w") as configfile:
                json.dump({}, configfile)
        else:
            return {"SUBSYSTEMS": {}, "PLUGINS": {}}

    with open(conf_path, "r") as configfile:
        try:
            _config = json.load(configfile)
        except json.decoder.JSONDecodeError:
            log.error(
                f"Could not load parse json at {conf_path}. Please correct or delete the file."
            )
            raise IOError(
                f"Could not load parse json at {conf_path}. Please correct or delete the file."
            )

    if "SUBSYSTEMS" not in _config:
        _config["SUBSYSTEMS"] = {}
        if is_main_process() and registry_writes_enabled():
            log.info(
                f"No SUBSYSTEMS found in config at {conf_path}. Creating empty SUBSYSTEMS section."
            )
            with open(conf_path, "w") as configfile:
                json.dump(_config, configfile, indent=4)

    if "PLUGINS" not in _config:
        _config["PLUGINS"] = {}
        if is_main_process() and registry_writes_enabled():
            log.info(
                f"No PLUGINS found in config at {conf_path}. Creating empty PLUGINS section."
            )
            with open(conf_path, "w") as configfile:
                json.dump(_config, configfile, indent=4)

    return _config


def is_continuous_type_or_value(val_or_type):
    # Handle values (e.g 1, 2.0, np.array([0.0,0.0,...]),np.array([0,0,...]))
    """Return whether a value or type is continuous-valued.

    Parameters
    ----------
    val_or_type : object
        Value or Python type to test.

    Returns
    -------
    bool
        True for floating-point scalars, floating-point NumPy arrays, the
        ``float`` type, NumPy floating subclasses, and ``numpy.ndarray``.
        Otherwise False.

    """
    if is_float(val_or_type):
        return True
    elif is_numpy_array(val_or_type):
        return np.issubdtype(val_or_type.dtype, np.floating)
    # Handle types
    if isinstance(val_or_type, type) and (
        val_or_type == float or issubclass(val_or_type, (np.floating, np.ndarray))
    ):
        return True

    return False


def is_scalar_type_or_value(val_or_type):
    """Return whether a value or type is a numeric scalar.

    Parameters
    ----------
    val_or_type : object
        Value or Python type to test.

    Returns
    -------
    bool
        True for real-valued scalar numbers and scalar numeric types, excluding
        booleans. Otherwise False.

    """
    if isinstance(val_or_type, (Real)) and not is_bool(val_or_type):
        return True
    if isinstance(val_or_type, type) and (
        issubclass(val_or_type, (Real, np.integer, np.floating))
        and val_or_type not in (np.bool_, bool)
    ):
        return True

    return False


def type_string_to_type(type_name: str):
    """Return a Python type from a serialized type name.

    Parameters
    ----------
    type_name : str
        Serialized type name, such as ``"int"``, ``"float"``, or
        ``"ndarray"``.

    Returns
    -------
    type
        Python type associated with ``type_name``.

    Raises
    ------
    KeyError
        If ``type_name`` is not in the supported type map.

    """
    return TYPE_MAP[type_name.lower()]


def convert_numpy_to_python(value):
    """Convert NumPy scalar and array values to Python-serializable objects.

    Parameters
    ----------
    value : object
        Value to convert recursively.

    Returns
    -------
    object
        Converted value with NumPy scalars converted to Python scalars and
        NumPy arrays converted to nested Python lists.

    """
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [convert_numpy_to_python(x) for x in value.tolist()]
    if isinstance(value, list):
        return [convert_numpy_to_python(x) for x in value]
    if isinstance(value, dict):
        return {k: convert_numpy_to_python(x) for k, x in value.items()}

    return value

def convert_numpy_to_python(value):
    """Convert NumPy scalar and array values to Python-serializable objects.

    Parameters
    ----------
    value : object
        Value to convert recursively.

    Returns
    -------
    object
        Converted value with NumPy scalars converted to Python scalars and
        NumPy arrays converted to nested Python lists.

    """
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [convert_numpy_to_python(x) for x in value.tolist()]
    if isinstance(value, list):
        return [convert_numpy_to_python(x) for x in value]
    if isinstance(value, dict):
        return {k: convert_numpy_to_python(x) for k, x in value.items()}

    return value

def check_numpy_shape(shape):
    # Integer shapes are allowed
    """Return whether a shape specification is valid for a NumPy array.

    Parameters
    ----------
    shape : object
        Candidate shape specification.

    Returns
    -------
    bool
        True if ``shape`` is a nonnegative integer or a tuple of nonnegative
        integers, excluding booleans. Otherwise False.

    """
    if is_bool(shape):
        return False
    elif is_integer(shape):
        return shape >= 0
    elif is_tuple(shape):
        for dim in shape:
            if is_bool(dim):
                return False
            if not is_integer(dim):
                return False
            if dim < 0:
                return False
        return True

    return False


def normalize_shape(shape):
    """Normalize a scalar or tuple shape into tuple form.

    Parameters
    ----------
    shape : int or tuple
        Shape specification to normalize.

    Returns
    -------
    tuple
        Tuple form of ``shape``.

    """
    if is_tuple(shape):
        return shape

    return (int(shape),)


# Nicer type checkers


def is_bool(value):
    """Return whether a value is boolean-like.

    Parameters
    ----------
    value : object
        Value to test.

    Returns
    -------
    bool
        True if ``value`` is a Python or NumPy boolean, otherwise False.

    """
    return isinstance(value, (bool, np.bool_))


def is_integer(value):
    """Return whether a value is an integer.

    Parameters
    ----------
    value : object
        Value to test.

    Returns
    -------
    bool
        True if ``value`` is a Python or NumPy integer, excluding booleans.

    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def is_float(value):
    """Return whether a value is a floating-point number.

    Parameters
    ----------
    value : object
        Value to test.

    Returns
    -------
    bool
        True if ``value`` is a Python or NumPy floating-point value.

    """
    return isinstance(value, (float, np.floating))


def is_tuple(value):
    """Return whether a value is a tuple.

    Parameters
    ----------
    value : object
        Value to test.

    Returns
    -------
    bool
        True if ``value`` is a tuple, otherwise False.

    """
    return isinstance(value, tuple)


def is_numpy_array(value):
    """Return whether a value is a NumPy array.

    Parameters
    ----------
    value : object
        Value to test.

    Returns
    -------
    bool
        True if ``value`` is a NumPy ndarray, otherwise False.

    """
    return isinstance(value, np.ndarray)


# Get type name from a value
def value_to_type_name(value):
    """Return a readable type name for a value.

    Parameters
    ----------
    value : object
        Value whose type name should be reported.

    Returns
    -------
    str or None
        ``__name__`` of the value's type when available, otherwise the string
        form of the type. Returns None only if neither representation is
        available.

    """
    value_type = type(value)
    if hasattr(value_type, "__name__"):
        return value_type.__name__
    elif str(value_type) is not None:
        return str(value_type)
    else:
        return None


def _canonicalize_state(x):
    """Convert a state object into a deterministic, hashable form.

    Parameters
    ----------
    x : object
        Value to canonicalize recursively.

    Returns
    -------
    object
        Canonicalized representation with dictionaries sorted by key, NumPy
        arrays converted to lists, NumPy scalars converted to Python scalars,
        and Python ``float`` values rounded for stable hashing.

    """
    if isinstance(x, dict):
        return {k: _canonicalize_state(x[k]) for k in sorted(x.keys())}

    if isinstance(x, list):
        return [_canonicalize_state(v) for v in x]

    if isinstance(x, np.ndarray):
        return _canonicalize_state(x.tolist())

    if isinstance(x, float):
        return round(x, 12)

    if isinstance(x, (np.integer, np.bool_)):
        return x.item()

    if isinstance(x, np.floating):
        return round(x.item(),12)

    return x


def hash_dict(some_dict):
    """Return a deterministic SHA-256 hash for a state dictionary.

    Parameters
    ----------
    state : dict
        State dictionary to hash.

    Returns
    -------
    str
        SHA-256 hash of the canonicalized state.

    Notes
    -----
    States are canonicalized with :func:`_canonicalize_state` before hashing so
    that equivalent states produce the same hash regardless of dictionary key
    order or NumPy container types.

    """
    canon = _canonicalize_state(some_dict)

    data = json.dumps(
        canon,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def _is_valid_square_brackets(string):
    """Validate that square brackets in an index string are balanced.

    Parameters
    ----------
    string : str
        Index string to validate.

    Returns
    -------
    bool
        True if square brackets are balanced.

    Raises
    ------
    ValueError
        If the bracket structure is invalid.

    """
    stack = []
    for letter in string:
        if letter == "[":
            stack.append(letter)
        if letter == "]" and not stack:
            raise ValueError(f"Brackets are invalid in indexing string: {string}")
        elif letter == "]":
            stack.pop()
    if not stack:
        return True
    else:
        raise ValueError(f"Brackets are inalid in indexing string: {string}")


def _is_valid_index(string: str):
    """Validate the syntax of a bracketed index expression.

    Parameters
    ----------
    string : str
        Bracketed index expression, such as ``"[0]"`` or ``"[1:5, [0, 2]]"``.

    Returns
    -------
    bool
        True if the index expression is syntactically valid.

    Raises
    ------
    ValueError
        If the index expression violates the supported indexing grammar.

    """

    valid_characters = [",", "-", ":", "[", "]"]
    # At most two consectutive colons
    if ":::" in string:
        raise ValueError(
            "At most 2 consecutive colons allowed when specifying indices. The string:\n"
            f"{string} has invalid indexes."
        )

    if not (string.startswith("[") and string.endswith("]")):
        raise ValueError(
            f"Indexes must be surrounded with [] the string: {string} does not have this"
        )

    for i, letter in enumerate(string[1:-1]):
        loc = i + 1
        # letter must be in the valid characters or an integer
        if not (letter in valid_characters or letter.isdigit()):
            raise ValueError(f"The string: {string} is not a valid array index")
        # number must come after minus sign
        if letter == "-" and not string[loc + 1].isdigit():
            raise ValueError(
                f"A minus sign must be proceeded with an integer index. The string: {string} is not valid."
            )
        # commas must be proceeded by -, : or int or opening bracket
        if letter == "," and not (
            string[loc + 1].isdigit() or string[loc + 1] in valid_characters[1:-1]
        ):
            raise ValueError(
                f"A comma must be proceeded with a colon, integer, or opening bracket. The string: {string}\n"
                f"is not valid. "
            )
        # check for leading zeros which occur after commas, colons or open brackets
        if (
            letter == "0"
            and string[loc - 1] in itemgetter(0, 2, 3)(valid_characters)
            and string[loc + 1].isdigit()
        ):
            raise ValueError(
                f"Leading zeros are not allowed, the string {string} is not valid. "
            )
        # handle [
        if letter == "[" and not (string[loc + 1].isdigit() or string[loc + 1] == ":"):
            raise ValueError(
                f"Internal brackets must be proceeded with a colon or an integer slice. The string {string} is not valid"
            )
        # handle ]
        if letter == "]" and loc != len(string) - 2 and string[loc + 1] != ",":
            raise ValueError(
                f"Internal closing brackets must be proceeded with commas. The string {string} is not valid"
            )

    return True


def _is_simple(string):
    """Return whether an index fragment is simple integer indexing.

    Parameters
    ----------
    string : str
        Index fragment to test.

    Returns
    -------
    bool
        True if the fragment contains only integer-style indexing and no
        slicing or nested fancy indexing.

    """
    for letter in string:
        if letter == "[":
            return False
        if letter == ":":
            return False
    return True


def _is_slice(string):
    """Return whether an index fragment contains slice notation.

    Parameters
    ----------
    string : str
        Index fragment to test.

    Returns
    -------
    bool
        True if the fragment contains ``":"``, otherwise False.

    """
    if ":" in string:
        return True
    return False


def _is_fancy(string):
    """Return whether an index fragment uses nested fancy indexing.

    Parameters
    ----------
    string : str
        Index fragment to test.

    Returns
    -------
    bool
        True if the fragment uses bracketed fancy indexing.

    Raises
    ------
    ValueError
        If slicing syntax is mixed into a fancy-index fragment.

    """
    if "[" in string and "]" in string:
        if ":" in string:
            raise ValueError(f"Slicing not allowed in fancy indexing")
        else:
            return True
    return False


def _parse_simple(string):
    """Parse a simple integer index fragment.

    Parameters
    ----------
    string : str
        Index fragment containing one or more integer indices.

    Returns
    -------
    int or tuple of int
        Parsed integer index or tuple of integer indices.

    """
    indices = re.findall(r"[-]?\d+", string)
    indices = [int(num) for num in indices]
    if len(indices) == 1:
        return indices[0]

    return tuple(indices)


def _parse_fancy(string):
    """Parse a fancy-index fragment into a list of integers.

    Parameters
    ----------
    string : str
        Fancy-index fragment.

    Returns
    -------
    list of int
        Parsed fancy-index values.

    """
    indices = re.findall(r"[-]?\d+", string)
    return [int(num) for num in indices]


def _parse_slice(string):
    """Parse a slice fragment into a Python ``slice`` object.

    Parameters
    ----------
    string : str
        Slice fragment, such as ``"1:5"`` or ``"::2"``.

    Returns
    -------
    slice
        Parsed Python slice object.

    Raises
    ------
    ValueError
        If the slice syntax is invalid or the slice step is zero.

    """
    if string.count(":") > 2:
        raise ValueError(
            f"At most 2 colons for slicing within category {string} in violation"
        )

    s = string.split(":")

    while len(s) < 3:
        s.append("")

    def empty_to_none(ex_string):
        """Execute empty to none.

        Parameters
        ----------
        ex_string : Any
            Input value for `ex_string`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        if ex_string == "":
            return None
        return int(ex_string)

    start, stop, step = [empty_to_none(letter) for letter in s]

    if step == 0:
        raise ValueError(f"Step size cannot be zero. Slice: {string} in violation")

    return slice(start, stop, step)


def _parse_complicated(string):
    """Parse a multi-part index expression.

    Parameters
    ----------
    string : str
        Bracketed index expression containing multiple comma-separated parts.

    Returns
    -------
    tuple
        Parsed index tuple containing integers, slices, and/or fancy-index
        lists.

    Raises
    ------
    ValueError
        If any index fragment cannot be parsed.

    """
    comma_inds = []
    depth = 0
    for i, letter in enumerate(string[1:-1]):
        loc = i + 1
        if letter == "[":
            depth += 1
        if letter == "]":
            depth -= 1
        if letter == "," and depth == 0:
            comma_inds.append(loc)

    lower_list = [1] + [comma_ind + 1 for comma_ind in comma_inds]
    upper_list = [comma_ind for comma_ind in comma_inds] + [-1]

    slice_list = []
    for lower, upper in zip(lower_list, upper_list):
        slice_list.append(slice(lower, upper))

    index_groups = []
    for slice_inds in slice_list:
        index_groups.append(string[slice_inds])

    full_index = []
    for group in index_groups:
        if _is_simple(group):
            full_index.append(_parse_simple(group))
        elif _is_fancy(group):
            full_index.append(_parse_fancy(group))
        elif _is_slice(group):
            full_index.append(_parse_slice(group))
        else:
            raise ValueError(f"Issue")

    if len(full_index) == 1:
        return full_index[0]

    return tuple(full_index)


def _parse_index(string):
    """Parse a validated bracketed index expression.

    Parameters
    ----------
    string : str
        Bracketed index expression.

    Returns
    -------
    object
        Parsed index object suitable for NumPy-style indexing.

    Raises
    ------
    ValueError
        If the bracket structure or index syntax is invalid.

    """
    string=string.replace(' ','')
    _is_valid_square_brackets(string)
    _is_valid_index(string)
    # Try first form:
    if _is_simple(string[1:-1]):
        return _parse_simple(string)
    else:
        return _parse_complicated(string)


def split_dotted_path_and_indices(dotted_path):
    """Split a dotted path and trailing index expression.

    Parameters
    ----------
    dotted_path : str
        Dotted variable path with a trailing index expression, such as
        ``"solution.step.output[0, 1:3]"``. The string must include a
        bracketed index suffix.

    Returns
    -------
    base_path : str
        Dotted path with the trailing index expression removed.
    inds : object
        Parsed index object suitable for NumPy-style indexing.

    Raises
    ------
    ValueError
        If the trailing index expression is not valid.

    """
    bracket_ind = dotted_path.find("[")
    inds = _parse_index(dotted_path[bracket_ind:])
    dotted_path = dotted_path[:bracket_ind]

    return dotted_path, inds

def parse_variable_reference(name):
    if "[" in name:
        return split_dotted_path_and_indices(name)
    return name,None


def replace_none_with_empty_quotes(value):
    """Replace None values recursively with empty strings.

    Parameters
    ----------
    value : object
        Value to transform recursively.

    Returns
    -------
    object
        Transformed value with None replaced by ``""`` inside nested arrays,
        lists, and dictionaries.

    """
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        return [replace_none_with_empty_quotes(x) for x in value.tolist()]
    if isinstance(value, list):
        return [replace_none_with_empty_quotes(x) for x in value]
    if isinstance(value, dict):
        return {k: replace_none_with_empty_quotes(x) for k, x in value.items()}

    return value


def replace_empty_quotes_with_none(value):
    """Replace empty strings recursively with None.

    Parameters
    ----------
    value : object
        Value to transform recursively.

    Returns
    -------
    object
        Transformed value with ``""`` replaced by None inside nested arrays,
        lists, and dictionaries.

    """
    if isinstance(value, str) and value == "":
        return None
    if isinstance(value, np.ndarray):
        return [replace_empty_quotes_with_none(x) for x in value.tolist()]
    if isinstance(value, list):
        return [replace_empty_quotes_with_none(x) for x in value]
    if isinstance(value, dict):
        return {k: replace_empty_quotes_with_none(x) for k, x in value.items()}

    return value

def _get_class_from_file(class_name, file_path):
    """Load a class object from a Python source file.

    Parameters
    ----------
    class_name : str
        Name of the class to retrieve from the loaded module.
    file_path : str or path-like
        Path to the Python source file.

    Returns
    -------
    tuple
        Two-element tuple ``(cls, message)`` where ``cls`` is the loaded class
        object and ``message`` is an empty string on success. If module
        execution or attribute lookup fails, returns ``(None, message)``.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ImportError
        If an import specification cannot be created for ``file_path``.

    """
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Python file not found: {file_path}")

    canon = canonicalize_path(file_path)
    module_name = f"_uam4x_dynamic_{hashlib.sha256(canon.encode('utf-8')).hexdigest()}"

    # spec defines the name of the module and the corresponding location of the file to import
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {file_path}")

    #this is just that defined module
    module = importlib.util.module_from_spec(spec)
    # add the module to the system modules
    sys.modules[module_name] = module

    try:
        #run the module code file to populate the module's namespace
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
    except Exception as e:
        msg = f"Error loading module {module_name}: {e}"
        return None, msg
    return cls, ""

def canonicalize_path(path):
    """Return a canonicalized absolute path.

    Parameters
    ----------
    path : str or path-like
        Path to canonicalize.

    Returns
    -------
    str
        Canonicalized absolute path.

    """
    return os.path.normcase(os.path.normpath(os.path.realpath(os.path.abspath(path))))


def declare_standard_io(plugin, plugin_options):
    continuous_inputs=[]
    for name, metadata in plugin_options.get("input_dict", {}).items():
        var_type=type_string_to_type(metadata["type"])
        plugin.declare_input(
            name,
            var_type=var_type,
            value=metadata.get("default_value"),
        )
        if is_continuous_type_or_value(var_type):
            continuous_inputs.append(name)


    continuous_outputs=[]
    for name, metadata in plugin_options.get("output_dict", {}).items():
        var_type=type_string_to_type(metadata["type"])
        plugin.declare_output(
            name,
            var_type=var_type,
            value=metadata.get("default_value"),
        )
        if is_continuous_type_or_value(var_type):
            continuous_outputs.append(name)

    return continuous_inputs,continuous_outputs
