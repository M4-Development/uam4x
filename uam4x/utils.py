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


def is_main_process():
    """Check whether main process.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    return mp.current_process().name == "MainProcess"


def registry_writes_enabled():
    """Execute registry writes enabled.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    return os.getenv("UAM4X_DISABLE_REGISTRY_WRITES") != "1"


def check_and_get_config(conf_path):
    """Validate and get config.

    Parameters
    ----------
    conf_path : Any
        Path used by this operation.

    Returns
    -------
    Any
        Boolean status for the requested check.

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
    """Validate continuous type or value.

    Parameters
    ----------
    val_or_type : Any
        Input value for `val_or_type`.

    Returns
    -------
    Any
        Boolean status for the requested check.

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
    """Validate if scalar type or value.

    Parameters
    ----------
    val_or_type : Any
        Input value for `val_or_type`.

    Returns
    -------
    Any
        Boolean status for the requested check.

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
    """Execute type string to type.

    Parameters
    ----------
    type_name : str
        Identifier used by this operation.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    return TYPE_MAP[type_name.lower()]


def convert_numpy_to_python(value):
    """Convert numpy to python.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Return value produced by this operation.

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
    """Validate numpy shape.

    Parameters
    ----------
    shape : Any
        Input value for `shape`.

    Returns
    -------
    Any
        Boolean status for the requested check.

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
    """Normalize shape.

    Parameters
    ----------
    shape : Any
        Input value for `shape`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    if is_tuple(shape):
        return shape

    return (int(shape),)


# Nicer type checkers


def is_bool(value):
    """Check whether a value is boolean-like.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    return isinstance(value, (bool, np.bool_))


def is_integer(value):
    """Check whether a value is an integer.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def is_float(value):
    """Check whether a value is a floating-point number.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    return isinstance(value, (float, np.floating))


def is_tuple(value):
    """Check whether a value is a tuple.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    return isinstance(value, tuple)


def is_numpy_array(value):
    """Check whether a value is a NumPy array.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    return isinstance(value, np.ndarray)


# Get type name from a value
def value_to_type_name(value):
    """Execute value to type name.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    value_type = type(value)
    if hasattr(value_type, "__name__"):
        return value_type.__name__
    elif str(value_type) is not None:
        return str(value_type)
    else:
        return None


def _canonicalize_state(x):
    """Execute canonicalize state.

    Parameters
    ----------
    x : Any
        Input value for `x`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    if isinstance(x, dict):
        return {k: _canonicalize_state(x[k]) for k in sorted(x.keys())}

    if isinstance(x, list):
        return [_canonicalize_state(v) for v in x]

    if isinstance(x, np.ndarray):
        return _canonicalize_state(x.tolist())

    if isinstance(x, float):
        return round(x, 12)

    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    return x


def hash_state(state):
    """Execute hash state.

    Parameters
    ----------
    state : Any
        Input value for `state`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    canon = _canonicalize_state(state)

    data = json.dumps(
        canon,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def _is_valid_square_brackets(string):
    """Check whether valid square brackets.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Boolean status for the requested check.

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
    """Check whether valid index.

    Parameters
    ----------
    string : str
        Input value for `string`.

    Returns
    -------
    Any
        Boolean status for the requested check.

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
    """Check whether simple.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    for letter in string:
        if letter == "[":
            return False
        if letter == ":":
            return False
    return True


def _is_slice(string):
    """Check whether slice.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    if ":" in string:
        return True
    return False


def _is_fancy(string):
    """Check whether fancy.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Boolean status for the requested check.

    """
    if "[" in string and "]" in string:
        if ":" in string:
            raise ValueError(f"Slicing not allowed in fancy indexing")
        else:
            return True
    return False


def _parse_simple(string):
    """Parse simple.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    indices = re.findall(r"[-]?\d+", string)
    indices = [int(num) for num in indices]
    if len(indices) == 1:
        return indices[0]

    return tuple(indices)


def _parse_fancy(string):
    """Parse fancy.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    indices = re.findall(r"[-]?\d+", string)
    return [int(num) for num in indices]


def _parse_slice(string):
    """Parse slice.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Return value produced by this operation.

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
    """Parse complicated.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Return value produced by this operation.

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

    return tuple(full_index)


def _parse_index(string):
    """Parse index.

    Parameters
    ----------
    string : Any
        Input value for `string`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
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
    dotted_path : Any
        Input value for `dotted_path`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    bracket_ind = dotted_path.find("[")
    inds = _parse_index(dotted_path[bracket_ind:])
    dotted_path = dotted_path[:bracket_ind]

    return dotted_path, inds


def replace_none_with_empty_quotes(value):
    """Execute replace none with empty quotes.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Return value produced by this operation.

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
    """Execute replace empty quotes with none.

    Parameters
    ----------
    value : Any
        Input value for `value`.

    Returns
    -------
    Any
        Return value produced by this operation.

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
