import warnings
from pathlib import Path
from typing import Dict, List, Union

import toml


def get_ints_from_list(lst: list) -> list:
    """Given a list, returns the subset of integer values. This subset is
    composed by type int object and str object such that applying int method
    does not return a ValueError.
    Parameters
    ----------
    lst : list
        Input list.
    Returns
    -------
    list
        The list containing the potential integer values.
    """
    lst_clean = []
    for s in lst:
        if isinstance(s, (int, str)):
            try:
                lst_clean.append(int(s))
            except ValueError:
                warnings.warn(f"The string '{s}' cannot be casted to an integer.")
        else:
            warnings.warn(f"Element '{s}' is not a string or an integer.")
    return lst_clean


def isfloat(value):
    """Returns True when the value can be converted to float"""
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_str(x: str):
    """Parses a string value x:
    - If x is "none", returns None
    - If x is numeric, returns int(x) or float(x)
    - In any other case, returns the original x
    """
    if not isinstance(x, str):  # Only True when value is not str
        return x
    elif x.lower() == "none":
        return None
    elif x.isnumeric():  # Only True when value is int
        return int(x)
    elif isfloat(x):  # Only True when value is float
        return float(x)
    else:
        return x


def parse_list(ser: list) -> list:
    """Parses the elements of a list"""
    return [parse_str(x) for x in ser]


def parse_dict(d: dict) -> dict:
    """Parses of the elements of a dictionary"""
    out_d = d.copy()
    for key, value in d.items():
        if type(value) is dict:
            out_d.update({key: parse_dict(value)})
        elif type(value) is list:
            out_d.update({key: parse_list(value)})
    return out_d


class Conf(dict):
    """Sub-class of dict that overrides __getitem__ to allow for keys not in
    the original dict, defaulting to None.
    """

    def __init__(self, *args, **kwargs):
        """Update dict with all keys from dict"""
        self.update(*args, **kwargs)
        # Parse the config (specifically, change "None" to None and "int" to int)
        self.update(parse_dict(self))

    def __getitem__(self, key):
        """Get key from dict. If not present, return None and raise warning
        Parameters
        ----------
        key : Hashable
            key to get from original dict
        Returns
        -------
            original value in the dict or None if not present
        """
        if key not in self:
            warnings.warn(f"Key '{key}' not in conf. Defaulting to None")
            val = None
        else:
            val = dict.__getitem__(self, key)
        return val

    def tolist(self) -> List[Dict]:
        list_dict = []
        for key, val in self.items():
            d = val.copy()
            d["key"] = key
            list_dict.append(d)
        return list_dict


def load_config(path: Union[str, Path], key: str = None) -> Conf:
    """Load TOML config as dict-like
    Parameters
    ----------
    path : str
        Path to TOML config file
    key : str, optional
        Section of the conf file to load
    Returns
    -------
    Conf
        Config dictionary
    """
    config = toml.load(path)
    return Conf(config) if key is None else Conf(config[key])


def save_conf(path: Union[str, Path], conf: Conf):
    """Save TOML config as dict-like
    Parameters
    ----------
    path : str
        Path to TOML config file
    conf : Conf
        Config object.
    """
    with open(path, "w") as f:
        toml.dump(conf, f)
