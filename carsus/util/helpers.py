import os
import re
import carsus
import numpy as np

from collections import OrderedDict


def data_path(fname):
    return os.path.join(
        os.path.dirname(carsus.__file__), 'data', fname
    )

atomic_symbols_data = np.recfromtxt(data_path('basic_atomic_data.csv'), skip_header=1,
                                    delimiter=',', usecols=(0, 1), names=['atomic_number', 'symbol'])

symbol2atomic_number = OrderedDict(zip(atomic_symbols_data['symbol'],
                                       atomic_symbols_data['atomic_number']))
atomic_number2symbol = OrderedDict(zip(atomic_symbols_data['atomic_number'],
                                       atomic_symbols_data['symbol']))


def convert_camel2snake(name):
    """
    Convert CamelCase to snake_case.

    http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def convert_wavelength_vacuum2air(wavelength_vacuum):
    """
    Convert vacuum wavelength to air wavelength
    Parameters
    -----------
    wavelength_vacuum: float
        Vacuum wavelength in Angstroms

    Returns
    --------
    float
        Air wavelength in Angstroms
    """
    sigma2 = (1e4/wavelength_vacuum)**2.
    fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return wavelength_vacuum/fact


def convert_wavelength_air2vacuum(wavelength_air):
    """
    Convert air wavelength to vacuum wavelength
    Parameters
    -----------
    wavelength_air: float
        Air wavelength in Angstroms

    Returns
    --------
    float
        Vacuum wavelength in Angstroms
    """
    sigma2 = (1e4/wavelength_air)**2.
    fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return wavelength_air * fact


def parse_selected_atoms(selected_atoms):
    """
    Parse the sting specifying selected atoms to the list of atomic numbers.

    Parameters
    ----------
    selected_atoms: str
        Sting that specifies selected atoms. It should consist of comma-separated entries
        that are either single atoms (e.g. "H") or ranges (indicated by using a hyphen between, e.g "H-Zn").
        Element symbols need **not** to be capitalized.

    Returns
    -------
    list of int
        List of atomic numbers

    Examples
    --------

    >>> parse_selected_atoms("H")
    [1]

    >>> parse_selected_atoms("H, Li-N")
    [1, 3, 4, 5, 6, 7]

    >>> parse_selected_atoms("H, Li-N, Si, S")
    [1, 3, 4, 5, 6, 7, 14, 16]

    """
    selected_atomic_numbers = list()
    selected_atoms = [_.strip() for _ in selected_atoms.split(',')]

    for entry in selected_atoms:
        # Case when `entry` is a single atom
        if "-" not in entry:
            entry = entry[:1].upper() + entry[1:].lower()
            try:
                entry_atomic_number = symbol2atomic_number[entry]
            except:
                raise ValueError
            selected_atomic_numbers.append(entry_atomic_number)

        # Case when `entry` is a range of atoms
        else:
            lower, upper = tuple(_.strip() for _ in entry.split('-'))
            lower, upper = map(lambda _: _[:1].upper() + _[1:].lower(), [lower, upper])
            try:
                lower_atomic_number = symbol2atomic_number[lower]
                upper_atomic_number = symbol2atomic_number[upper]
            except:
                raise ValueError
            selected_atomic_numbers += range(lower_atomic_number, upper_atomic_number + 1)

    # Get rid of duplicate numbers if any
    selected_atomic_numbers = list(set(selected_atomic_numbers))

    return selected_atomic_numbers