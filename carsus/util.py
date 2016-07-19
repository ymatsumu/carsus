import os
import re
import numpy as np

from collections import OrderedDict
import numpy as np


def data_path(fname):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'data', fname)

atomic_symbols_data = np.recfromtxt(data_path('atomic_symbols.dat'), names=['atomic_number', 'symbol'])

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
