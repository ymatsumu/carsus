import os
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