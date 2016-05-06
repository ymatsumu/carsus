"""Types and SQL constructs specific to carsus"""

from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.orm.attributes import InstrumentedAttribute
from astropy.units import Quantity, Unit, dimensionless_unscaled
import numpy as np


class DBQuantity(Quantity):
    def __new__(cls, value, unit=None, dtype=None, copy=True, order=None,
                subok=False, ndmin=0):

        if (isinstance(value, InstrumentedAttribute) or
                isinstance(value, ClauseElement)):
            if unit is None:
                unit = dimensionless_unscaled
            else:
                unit = Unit(unit)

            value = np.array(value, dtype=dtype, copy=copy, order=order,
                             subok=False, ndmin=ndmin)
            value = value.view(cls)
            value._unit = unit
            return value

        return Quantity.__new__(Quantity, value, unit=unit, dtype=dtype, copy=copy, order=order,
                                subok=subok, ndmin=ndmin)

    def __gt__(self, other):
        return self.value > other.to(self.unit).value

    def __lt__(self, other):
        return self.value < other.to(self.unit).value

    def __eq__(self, other):
        return self.value == other.to(self.unit).value

    @hybrid_method
    def to(self, other_unit):
        return DBQuantity(self.value * self.unit.to(other_unit), other_unit)
