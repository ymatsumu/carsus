"""Types and SQL constructs specific to carsus"""

from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.orm.attributes import QueryableAttribute
from astropy.units import Quantity, Unit, dimensionless_unscaled, UnitsError
from astropy.units.quantity_helper import get_converter
import numpy as np


class DBQuantity(Quantity):
    """
    Represents a number with some associated unit and
    is used as the public interface for `.quantity` in the database.

    Parameters:
    -----------
    value : sqlalchemy.sql.expression.ClauseElement instance,
            sqlalchemy.orm.attributes.QueryableAttibure instance,
            an object that could be passed to astropy.units.Quantity

    unit : astropy.units.UnitBase instance, str

    """

    def __new__(cls, value, unit=dimensionless_unscaled):

        if (isinstance(value, QueryableAttribute) or
                isinstance(value, ClauseElement)):
            unit = Unit(unit)
            value = np.array(value)
            value = value.view(cls)
            value._unit = unit
            return value

        else:
            return Quantity.__new__(Quantity, value, unit=unit)

    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.

        Parameters
        ----------
        unit : UnitBase
            The unit for which the appropriate class should be returned

        Returns
        -------
        tuple :
            - `Quantity` subclass
            - bool: True is subclasses of the given class are ok
        """
        return DBQuantity, True

    def _helper_twoarg_comparison(self, other):
        other_unit = getattr(other, 'unit', dimensionless_unscaled)
        try:
            converter = get_converter(other_unit, self.unit)
        except UnitsError:
            # special case: would be OK if unitless number is zero
            # ToDo also inf?
            if other == 0:
                converter = None
            else:
                raise UnitsError("Can only apply function to quantities "
                                 "with compatible dimensions")
        return converter

    def __gt__(self, other):
        converter = self._helper_twoarg_comparison(other)

        if converter:
            res = self.value > converter(other.value)
        else:  # with no conversion, other can be non-Quantity.
            res = self.value > getattr(other, 'value', other)

        return res

    def __lt__(self, other):
        converter = self._helper_twoarg_comparison(other)

        if converter:
            res = self.value < converter(other.value)
        else:  # with no conversion, other can be non-Quantity.
            res = self.value < getattr(other, 'value', other)

        return res

    def __eq__(self, other):
        converter = self._helper_twoarg_comparison(other)

        if converter:
            res = self.value = converter(other.value)
        else:  # with no conversion, other can be non-Quantity.
            res = self.value = getattr(other, 'value', other)

        return res

    def to(self, unit):
        """
        Returns a new `DBQuantity` object with the specified
        unit.

        """
        # ToDo improve this implementation
        # `to()` method from `Quantity` doesn't work because it calls
        # `self.unit.to(unit, self.value, equivalencies=equivalencies))`
        # where the value is validated (not acceptable in our case where
        # the value could be a SQLAlchemy constuct

        return DBQuantity(self.value * self.unit.to(unit), unit)
