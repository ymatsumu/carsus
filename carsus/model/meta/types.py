"""Types and SQL constructs specific to carsus"""

from sqlalchemy.ext.hybrid import hybrid_method
from decimal import Decimal
from astropy import units as u
from sqlalchemy import literal

class Quantity(object):

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def __add__(self, other):
        return Quantity(
                self.value + other.convert_to(self.unit).value,
                self.unit
            )

    def __sub__(self, other):
        return Quantity(
                self.value - other.convert_to(self.unit).value,
                self.unit
            )

    def __lt__(self, other):
        return self.value < other.convert_to(self.unit).value

    def __gt__(self, other):
        return self.value > other.convert_to(self.unit).value

    def __eq__(self, other):
        return self.value == other.convert_to(self.unit).value

    @hybrid_method
    def convert_to(self, other_unit):
        return Quantity(
            self.value * self.unit.to(other_unit),
            other_unit
        )

    def __clause_element__(self):
        # helper method for SQLAlchemy to interpret
        # the Quantity object as a SQL element
        if isinstance(self.value, (float, int, Decimal)):
            return literal(self.value)
        else:
            return self.value

    def __str__(self):
        return "%2.4f %s" % (self.value, self.unit)

