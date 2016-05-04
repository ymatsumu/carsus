from ..meta.types import Quantity
from astropy import units as u


def test_quantity_convert_to():
    q1 = Quantity(100, u.m)
    q1.convert_to(u.km).value == 0.1


def test_quantity_add_same_units():
    q1 = Quantity(100, u.m)
    q2 = Quantity(50, u.m)
    assert (q1 + q2).value == 150


def test_quantity_add_diff_units():
    q1 = Quantity(100, u.m)
    q2 = Quantity(0.5, u.km)
    assert (q1 + q2).value == 600


