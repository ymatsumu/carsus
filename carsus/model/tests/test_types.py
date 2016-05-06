from ..meta.types import DBQuantity
from astropy import units as u


def test_quantity_convert_to():
    q1 = DBQuantity(100, u.m)
    q1.to(u.km).value == 0.1


def test_quantity_add_same_units():
    q1 = DBQuantity(100, u.m)
    q2 = DBQuantity(50, u.m)
    assert (q1 + q2).value == 150


def test_quantity_add_diff_units():
    q1 = DBQuantity(100, u.m)
    q2 = DBQuantity(0.5, u.km)
    assert (q1 + q2).value == 600


