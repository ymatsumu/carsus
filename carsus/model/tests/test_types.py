
import pytest

from astropy import units as u
from astropy.units import UnitsError, UnitConversionError
from numpy.testing import assert_almost_equal
from sqlalchemy import column
from carsus.model.meta.types import DBQuantity


@pytest.fixture
def dbquantity():
    return DBQuantity(column("value"), u.m)


@pytest.fixture
def dbquantity_unitless():
    return DBQuantity(column("value"))


def test_dbquantity_to(dbquantity):
    expr = dbquantity.to(u.km).value  # value * :value_1
    value_1 = expr.right.value
    assert_almost_equal(value_1, 0.001)  # check value_1


def test_dbquantity_add_same_units(dbquantity):
    expr = (dbquantity + 100*u.m).value  # value + :value_1
    value_1 = expr.right.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_radd(dbquantity):
    expr = (100*u.m + dbquantity).value  # :value_1 + value
    value_1 = expr.left.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_add_diff_units(dbquantity):
    expr = (dbquantity + 0.1*u.km).value  # value + :value_1
    value_1 = expr.right.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_sub_same_untis(dbquantity):
    expr = (dbquantity - 100*u.m).value  # value - :value_1
    value_1 = expr.right.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_rsub_same_units(dbquantity):
    expr = (100*u.m - dbquantity).value  # :value_1 - value
    value_1 = expr.left.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_rsub_diff_untis(dbquantity):
    expr = (0.1*u.km - dbquantity).value  # :param_1 - :value_1 * value
    value_1 = expr.right.left.value
    param_1 = expr.left.value
    assert_almost_equal(value_1, 0.001)
    assert_almost_equal(param_1, 0.1)


def test_dbquantity_mul_scalar(dbquantity):
    q = dbquantity*2  # value + :value_1
    expr = q.value
    value_1 = expr.right.value
    assert_almost_equal(value_1, 2)
    assert q.unit == u.m


def test_dbquantity_mul_another_unit(dbquantity):
    q = dbquantity*(2*u.m)  # value + :value_1
    expr = q.value
    value_1 = expr.right.value
    assert_almost_equal(value_1, 2)
    assert q.unit == u.Unit("m**2")


def test_dbquantity_div_scalar(dbquantity):
    q = dbquantity/2
    expr = q.value
    value_1 = expr.right.value
    assert_almost_equal(value_1, 2)
    assert q.unit == u.m

def test_dbquantity_div_another_unit(dbquantity):
    q = dbquantity/(2*u.m)
    expr = q.value
    value_1 = expr.right.value
    assert_almost_equal(value_1, 2)
    assert q.unit == u.Unit("")


def test_dbquantity_greater_same_units(dbquantity):
    expr = dbquantity > 100*u.m  # value > :value_1
    value_1 = expr.right.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_greater_diff_units(dbquantity):
    expr = dbquantity > 0.1*u.km  # value > :value_1
    value_1 = expr.right.value
    assert_almost_equal(value_1, 100)


def test_dbquantity_greater_uncompatible_units(dbquantity):
    with pytest.raises(UnitConversionError):
        expr = dbquantity > 0.1 * u.kg


def test_dbquantity_greater_zero(dbquantity):
    expr = dbquantity > 0
    value_1 = expr.right.value
    assert value_1 == 0


def test_dbquantity_greater_dimensionless_value(dbquantity):
    with pytest.raises(UnitsError):
        expr = dbquantity > 1


def test_dbquantity_unitless_greater(dbquantity_unitless):
    expr = dbquantity_unitless > 1
    value_1 = expr.right.value
    assert value_1 == 1

def test_dbquantity_to_with_equivalence(dbquantity):
    q = dbquantity.to(u.MHz, equivalencies=u.spectral())  #value * :value_1 MHz
    expr = q.value
    value_1 = expr.right.value
    assert_almost_equal(value_1, 299.792458)
    assert q.unit == u.MHz