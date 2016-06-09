import pytest
from carsus.model import Atom, AtomWeight, DataSource
from astropy import units as u
from astropy.units import UnitsError, UnitConversionError
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal



@pytest.mark.parametrize("atomic_number, expected_symbol",[
    (1, "H"),
    (10, "Ne")
])
def test_get_atom(foo_session, atomic_number, expected_symbol):
    atom = foo_session.query(Atom).get(atomic_number)
    assert atom.symbol == expected_symbol
