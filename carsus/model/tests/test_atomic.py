import pytest
from carsus.model import Atom, BasicAtom, AtomWeight, DataSource
from astropy import units as u
from astropy.units import UnitsError, UnitConversionError
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.tests.helper import assert_quantity_allclose


def test_basic_atom_count(atom_session):
    assert atom_session.query(BasicAtom).count() == 2


def test_data_source_as_unique(empty_session):
    nist = DataSource.as_unique(empty_session, short_name="nist",
                                name="National Institute of Standards and Technology")
    nist2 = DataSource.as_unique(empty_session, short_name="nist",
                                 name="National Institute of Standards and Technology")
    assert nist is nist2


def test_atom_as_unique(empty_session):
    nist = DataSource.as_unique(empty_session, short_name="nist")
    h_nist1 = Atom.as_unique(empty_session, atomic_number=1, data_source=nist)
    h_nist2 = Atom.as_unique(empty_session, atomic_number=1, data_source=nist)
    assert h_nist1 is h_nist2


def test_data_source_count(atom_session):
    assert atom_session.query(DataSource).count() == 2


def test_data_source_unique_constraint(atom_session):
    duplicate = DataSource(short_name="nist")
    atom_session.add(duplicate)
    with pytest.raises(IntegrityError):
        atom_session.commit()


def test_atom_relationship_basic_atom(atom_session):
    ku = DataSource.as_unique(atom_session, short_name="ku")
    h_ku = Atom.as_unique(atom_session, atomic_number=1, data_source=ku)
    assert h_ku.basic_atom.symbol == "H"


def test_atom_unique_constraint(atom_session):
    ku = DataSource.as_unique(atom_session, short_name="ku")
    duplicate_h_ku = Atom(atomic_number=1, data_source=ku)
    atom_session.add(duplicate_h_ku)
    with pytest.raises(IntegrityError):
        atom_session.commit()


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight",[
    (1, "ku", 1.00811*u.u),
    (1, "nist", 1.00784*u.u)
])
def test_atomic_weights_values(atom_session, atomic_number, ds_short_name, expected_weight):
    ds = DataSource.as_unique(atom_session, short_name=ds_short_name)
    atom = Atom.as_unique(atom_session, atomic_number=atomic_number, data_source=ds)
    aw = atom_session.query(AtomWeight).\
        filter(AtomWeight.atom==atom).one()
    assert_quantity_allclose([aw.quantity], expected_weight)


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight_value",[
    (1, "ku", 8.41364096027349e-58),
    (1, "nist", 8.415894971881755e-58)
])
def test_atomic_weights_convert_to(atom_session, atomic_number, ds_short_name, expected_weight_value):
    ds = DataSource.as_unique(atom_session, short_name=ds_short_name)
    atom = Atom.as_unique(atom_session, atomic_number=atomic_number, data_source=ds)
    atom_weight_value = atom_session.query(AtomWeight.quantity.to(u.solMass).value). \
                        filter(AtomWeight.atom == atom).scalar()
    assert_almost_equal(atom_weight_value, expected_weight_value)
