import pytest
from carsus.model import Atom, AtomWeight, DataSource,\
    Ion, IonizationEnergy
from astropy import units as u
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("atomic_number, expected_symbol",[
    (1, "H"),
    (10, "Ne")
])
def test_atom_query(foo_session, atomic_number, expected_symbol):
    atom = foo_session.query(Atom).get(atomic_number)
    assert atom.symbol == expected_symbol


def test_data_source_as_unique(foo_session):
    nist = DataSource.as_unique(foo_session, short_name="nist",
                                name="National Institute of Standards and Technology")
    nist2 = DataSource.as_unique(foo_session, short_name="nist",
                                 name="National Institute of Standards and Technology")
    assert nist is nist2


def test_data_source_unique_constraint(foo_session):
    nist_duplicate = DataSource(short_name="nist")
    foo_session.add(nist_duplicate)
    with pytest.raises(IntegrityError):
        foo_session.commit()


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight",[
    (1, "ku", 1.00811*u.u),
    (1, "nist", 1.00784*u.u)
])
def test_atomic_weights_query(foo_session, atomic_number, ds_short_name, expected_weight):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom, atom_weight = foo_session.query(Atom, AtomWeight).\
        filter(Atom.atomic_number == atomic_number).\
        join(Atom.weights).\
        filter(AtomWeight.data_source == data_source).one()
    assert_quantity_allclose([atom_weight.quantity], expected_weight)


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight_value",[
    (1, "ku", 8.41364096027349e-58),
    (1, "nist", 8.415894971881755e-58)
])
def test_atomic_weights_convert_to(foo_session, atomic_number, ds_short_name, expected_weight_value):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom, atom_weight_value = foo_session.query(Atom, AtomWeight.quantity.to(u.solMass).value). \
        filter(Atom.atomic_number == atomic_number). \
        join(Atom.weights). \
        filter(AtomWeight.data_source == data_source).one()
    assert_almost_equal(atom_weight_value, expected_weight_value)


@pytest.mark.parametrize("atomic_number, ion_charge, symbol",[
    (1, 0, "H"),
    (10, 1, "Ne")
])
def test_ion_query(foo_session, atomic_number, ion_charge, symbol):
    ion = foo_session.query(Ion).get((atomic_number, ion_charge))
    assert ion.atom.symbol == symbol


@pytest.mark.parametrize("atomic_number, ion_charge",[
    (1, 0)
])
def test_ion_as_unique(foo_session, atomic_number, ion_charge):
    ion = foo_session.query(Ion).get((atomic_number, ion_charge))
    ion2 = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    assert ion is ion2
    

@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, "
                         "method, expected_ionization_energy",[
    (1, 0, "nist", "th", 13.5984*u.eV),
    (10, 1, "nist", "th", 40.96296*u.eV),
    (10, 1, "nist", "m", 40.97*u.eV),
])
def test_ionization_energies_query(foo_session, atomic_number, ion_charge, ds_short_name,
                                   method, expected_ionization_energy):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion, ionization_energy = foo_session.query(Ion, IonizationEnergy).\
        filter(and_(Ion.atomic_number == atomic_number,
                    Ion.ion_charge == ion_charge)).\
        join(Ion.ionization_energies).\
        filter(and_(IonizationEnergy.data_source == data_source,
                    IonizationEnergy.method == method)).one()
    assert_quantity_allclose([ionization_energy.quantity], expected_ionization_energy)