import pytest
from carsus.model import Atom, DataSource, AtomWeight, BasicAtom, \
    Ion, Level, ChiantiLevel, LevelQuantity, LevelEnergy, Transition, Line,\
    LineGFValue, LineAValue, LineWavelength, LineQuantity, ECollision,\
    ECollisionEnergy, ECollisionTempStrength,  ECollisionGFValue, ECollisionQuantity
from astropy import units as u
from astropy.units import UnitsError, UnitConversionError
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.tests.helper import assert_quantity_allclose


def test_basic_atom_count(foo_session):
    assert foo_session.query(BasicAtom).count() == 2


def test_data_source_as_unique(foo_session):
    nist = DataSource.as_unique(foo_session, short_name="nist",
                                name="National Institute of Standards and Technology")
    nist2 = DataSource.as_unique(foo_session, short_name="nist",
                                 name="National Institute of Standards and Technology")
    assert nist is nist2


def test_atom_query(foo_session):
    nist = DataSource.as_unique(foo_session, short_name="nist")
    foo_session.query(Atom).\
        filter(and_(Atom.atomic_number==1,
                   Atom.data_source==nist)).one()


def test_data_source_unique_constraint(foo_session):
    nist_duplicate = DataSource(short_name="nist")
    foo_session.add(nist_duplicate)
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_atom_relationship_basic_atom(foo_session):
    ku = DataSource.as_unique(foo_session, short_name="ku")
    h_ku = foo_session.query(Atom).\
        filter(and_(Atom.atomic_number==1,
                    Atom.data_source==ku)).one()
    assert h_ku.basic_atom.symbol == "H"


def test_atom_unique_constraint(foo_session):
    ku = DataSource.as_unique(foo_session, short_name="ku")
    duplicate_h_ku = Atom(atomic_number=1, data_source=ku)
    foo_session.add(duplicate_h_ku)
    with pytest.raises(IntegrityError):
        foo_session.commit()


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight",[
    (1, "ku", 1.00811*u.u),
    (1, "nist", 1.00784*u.u)
])
def test_atomic_weights_values(foo_session, atomic_number, ds_short_name, expected_weight):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom = foo_session.query(Atom).\
        filter(and_(Atom.atomic_number==atomic_number,
                    Atom.data_source==data_source)).one()
    aw = foo_session.query(AtomWeight).\
        filter(AtomWeight.atom==atom).one()
    assert_quantity_allclose([aw.quantity], expected_weight)


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight_value",[
    (1, "ku", 8.41364096027349e-58),
    (1, "nist", 8.415894971881755e-58)
])
def test_atomic_weights_convert_to(foo_session, atomic_number, ds_short_name, expected_weight_value):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom = foo_session.query(Atom). \
        filter(and_(Atom.atomic_number == atomic_number,
                    Atom.data_source == data_source)).one()
    atom_weight_value = foo_session.query(AtomWeight.quantity.to(u.solMass).value). \
                        filter(AtomWeight.atom == atom).scalar()
    assert_almost_equal(atom_weight_value, expected_weight_value)


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, level_index, method, expected",[
    (10, 2, "ch_v8.0.2", 1, "th", 0*u.Unit("cm-1")),
    (10, 2, "ch_v8.0.2", 1, "m", 0 * u.Unit("cm-1")),
    (10, 2, "ch_v8.0.2", 2, "th", 780.0 * u.Unit("cm-1")),
    (10, 2, "ch_v8.0.2", 2, "m", 780.4 * u.Unit("cm-1"))
])
def test_level_energies_query(foo_session, atomic_number, ion_charge, ds_short_name,
                              level_index, method, expected):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom = foo_session.query(Atom). \
        filter(and_(Atom.atomic_number == atomic_number,
                    Atom.data_source == data_source)).one()

    ion = foo_session.query(Ion). \
        filter(and_(Ion.atom == atom,
                    Ion.ion_charge == ion_charge,
                    Ion.data_source == data_source)).one()

    level, energy = foo_session.query(Level, LevelEnergy).\
        filter(and_(Level.ion==ion,
                    Level.level_index==level_index,
                    Level.data_source==data_source)).\
        join(Level.energies).filter(LevelEnergy.method == method).one()

    with u.set_enabled_equivalencies(u.spectral()):
        assert_quantity_allclose(energy.quantity, expected)



@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, "
                         "lower_level_index, upper_level_index, expected",[
    (10, 2, "ch_v8.0.2", 1, 2, 1*u.AA),
])
def test_line_wavelength_query(foo_session, atomic_number, ion_charge, ds_short_name,
                               lower_level_index, upper_level_index, expected):

    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)

    atom = foo_session.query(Atom). \
        filter(and_(Atom.atomic_number == atomic_number,
                    Atom.data_source == data_source)).one()

    ion = foo_session.query(Ion). \
        filter(and_(Ion.atom == atom,
                    Ion.ion_charge == ion_charge,
                    Ion.data_source == data_source)).one()

    lower_level = foo_session.query(Level).\
        filter(and_(Level.ion==ion,
                    Level.level_index==lower_level_index,
                    Level.data_source==data_source)).one()

    upper_level = foo_session.query(Level). \
        filter(and_(Level.ion == ion,
                    Level.level_index == upper_level_index,
                    Level.data_source == data_source)).one()

    line, wavelength = foo_session.query(Line, LineWavelength). \
        filter(and_(Line.lower_level == lower_level,
                    Line.upper_level == upper_level,
                    Level.data_source == data_source)). \
        join(Line.wavelengths).one()

    assert_quantity_allclose(wavelength.quantity, expected)



@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, "
                         "lower_level_index, upper_level_index, expected", [
                          (10, 2, "ch_v8.0.2", 1, 2, [(0.0, 0.255), (0.07394, 0.266)]),
])
def test_e_collision_temp_strength_query(foo_session, atomic_number, ion_charge, ds_short_name,
                               lower_level_index, upper_level_index, expected):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)

    atom = foo_session.query(Atom). \
        filter(and_(Atom.atomic_number == atomic_number,
                    Atom.data_source == data_source)).one()

    ion = foo_session.query(Ion). \
        filter(and_(Ion.atom == atom,
                    Ion.ion_charge == ion_charge,
                    Ion.data_source == data_source)).one()

    lower_level = foo_session.query(Level). \
        filter(and_(Level.ion == ion,
                    Level.level_index == lower_level_index,
                    Level.data_source == data_source)).one()

    upper_level = foo_session.query(Level). \
        filter(and_(Level.ion == ion,
                    Level.level_index == upper_level_index,
                    Level.data_source == data_source)).one()

    e_col = foo_session.query(ECollision). \
        filter(and_(Line.lower_level == lower_level,
                    Line.upper_level == upper_level,
                    Level.data_source == data_source)).one()

    temp_strengths_list = e_col.temp_strengths_tuple
    sorted([temp_strengths_list], key=lambda x: x[0])  # sort the list on temperatures

    for actual_temp_strength, expected_temp_strength in zip(temp_strengths_list, expected):
        assert_allclose(actual_temp_strength, expected_temp_strength)

