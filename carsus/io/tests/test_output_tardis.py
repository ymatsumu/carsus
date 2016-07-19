import pytest
import os

from carsus.io.output.tardis_op import AtomData
from carsus.model import DataSource, Ion
from numpy.testing import assert_almost_equal
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from sqlalchemy import and_


with_test_db = pytest.mark.skipif(
    not pytest.config.getoption("--test-db"),
    reason="--testing database was not specified"
)


@pytest.fixture
def atom_data(test_session):
    atom_data = AtomData(test_session,
                         ions=["He II", "Be III", "B IV", "N VI", "Si II", "Zn XX"],
                         chianti_ions=["He II", "N VI"])
    return atom_data


@pytest.fixture
def atom_data_only_be2(test_session):
    atom_data = AtomData(test_session, ions=["Be III"])
    return atom_data


@pytest.fixture
def atom_masses(atom_data):
    return atom_data.atom_masses


@pytest.fixture
def ionization_energies(atom_data):
    return atom_data.ionization_energies


@pytest.fixture
def levels(atom_data):
    return atom_data.levels


@pytest.fixture
def lines(atom_data):
    return atom_data.lines


@pytest.fixture
def collisions(atom_data):
    return atom_data.collisions


@pytest.fixture
def macro_atom(atom_data):
    return atom_data.macro_atom


@pytest.fixture
def macro_atom_references(atom_data):
    return atom_data.macro_atom_references


@pytest.fixture
def zeta_data(atom_data):
    return atom_data.zeta_data


@pytest.fixture
def hdf5_path(request, data_dir):
    hdf5_path = os.path.join(data_dir, "test_hdf.hdf5")

    def fin():
      os.remove(hdf5_path)
    request.addfinalizer(fin)

    return hdf5_path


def test_atom_data_init(memory_session):
    nist = DataSource.as_unique(memory_session, short_name="nist-asd")
    ch = DataSource.as_unique(memory_session, short_name="chianti_v8.0.2")
    ku = DataSource.as_unique(memory_session, short_name="ku_latest")
    atom_data = AtomData(memory_session,
                         ions=["He II", "Be III", "B IV", "N VI"],
                         chianti_ions=["He II", "N VI"])
    assert set(atom_data.ions) == set([(2,1), (4,2), (5,3), (7,5)])
    assert set(atom_data.chianti_ions) == set([(2,1), (7,5)])


def test_atom_data_chianti_ions_subset(memory_session):
    nist = DataSource.as_unique(memory_session, short_name="nist-asd")
    ch = DataSource.as_unique(memory_session, short_name="chianti_v8.0.2")
    ku = DataSource.as_unique(memory_session, short_name="ku_latest")
    with pytest.raises(ValueError):
        atom_data = AtomData(memory_session,
                             ions=["He II", "Be III", "B IV", "N VI"],
                             chianti_ions=["He II", "N VI", "Si II"])


def test_atom_data_wo_chianti_ions_attributes(atom_data_only_be2, test_session):
    assert atom_data_only_be2.chianti_ions == list()
    assert test_session.query(atom_data_only_be2.chianti_ions_table).count() == 0


def test_atom_data_wo_chianti_ions_levels(atom_data_only_be2):
    levels402 = atom_data_only_be2.levels.copy()
    assert ((levels402["atomic_number"] == 4) & (levels402["ion_number"] == 2)).all()


@with_test_db
def test_atom_data_join_on_chianti_ions_table(test_session, atom_data):
    chiatni_ions_q = test_session.query(Ion).join(atom_data.chianti_ions_table,
                                     and_(Ion.atomic_number == atom_data.chianti_ions_table.c.atomic_number,
                                          Ion.ion_charge == atom_data.chianti_ions_table.c.ion_charge)).\
        order_by(Ion.atomic_number, Ion.ion_charge)
    chianti_ions = [(ion.atomic_number, ion.ion_charge) for ion in chiatni_ions_q]
    assert set(chianti_ions) == set([(2,1), (7,5)])


@with_test_db
def test_atom_data_two_instances_same_session(test_session):

    atom_data1 = AtomData(test_session,
                         ions=["He II", "Be III", "B IV", "N VI", "Zn XX"],
                         chianti_ions=["He II", "N VI"])
    atom_data2 = AtomData(test_session,
                         ions=["He II", "Be III", "B IV", "N VI", "Zn XX"],
                         chianti_ions=["He II", "N VI"])
    atom_data1.ions_table
    atom_data2.ions_table
    atom_data1.chianti_ions_table
    atom_data2.chianti_ions_table


@with_test_db
@pytest.mark.parametrize("atomic_number, exp_mass", [
    (2, 4.002602 * u.u),
    (4, 9.0121831 * u.u),
    (5, (10.806 + 10.821)/2 * u.u),
    (7, (14.00643 + 14.00728)/2 * u.u),
    (14, (28.084 + 28.086)/2 *u.u),
    (30, 65.38 *u.u)
])
def test_create_atom_masses(atom_masses, atomic_number, exp_mass):
    atom_masses = atom_masses.set_index("atomic_number")
    assert_quantity_allclose(
        atom_masses.loc[atomic_number]["mass"] * u.u,
        exp_mass
    )


@with_test_db
def test_create_atom_masses_max_atomic_number(test_session):
    atom_data = AtomData(test_session, ions=[], atom_masses_max_atomic_number=15)
    atom_masses = atom_data.atom_masses
    assert atom_masses["atomic_number"].max() == 15


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (2, 1,  54.41776311 * u.eV),
    (4, 2, 153.896198 * u.eV),
    (5, 3, 259.3715 * u.eV),
    (7, 5, 552.06731 * u.eV),
    (14, 1, 16.345845 * u.eV),
    (30, 19, 737.366 * u.eV)
])
def test_create_ionizatinon_energies(ionization_energies, atomic_number, ion_number, exp_ioniz_energy):
    ionization_energies = ionization_energies.set_index(["atomic_number", "ion_number"])
    assert_quantity_allclose(
        ionization_energies.loc[(atomic_number, ion_number)]["ionization_energy"] * u.eV,
        exp_ioniz_energy
    )


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number, exp_energy, exp_g, exp_metastable_flag",[
    # Kurucz levels
    (4, 2, 0, 0.0 * u.Unit("cm-1"), 1, True),
    (4, 2, 1, 956501.9 * u.Unit("cm-1"), 3, True),
    (4, 2, 6, 997455.0 * u.Unit("cm-1"), 3, False),
    # CHIANTI levels
    # Theoretical values from CHIANTI aren't ingested!!!
    (7, 5, 0, 0.0 * u.Unit("cm-1"), 1, True),
    (7, 5, 7, 3991860.0 * u.Unit("cm-1"), 3, False),
    (7, 5, 43, 4294670.00 * u.Unit("cm-1"), 5, False),
    # NIST Ground level
    (30, 19, 0, 0.0 * u.eV, 2, True)
])
def test_create_levels(levels, atomic_number, ion_number, level_number,
                       exp_energy, exp_g, exp_metastable_flag):
    levels = levels.set_index(["atomic_number", "ion_number", "level_number"])
    energy = levels.loc[(atomic_number, ion_number, level_number)]["energy"] * u.eV
    g = levels.loc[(atomic_number, ion_number, level_number)]["g"]
    metastable_flag = levels.loc[(atomic_number, ion_number, level_number)]["metastable"]

    # Convert the expected energy using equivalencies
    exp_energy = exp_energy.to(u.eV, equivalencies=u.spectral())

    assert_quantity_allclose(energy, exp_energy)
    assert g == exp_g
    assert metastable_flag == exp_metastable_flag


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number_lower, level_number_upper, exp_wavelength",[
    (7, 5, 1, 2, 1907.9000 * u.Unit("angstrom")),
    (4, 2, 0, 6, 10.0255 * u.Unit("nm"))
])
def test_create_lines(lines, atomic_number, ion_number,
                       level_number_lower, level_number_upper, exp_wavelength):
    lines = lines.set_index(["atomic_number", "ion_number",
                              "level_number_lower", "level_number_upper"])
    wavelength = lines.loc[(atomic_number, ion_number,
                                        level_number_lower, level_number_upper)]["wavelength"] * u.Unit("angstrom")
    assert_quantity_allclose(wavelength, exp_wavelength)


# ToDo: Implement real tests
@with_test_db
def test_create_collisions_df(collisions):
    assert True


@with_test_db
def test_create_macro_atom_df(macro_atom):
    assert True


@with_test_db
def test_create_macro_atom_ref_df(macro_atom_references):
    assert True


@with_test_db
def test_create_zeta_data(zeta_data):
    assert True


@with_test_db
def test_atom_data_to_hdf(atom_data, hdf5_path):
    atom_data.to_hdf(hdf5_path)