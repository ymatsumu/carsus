import pytest
import os

from carsus.io.output.tardis_op import AtomData
from carsus.model import DataSource
from numpy.testing import assert_almost_equal
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose


with_test_db = pytest.mark.skipif(
    not pytest.config.getoption("--test-db"),
    reason="--testing database was not specified"
)


@pytest.fixture
def atom_data(test_session):
    atom_data = AtomData(test_session, chianti_species=["He 2", "N 6"])
    return atom_data


@pytest.fixture
def atom_masses_prepared(atom_data):
    return atom_data.atom_masses_prepared


@pytest.fixture
def ionization_energies_prepared(atom_data):
    return atom_data.ionization_energies_prepared


@pytest.fixture
def levels_prepared(atom_data):
    return atom_data.levels_prepared


@pytest.fixture
def lines_prepared(atom_data):
    return atom_data.lines_prepared


@pytest.fixture
def collisions_prepared(atom_data):
    return atom_data.collisions_prepared


@pytest.fixture
def macro_atom_prepared(atom_data):
    return atom_data.macro_atom_prepared


@pytest.fixture
def macro_atom_references_prepared(atom_data):
    return atom_data.macro_atom_references_prepared


@pytest.fixture
def hdf5_path(request, data_dir):
    hdf5_path = os.path.join(data_dir, "test_hdf.hdf5")

    def fin():
      os.remove(hdf5_path)
    request.addfinalizer(fin)

    return hdf5_path


@with_test_db
@pytest.mark.parametrize("atomic_number, exp_mass", [
    (2, 4.002602),
    (11, 22.98976928)
])
def test_prepare_atom_masses(atom_masses_prepared, atomic_number, exp_mass):
    assert_almost_equal(atom_masses_prepared.loc[atomic_number]["mass"], exp_mass)


@with_test_db
def test_prepare_atom_masses_max_atomic_number(test_session):
    atom_data = AtomData(test_session, atom_masses_max_atomic_number=15)
    atom_masses = atom_data.atom_masses_prepared
    atom_masses.reset_index(inplace=True)
    assert atom_masses["atomic_number"].max() == 15


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (8, 6, 138.1189),
    (11, 1,  5.1390767)
])
def test_prepare_ionizatinon_energies(ionization_energies_prepared, atomic_number, ion_number, exp_ioniz_energy):
    assert_almost_equal(ionization_energies_prepared.loc[(atomic_number, ion_number)]["ionization_energy"],
                        exp_ioniz_energy)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number, exp_energy",[
    (7, 5, 7, 3991860.0 * u.Unit("cm-1")),
    (4, 2, 2, 981177.5 * u.Unit("cm-1"))
])
def test_prepare_levels(levels_prepared, atomic_number, ion_number, level_number, exp_energy):
    levels_prepared.set_index(["atomic_number", "ion_number", "level_number"], inplace=True)
    energy = levels_prepared.loc[(atomic_number, ion_number, level_number)]["energy"] * u.eV
    energy = energy.to(u.Unit("cm-1"), equivalencies=u.spectral())
    assert_quantity_allclose(energy, exp_energy)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number_lower, level_number_upper, exp_wavelength",[
    (7, 5, 1, 2, 1907.9000 * u.Unit("angstrom")),
    (4, 2, 0, 6, 10.0255 * u.Unit("angstrom"))
])
def test_prepare_lines(lines_prepared, atomic_number, ion_number,
                       level_number_lower, level_number_upper, exp_wavelength):
    lines_prepared.set_index(["atomic_number", "ion_number",
                              "level_number_lower", "level_number_upper"], inplace=True)
    wavelength = lines_prepared.loc[(atomic_number, ion_number,
                                        level_number_lower, level_number_upper)]["wavelength"] * u.Unit("angstrom")
    assert_quantity_allclose(wavelength, exp_wavelength)


# ToDo: Implement real tests
@with_test_db
def test_prepare_collisions_df(collisions_prepared):
    assert True


@with_test_db
def test_prepare_macro_atom_df(macro_atom_prepared):
    assert True


@with_test_db
def test_prepare_macro_atom_ref_df(macro_atom_references_prepared):
    assert True


@with_test_db
def test_atom_data_to_hdf(atom_data, hdf5_path):
    atom_data.to_hdf(hdf5_path)