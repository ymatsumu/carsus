import pytest

from carsus.io.output.tardis_op import create_basic_atom_df, create_ionization_df,\
    create_levels_df
from numpy.testing import assert_almost_equal
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose


@pytest.fixture
def basic_atom_df(test_session):
    return create_basic_atom_df(test_session)


@pytest.fixture
def ionization_df(test_session):
    return create_ionization_df(test_session)


@pytest.fixture
def levels_df(test_session):
    return create_levels_df(test_session, chianti_species=["He 2", "N 6"])


@pytest.mark.parametrize("atomic_number, exp_weight", [
    (2, 4.002602),
    (11, 22.98976928)
])
def test_create_basic_atom_df(basic_atom_df, atomic_number, exp_weight):
    assert_almost_equal(basic_atom_df.loc[atomic_number]["weight"],
                        exp_weight)


def test_create_basic_atom_df_max_atomic_number(test_session):
    basic_atom_df = create_basic_atom_df(test_session, max_atomic_number=15)
    basic_atom_df.reset_index(inplace=True)
    assert basic_atom_df["atomic_number"].max() == 15


@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (8, 6, 138.1189),
    (11, 1,  5.1390767)
])
def test_create_ionizatinon_df(ionization_df, atomic_number, ion_number, exp_ioniz_energy):
    assert_almost_equal(ionization_df.loc[(atomic_number, ion_number)]["ionization_energy"],
                        exp_ioniz_energy)


@pytest.mark.parametrize("atomic_number, ion_number, level_number, exp_energy",[
    (7, 6, 7, 3991860.0 * u.Unit("cm-1")),
    (4, 3, 2, 981177.5 * u.Unit("cm-1"))
])
def test_create_levels_df(levels_df, atomic_number, ion_number, level_number, exp_energy):
    energy = levels_df.loc[(atomic_number, ion_number, level_number)]["energy"]*u.eV
    energy = energy.to(u.Unit("cm-1"), equivalencies=u.spectral())
    assert_quantity_allclose(energy, exp_energy)