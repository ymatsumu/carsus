import pytest

from carsus.io.output.tardis_op import BasicAtomData, IonData
from numpy.testing import assert_almost_equal


@pytest.fixture
def basic_atom_data(test_session):
    return BasicAtomData(test_session)


@pytest.fixture
def basic_atom_df(basic_atom_data):
    return basic_atom_data.basic_atom_df


@pytest.fixture
def ion_data(test_session):
    return IonData(test_session)


@pytest.fixture
def ionization_df(ion_data):
    return ion_data.ionization_df


@pytest.mark.parametrize("atomic_number, exp_weight", [
    (2, 4.002602),
    (11, 22.98976928)
])
def test_basic_atom_data(basic_atom_df, atomic_number, exp_weight):
    assert_almost_equal(basic_atom_df.loc[atomic_number]["weight"],
                        exp_weight)


def test_create_basic_atom_data_max_atomic_number(test_session):
    basic_atom_data = BasicAtomData(test_session, max_atomic_number=15)
    basic_atom_df = basic_atom_data.basic_atom_df
    basic_atom_df.reset_index(inplace=True)
    assert basic_atom_df["atomic_number"].max() == 15


@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (8, 6, 138.1189),
    (11, 1,  5.1390767)
])
def test_ion_data(ionization_df, atomic_number, ion_number, exp_ioniz_energy):
    assert_almost_equal(ionization_df.loc[(atomic_number, ion_number)]["ionization_energy"],
                        exp_ioniz_energy)