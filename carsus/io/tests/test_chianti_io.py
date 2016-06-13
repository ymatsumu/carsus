import pytest
from ..chianti_io import ChiantiIonReader, ChiantiIngester
from carsus.model import Level, LevelEnergy, Ion
from numpy.testing import assert_almost_equal


@pytest.fixture(scope="module")
def ch_ion_reader():
    return ChiantiIonReader("ne_2")


@pytest.fixture
def ch_ingester(test_session):
    ions_list = ['ne_2', 'cl_4']
    ingester = ChiantiIngester(test_session, ions_list=ions_list)
    return ingester


@pytest.mark.parametrize("level_index, energy, energy_theoretical",[
    (1, 0, 0),
    (21, 252953.5, 252954),
])
def test_chianti_reader_read_levels(ch_ion_reader, level_index, energy, energy_theoretical):
    row = ch_ion_reader.levels_df.loc[level_index]
    assert_almost_equal(row['energy'], energy)
    assert_almost_equal(row['energy_theoretical'], energy_theoretical)


@pytest.mark.parametrize("atomic_number, ion_charge, levels_count",[
    (10, 1, 138),
    (17, 3, 5)
])
def test_chianti_ingest_levels_count(test_session, ch_ingester, atomic_number, ion_charge, levels_count):
    ch_ingester.ingest(levels=True)
    ion = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
    assert len(ion.levels) == levels_count