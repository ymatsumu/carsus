import pytest
import os

from carsus.io.kurucz import GFALLReader, GFALLIngester
from numpy.testing import assert_almost_equal, assert_allclose


@pytest.fixture()
def gfall_fname():
    return os.path.join(os.path.dirname(__file__), 'data', 'gftest.all')


@pytest.fixture()
def gfall_rdr(gfall_fname):
    return GFALLReader(gfall_fname)


@pytest.fixture()
def gfall_raw_df(gfall_rdr):
    return gfall_rdr.gfall_raw


@pytest.fixture()
def gfall_df(gfall_rdr):
    return gfall_rdr.gfall_df


@pytest.fixture()
def levels_df(gfall_rdr):
    return gfall_rdr.levels_df

@pytest.fixture()
def lines_df(gfall_rdr):
    return gfall_rdr.lines_df


@pytest.fixture()
def gfall_ingester(test_session, gfall_fname):
    return GFALLIngester(test_session, gfall_fname)


@pytest.mark.parametrize("index, wavelength, element_code, e_first, e_second",[
    (14, 72.5537, 4.02, 983355.0, 1121184.0),
    (37, 2.4898, 7.05, 0.0, 4016390.0)
])
def test_grall_reader_gfall_raw_df(gfall_raw_df, index, wavelength, element_code, e_first, e_second):
    row = gfall_raw_df.loc[index]
    assert_almost_equal(row["element_code"], element_code)
    assert_almost_equal(row["wavelength"], wavelength)
    assert_allclose([row["e_first"], row["e_second"]], [e_first, e_second])


@pytest.mark.parametrize("index, wavelength, atomic_number, ion_charge, "
                         "e_lower, e_upper, e_lower_predicted, e_upper_predicted",[
    (12, 67.5615, 4, 2, 983369.8, 1131383.0, False, False),
    (17, 74.6230, 4, 2, 997455.000, 1131462.0, False, False),
    (41, 16.1220, 7, 5, 3385890.000, 4006160.0, False, True)
])
def test_gfall_reader_gfall_df(gfall_df, index, wavelength, atomic_number, ion_charge,
                               e_lower, e_upper, e_lower_predicted, e_upper_predicted):
    row = gfall_df.loc[index]
    assert row["atomic_number"] == atomic_number
    assert row["ion_charge"] == ion_charge
    assert_allclose([row["wavelength"], row["e_lower"], row["e_upper"]],
                    [wavelength, e_lower, e_upper])
    assert row["e_lower_predicted"] == e_lower_predicted
    assert row["e_upper_predicted"] == e_upper_predicted


@pytest.mark.parametrize("atomic_number, ion_charge, level_index, "
                         "energy, j, method, configuration, term",[
    (4, 2, 0, 0.0, 0.0, "meas", "1s2", "1S"),
    (4, 2, 11, 1128300.0, 2.0, "meas", "s3p", "*3P"),
    (7, 5, 7, 4006160.0, 0.0,  "theor", "s3p",  "*3P")
])
def test_gfall_reader_levels_df(levels_df, atomic_number, ion_charge, level_index,
                                energy, j, method, configuration, term):
    row = levels_df.loc[(atomic_number, ion_charge, level_index)]
    assert_almost_equal(row["energy"], energy)
    assert_almost_equal(row["j"], j)
    assert row["method"] == method
    assert row["configuration"] == configuration
    assert row["term"] == term


@pytest.mark.parametrize("atomic_number, ion_charge, level_index_lower, level_index_upper,"
                         "wavelength, gf",[
    (4, 2, 0, 16, 8.8309, 0.12705741),
    (4, 2, 6, 15, 74.6230, 2.1330449131)
])
def test_gfall_reader_lines_df(lines_df, atomic_number, ion_charge,
                               level_index_lower, level_index_upper, wavelength, gf):
    row = lines_df.loc[(atomic_number, ion_charge, level_index_lower, level_index_upper)]
    assert_almost_equal(row["wavelength"], wavelength)
    assert_almost_equal(row["gf"], gf)

# def test_gfall_ingester_ingest_levels(gfall_ingester):
#     gfall_ingester.ingest_levels()