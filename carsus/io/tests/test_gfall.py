import pytest
import os

from carsus.io.kurucz import GFALLReader, GFALLIngester
from carsus.model import Ion, Level, LevelEnergy, DataSource, Line, LineWavelength, LineGFValue
from sqlalchemy import and_
from numpy.testing import assert_almost_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as u


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
def gfall_ingester(memory_session, gfall_fname):
    return GFALLIngester(memory_session, gfall_fname)


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


@pytest.mark.parametrize("atomic_number, ion_charge, level_index, "
                          "exp_energy, exp_j, exp_method, exp_configuration, exp_term", [
    (4, 2, 0, 0.0*u.Unit("cm-1"), 0.0, "meas", "1s2", "1S"),
    (4, 2, 11, 1128300.0*u.Unit("cm-1"), 2.0, "meas", "s3p", "*3P"),
    (7, 5, 7, 4006160.0*u.Unit("cm-1"), 0.0, "theor", "s3p", "*3P")
])
def test_gfall_ingester_ingest_levels(memory_session, gfall_ingester, atomic_number, ion_charge, level_index,
                                exp_energy, exp_j, exp_method, exp_configuration, exp_term):
    gfall_ingester.ingest(levels=True, lines=False)
    ion = Ion.as_unique(memory_session, atomic_number=atomic_number, ion_charge=ion_charge)
    data_source = DataSource.as_unique(memory_session, short_name="ku_latest")
    level, energy = memory_session.query(Level, LevelEnergy).\
        filter(and_(Level.ion==ion,
                    Level.level_index==level_index),
                    Level.data_source==data_source).\
        join(Level.energies).one()
    assert level.configuration == exp_configuration
    assert level.term == exp_term
    assert_almost_equal(level.J, exp_j)
    assert_quantity_allclose(energy.quantity, exp_energy.to(u.eV, equivalencies=u.spectral()))
    assert energy.method == exp_method


@pytest.mark.parametrize("atomic_number, ion_charge, level_index_lower, level_index_upper,"
                         "exp_wavelength, exp_gf_value", [
    (4, 2, 0, 16, 8.8309*u.AA, 0.12705741),
    (4, 2, 6, 15, 74.6230*u.AA, 2.1330449131)
])
def test_gfall_ingester_ingest_lines(memory_session, gfall_ingester, atomic_number, ion_charge,
                                     level_index_lower, level_index_upper, exp_wavelength, exp_gf_value):
    gfall_ingester.ingest(levels=True, lines=True)
    ion = Ion.as_unique(memory_session, atomic_number=atomic_number, ion_charge=ion_charge)
    data_source = DataSource.as_unique(memory_session, short_name="ku_latest")
    lower_level = memory_session.query(Level).\
        filter(and_(Level.data_source==data_source,
                    Level.ion==ion,
                    Level.level_index==level_index_lower)).one()
    upper_level = memory_session.query(Level). \
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == level_index_upper)).one()
    line = memory_session.query(Line).\
        filter(and_(Line.data_source==data_source,
                    Line.lower_level==lower_level,
                    Line.upper_level==upper_level)).one()
    wavelength = line.wavelengths[0].quantity
    gf_value = line.gf_values[0].quantity
    assert_quantity_allclose(wavelength, exp_wavelength)
    assert_quantity_allclose(gf_value, exp_gf_value)
