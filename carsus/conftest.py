# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *
from astropy.tests.pytest_plugins import (
        pytest_addoption as _pytest_add_option
    )

from carsus import init_db
from carsus.io.nist import NISTWeightsCompIngester, NISTIonizationEnergiesIngester
from carsus.io.kurucz import GFALLIngester
from carsus.io.chianti_io import ChiantiIngester
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

## exceptions
# enable_deprecations_as_exceptions()

## Uncomment and customize the following lines to add/remove entries
## from the list of packages for which version numbers are displayed
## when running the tests
# try:
#     PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
#     PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'
#     del PYTEST_HEADER_MODULES['h5py']
# except NameError:  # needed to support Astropy < 1.0
#     pass

## Uncomment the following lines to display the version number of the
## package rather than the version number of Astropy in the top line when
## running the tests.
# import os
#
## This is to figure out the affiliated package version, rather than
## using Astropy's
# from . import version
#
# try:
#     packagename = os.path.basename(os.path.dirname(__file__))
#     TESTED_VERSIONS[packagename] = version.version
# except NameError:   # Needed to support Astropy <= 1.0.0
#     pass

def pytest_addoption(parser):
    _pytest_add_option(parser)
    parser.addoption("--runslow", action="store_true",
                     help="include running slow tests during run")


@pytest.fixture
def memory_session():
    session = init_db(url="sqlite://")
    session.commit()
    return session


@pytest.fixture(scope="session")
def data_dir():
    data_dir = os.path.join(os.path.dirname(__file__), 'tests', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


@pytest.fixture(scope="session")
def test_db_path(data_dir):
    return os.path.join(data_dir, 'test.db')


@pytest.fixture(scope="session")
def test_db_url(test_db_path):
    return "sqlite:///" + test_db_path


@pytest.fixture(scope="session")
def gfall_fname(data_dir):
    return os.path.join(data_dir, 'gftest.all')  # Be III, B IV, N VI


@pytest.mark.remote_data
@pytest.fixture(scope="session")
def test_engine(test_db_path, test_db_url, gfall_fname):

    # If the database for testing exists then just create an engine
    if os.path.isfile(test_db_path):
        engine = create_engine(test_db_url)

    # Else create the database
    else:
        session = init_db(url=test_db_url)
        session.commit()

        # Ingest atomic weights
        weightscomp_ingester = NISTWeightsCompIngester(session)
        weightscomp_ingester.download()
        weightscomp_ingester.ingest()
        session.commit()

        # Ingest ionization energies
        ioniz_energies_ingester = NISTIonizationEnergiesIngester(session)
        ioniz_energies_ingester.download()
        ioniz_energies_ingester.ingest()
        session.commit()

        # Ingest kurucz levels and lines
        gfall_ingester = GFALLIngester(session, gfall_fname)
        gfall_ingester.ingest(levels=True, lines=True)
        session.commit()

        # Ingest chianti levels and lines
        chianti_ingester = ChiantiIngester(session, ions_list=["he_2", "n_6"])
        chianti_ingester.ingest(levels=True, lines=True)
        session.commit()

        session.close()
        engine = session.get_bind()

    return engine


@pytest.fixture
def test_session(test_engine, request):

    # engine.echo=True
    # connect to the database
    connection = test_engine.connect()

    # begin a non-ORM transaction
    trans = connection.begin()

    # bind an individual Session to the connection
    session = Session(bind=connection)

    def fin():
        session.close()
        # rollback - everything that happened with the
        # Session above (including calls to commit())
        # is rolled back.
        trans.rollback()
        # return connection to the Engine
        connection.close()

    request.addfinalizer(fin)

    return session
