import os

from carsus import init_db
from carsus.io.nist import NISTWeightsCompIngester, NISTIonizationEnergiesIngester
from carsus.io.kurucz import GFALLIngester
from carsus.io.chianti_io import ChiantiIngester

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data')
TEST_DB_FNAME = os.path.join(DATA_DIR, 'test.db')
GFALL_FNAME = os.path.join(DATA_DIR, "gftest.all")


def create_test_db(test_db_fname=None, gfall_fname=None):
    """
    Create a database for testing

    Parameters
    ----------
    test_db_fname : str
        Filename for the testing database
    gfall_fname : str
        Filename for the GFALL file
    """
    if test_db_fname is None:
        test_db_fname = TEST_DB_FNAME

    if gfall_fname is None:
        gfall_fname = GFALL_FNAME

    test_db_f = open(test_db_fname, "w")
    test_db_f.close()
    test_db_url = "sqlite:///" + test_db_fname

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