import os
import pandas as pd
import carsus
from carsus.model import BasicAtom, setup


basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data',
                                       'basic_atomic_data.csv')


def init_db(url, **kwargs):
    """
    Initializes the database.
    If the database is empty ingests basic atomic data (atomic numbers, symbols, etc.)

    Parameters
    ----------
    url : str
        The database url

    kwargs
        Additional keyword arguments that can be passed to the `create_engine` function (e.g. echo=True)

    Returns
    --------
    an instance of the sqlalchemy.orm.session.Session class

    """
    print "Initializing the database"
    session = setup(url, **kwargs)

    if session.query(BasicAtom).count() == 0:
        _init_empty_db(session)

    return session


def _init_empty_db(session):
    """ Ingests basic atomic data to an empty database """
    basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
    print "Ingesting basic atomic data"
    for i, row in basic_atomic_data.iterrows():
        session.add(BasicAtom(atomic_number=row['atomic_number'], name=row['name'],
                          symbol=row['symbol'], group=row['group'],
                          period=row['period']))