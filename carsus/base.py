import os
import pandas as pd
import carsus
from carsus.alchemy import Atom, setup, session_scope
print "BASE"

basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data',
                                       'basic_atomic_data.csv')


# class AtomicDatabase(object):
#
#     def __init__(self, db_url):
#         self.engine = create_engine(db_url)
#         Base.metadata.create_all(self.engine)
#         Session.configure(bind=self.engine)
#         self.session_maker = Session
#
#         with session_scope() as session:
#             if session.query(Atom).count() == 0:
#                 self._init_empty_db(session)
#
#     def _init_empty_db(self, session):
#         """ Adds atoms to empty database """
#         basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
#
#         for i, row in basic_atomic_data.iterrows():
#             session.add(Atom(atomic_number=row['atomic_number'], name=row['name'],
#                               symbol=row['symbol'], group=row['group'],
#                               period=row['period']))


def init_db(url, **kwargs):
    config={"url":url}
    config.update(kwargs)
    setup(config)

    with session_scope() as session:
        if session.query(Atom).count() == 0:
             _init_empty_db(session)


def _init_empty_db(session):
    """ Adds atoms to empty database """
    basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
    print "Initializing the database"
    for i, row in basic_atomic_data.iterrows():
        session.add(Atom(atomic_number=row['atomic_number'], name=row['name'],
                          symbol=row['symbol'], group=row['group'],
                          period=row['period']))