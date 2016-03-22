import os
import pandas as pd
import carsus
from carsus.alchemy import Base, Atom, Session
from sqlalchemy import create_engine
from contextlib import contextmanager

basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data',
                                       'basic_atomic_data.csv')


class AtomicDatabase(object):

    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Base.metadata.bind = self.engine
        Session.configure(bind=self.engine)

        self.session_maker = Session

        with self.session_scope() as session:
            if session.query(Atom).count() == 0:
                self._init_empty_db(session)

    def _init_empty_db(self, session):
        """ Adds atoms to empty database """
        basic_atomic_data = pd.read_csv(basic_atomic_data_fname)

        for i, row in basic_atomic_data.iterrows():
            session.add(Atom(atomic_number=row['atomic_number'], name=row['name'],
                              symbol=row['symbol'], group=row['group'],
                              period=row['period']))

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()