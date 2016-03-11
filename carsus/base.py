import os
import pandas as pd
import carsus
from carsus.alchemy.atomic import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data',
                                       'basic_atomic_data.csv')


class AtomicDatabase(object):

    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Base.metadata.bind = self.engine
        self.session = Session(bind=self.engine)

        if self.session.query(Atom).count() == 0:
            self._init_empty_db()

    def _init_empty_db(self):
        """ Adds atoms to empty database """
        basic_atomic_data = pd.read_csv(basic_atomic_data_fname)

        for i, row in basic_atomic_data.iterrows():
            self.session.add(Atom(atomic_number=row['atomic_number'], name=row['name'],
                              symbol=row['symbol'], group=row['group'],
                              period=row['period']))

        self.session.commit()