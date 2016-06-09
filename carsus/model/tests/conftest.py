
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.model import Base, Atom, DataSource, AtomWeight
from astropy import units as u

data_dir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

foo_db_url = 'sqlite:///' + os.path.join(data_dir, 'foo.db')


@pytest.fixture
def memory_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    return session


@pytest.fixture(scope="session")
def foo_engine():
    engine = create_engine(foo_db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = Session(bind=engine)

    # atoms
    h = Atom(atomic_number=1, symbol='H')
    ne = Atom(atomic_number=10, symbol='Ne')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')

    # atomic weights
    h.weights = [
        AtomWeight(quantity=1.00784*u.u, data_source=nist, std_dev=4e-3),
        AtomWeight(quantity=1.00811*u.u, data_source=ku, std_dev=4e-3),
    ]

    session.add_all([h, ne, nist, ku])
    session.commit()
    session.close()
    return engine


@pytest.fixture
def foo_session(foo_engine, request):
    # connect to the database
    connection = foo_engine.connect()

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


