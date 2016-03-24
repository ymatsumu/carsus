import pytest
import os
from sqlalchemy import create_engine
from carsus.alchemy import Base, Session, Atom, DataSource, AtomicWeight, UnitDB
from astropy import units as u

foo_db_url = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'data', 'foo.db')


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
    H = Atom(atomic_number=1, symbol='H')
    O = Atom(atomic_number=8, symbol='O')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')

    # units
    u_m = UnitDB(unit=u.m)
    u_u = UnitDB(unit=u.u)

    # atomic weights
    H.quantities = [
        AtomicWeight(value=1.00784, unit_db=u_u, data_source=nist, std_dev=4e-3),
        AtomicWeight(value=1.00811, unit_db=u_u, data_source=ku, std_dev=4e-3),
    ]

    session.add_all([H, O, nist, ku, u_m, u_u])
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

