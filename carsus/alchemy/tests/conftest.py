import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.alchemy import Base, Atom, DataSource, AtomicWeight, UnitDB, PhysicalType
from astropy import units as u

@pytest.fixture
def memory_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    return session

"""
@pytest.fixture
def foo_engine(scope="session"):
    engine = create_engine("sqlite:///tests/foo.db")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="session")
def foo_session(foo_engine):
    # atoms
    H = Atom(atomic_number=1, symbol='H')
    O = Atom(atomic_number=8, symbol='O')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')

    # physical types and units
    u.m = UnitDB(unit=u.m, physical_type=PhysicalType(type=u.m.physical_type))
    mass = PhysicalType(type=u.u.physical_type)


@pytest.fixture
def session_w_units(memory_session):
    m = UnitDB(unit=u.m)
    atomic_mass_unit = UnitDB(unit=u.u)
    memory_session.add_all([m, atomic_mass_unit])
    memory_session.commit()
    return memory_session


@pytest.fixture
def session_w_atoms(memory_session):
    H = Atom(atomic_number=1, symbol='H')
    O = Atom(atomic_number=8, symbol='O')
    memory_session.add_all([H, O])
    memory_session.commit()
    return memory_session


@pytest.fixture
def session_w_atomic_weights(session_w_atoms):
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')
    u_u = UnitDB(unit=u.u)
    H = session_w_atoms.query(Atom).filter(Atom.atomic_number==1).one()
    H.quantities = ([
        AtomicWeight(data_source=ku, value=1.00784, unit=u_u),
        AtomicWeight(data_source=nist, value=1.00811, unit=u_u)
    ])
    session_w_atoms.commit()
    return session_w_atoms
"""