import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.alchemy.atomic import *

@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    return session


@pytest.fixture
def session_w_atoms(session):
    H = Atom(atomic_number=1, symbol='H')
    O = Atom(atomic_number=8, symbol='O')
    session.add_all([H,O])
    session.commit()
    return session


@pytest.fixture
def session_w_atomic_weights(session_w_atoms):
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')
    H = session_w_atoms.query(Atom).filter(Atom.atomic_number==1).one()
    H.quantities = ([
        AtomicWeight(atom=H, data_source=ku, value=1.00784),
        AtomicWeight(atom=H, data_source=nist, value=1.00811)
    ])
    session_w_atoms.commit()
    return session_w_atoms