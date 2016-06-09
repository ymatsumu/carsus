
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.model import Base, Atom, DataSource, AtomWeight, BasicAtom
from astropy import units as u


@pytest.fixture
def memory_engine():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def empty_session(memory_engine):
    session = Session(bind=memory_engine)
    return session


@pytest.fixture
def atom_session(empty_session):

    # basic atoms
    basic_h = BasicAtom(atomic_number=1, symbol='H')
    basic_o = BasicAtom(atomic_number=8, symbol='O')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')

    # atoms
    h_nist = Atom(atomic_number=1,
             data_source=nist,
             weights=[
                 AtomWeight(quantity=1.00784 * u.u, uncert=4e-3)
             ])

    h_ku = Atom(atomic_number=1,
              data_source=ku,
              weights=[
                  AtomWeight(quantity=1.00811*u.u, uncert=4e-3)
              ])

    o = Atom(atomic_number=8, data_source=nist)

    empty_session.add_all([basic_h, basic_o, h_nist, h_ku, o, nist, ku])
    empty_session.commit()
    return empty_session