
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.model import Base, Atom, DataSource, AtomWeight, BasicAtom, \
    Ion, Level, ChiantiLevel, LevelQuantity, LevelEnergy, Transition, Line,\
    LineGFValue, LineAValue, LineWavelength, LineQuantity, ECollision,\
    ECollisionEnergy, ECollisionTempStrength,  ECollisionGFValue, ECollisionQuantity
from astropy import units as u


@pytest.fixture
def atom_session(empty_session):

    # basic atoms
    basic_h = BasicAtom(atomic_number=1, symbol='H')
    basic_o = BasicAtom(atomic_number=8, symbol='O')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')

    import pdb; pdb.set_trace()
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




@pytest.fixture(scope="session")
def foo_engine():
    engine = create_engine("sqlite://")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = Session(bind=engine)

    # basic atoms
    basic_h = BasicAtom(atomic_number=1, symbol='H')
    basic_ne = BasicAtom(atomic_number=10, symbol='Ne')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')
    ch = DataSource(short_name='ch_v8.0.2')

    # atoms
    h_nist = Atom(
        atomic_number=1, data_source=nist,
        weights=[
            AtomWeight(quantity=1.00784 * u.u, uncert=4e-3)
        ]
    )

    h_ku = Atom(
        atomic_number=1, data_source=ku,
        weights=[
            AtomWeight(quantity=1.00811 * u.u, uncert=4e-3)
        ]
    )

    ne_ch = Atom(
        atomic_number=10,
        data_source=ch
    )

    # ions
    ne1_ch = Ion(atom=ne_ch,
                 data_source=ch,
                 ion_charge=1)
    ne2_ch = Ion(atom=ne_ch,
                 data_source=ch,
                 ion_charge=2)

    # levels
    ne2_lvl0_ch = Level(
        ion=ne2_ch, data_source=ch, level_index=1,
        configuration="2s2.2p5", term="2P1.5", L="P", J=1.5, spin_multiplicity=2, parity=1,
        energies=[
            LevelEnergy(quantity=0, method="m"),
            LevelEnergy(quantity=0, method="th")
        ])

    ne2_lvl1_ch = Level(
        ion=ne2_ch, data_source=ch, level_index=2,
        configuration="2s2.2p5", term="2P0.5", L="P", J=0.5, spin_multiplicity=2, parity=1,
        energies=[
            LevelEnergy(quantity=780.4*u.Unit("cm-1"), method="m"),
            LevelEnergy(quantity=780.0*u.Unit("cm-1"), method="th")
    ])

    ne2_lvl2_ch = Level(
        ion=ne2_ch, data_source=ch, level_index=3,
        configuration="2s2p6", term="2S0.5", L="S", J=0.5, spin_multiplicity=2, parity=0,
        energies=[
            LevelEnergy(quantity=217047.594*u.Unit("cm-1"), method="m"),
            LevelEnergy(quantity=217048*u.Unit("cm-1"), method="th")
    ])

    # lines
    ne2_line0_ch = Line(
        lower_level=ne2_lvl0_ch,
        upper_level=ne2_lvl1_ch,
        data_source=ch,
        wavelengths=[
            LineWavelength(quantity=1*u.AA)
        ],
        a_values=[
            LineAValue(quantity=48*u.Unit("s-1"))
        ],
        gf_values=[
            LineGFValue(quantity=23)
        ]
    )

    # electron collisions
    ne2_ecol0_ch = ECollision(
        lower_level=ne2_lvl0_ch,
        upper_level=ne2_lvl1_ch,
        data_source=ch,
        bt92_ttype=2,
        bt92_cups=11.16,
        energies=[
            ECollisionEnergy(quantity=0.007108*u.rydberg)
        ],
        temp_strengths=[
            (ECollisionTempStrength(temp=0.0, strength=0.255)),
            (ECollisionTempStrength(temp=0.07394, strength=0.266))
        ]
    )

    session.add_all([basic_ne, basic_h, h_nist, h_ku, nist, ku, ch,
                     ne1_ch, ne2_ch, ne2_lvl0_ch, ne2_lvl1_ch, ne2_lvl2_ch,
                     ne2_line0_ch, ne2_ecol0_ch])
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
