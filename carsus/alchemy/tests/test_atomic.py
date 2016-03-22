import pytest
from carsus.alchemy import Atom, AtomicWeight, DataSource, UnitDB
from astropy import units as u
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal


def test_atom_count(foo_session):
    assert foo_session.query(Atom).count() == 2


def test_atom_query_atom(foo_session):
    H = foo_session.query(Atom).\
        filter(Atom.symbol=="H").one()
    assert H.atomic_number == 1


def test_atom_unique_constraint(foo_session):
    fake = Atom(atomic_number=1, symbol="Fk")
    foo_session.add(fake)
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_data_source_as_unique(memory_session):
    nist = DataSource.as_unique(memory_session, short_name="nist",
                                name="National Institute of Standards and Technology")
    nist2 = DataSource.as_unique(memory_session, short_name="nist")
    assert nist is nist2


def test_data_sources_count(foo_session):
    assert foo_session.query(DataSource).count() == 2


def test_data_sources_unique_constraint(foo_session):
    duplicate = DataSource(short_name="nist")
    foo_session.add(duplicate)
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_atomic_weights_count(foo_session):
    q = foo_session.query(AtomicWeight).\
        filter(AtomicWeight.atomic_number==1).\
        order_by(AtomicWeight.value).all()
    values = [_.value for _ in q]
    assert_allclose(values, [1.00784, 1.00811])


def test_atomic_weights_atom_relationship(foo_session):
    nist = DataSource.as_unique(foo_session, short_name="nist")
    q = foo_session.query(Atom, AtomicWeight).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        filter(AtomicWeight.data_source == nist).first()

    assert_almost_equal(q.AtomicWeight.value, 1.00784)


def test_atomic_weights_unique_constraint(foo_session):
    nist = DataSource.as_unique(foo_session, short_name="nist")
    H = foo_session.query(Atom).filter(Atom.atomic_number==1).one()
    H.quantities.append(
        AtomicWeight(data_source=nist, value=666)
    )
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_atom_merge_new_quantity(foo_session):
    H = foo_session.query(Atom).filter(Atom.atomic_number==1).one()
    cr = DataSource.as_unique(foo_session, short_name="cr")
    u_u = UnitDB.as_unique(foo_session, unit=u.u)
    H.merge_quantity(foo_session, AtomicWeight(data_source=cr, unit_db=u_u, value=1.00754, std_dev=3e-5,))
    foo_session.commit()
    q = foo_session.query(AtomicWeight).filter(and_(AtomicWeight.atom==H,
                                                    AtomicWeight.data_source==cr)).one()
    assert_almost_equal(q.value, 1.00754)


def test_atom_merge_existing_quantity(foo_session):
    H = foo_session.query(Atom).filter(Atom.atomic_number==1).one()
    nist = DataSource.as_unique(foo_session, short_name="nist")
    u_u = UnitDB.as_unique(foo_session, unit=u.u)
    H.merge_quantity(foo_session,  AtomicWeight(data_source=nist, unit_db=u_u, value=1.00654, std_dev=4e-5,))
    foo_session.commit()
    q = foo_session.query(AtomicWeight).filter(and_(AtomicWeight.atom==H,
                                                    AtomicWeight.data_source==nist)).one()
    assert_almost_equal(q.value, 1.00654)