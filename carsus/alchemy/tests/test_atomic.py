import pytest
from carsus.alchemy.atomic import *
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal


def test_atom_count(session_w_atoms):
    assert session_w_atoms.query(Atom).count() == 2

def test_atom_query_atom(session_w_atoms):
    H = session_w_atoms.query(Atom).\
        filter(Atom.symbol=="H").one()
    assert H.atomic_number == 1


def test_atom_integrity(session_w_atoms):
    fake = Atom(atomic_number=1, symbol="Fk")
    session_w_atoms.add(fake)
    with pytest.raises(IntegrityError):
        session_w_atoms.commit()


def test_data_sources_count(session_w_atomic_weights):
    assert session_w_atomic_weights.query(DataSource).count() == 2


def test_data_sources_integrity(session_w_atomic_weights):
    fake = DataSource(short_name="nist")
    session_w_atomic_weights.add(fake)
    with pytest.raises(IntegrityError):
        session_w_atomic_weights.commit()


def test_atomic_weights_count(session_w_atomic_weights):
    q = session_w_atomic_weights.query(AtomicWeight).\
        filter(AtomicWeight.atomic_number==1).\
        order_by(AtomicWeight.value).all()
    values = [_.value for _ in q]
    assert_allclose(values, [1.00784, 1.00811])


def test_atomic_weights_atom_relationship(session_w_atomic_weights):
    q = session_w_atomic_weights.query(Atom, AtomicWeight).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        join(AtomicWeight.data_source).\
        filter(DataSource.short_name=='nist').first()
    assert_almost_equal(q.AtomicWeight.value, 1.00811)


def test_atomic_weights_integrity(session_w_atomic_weights):
    nist = session_w_atomic_weights.query(DataSource).filter(DataSource.short_name=='nist').one()
    H = session_w_atomic_weights.query(Atom).filter(Atom.atomic_number==1).one()
    H.quantities.append(
        AtomicWeight(data_source=nist, value=666)
    )
    with pytest.raises(IntegrityError):
        session_w_atomic_weights.commit()