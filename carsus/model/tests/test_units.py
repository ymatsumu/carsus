import pytest
from carsus.model import UnitDB
from sqlalchemy.exc import IntegrityError
from astropy import units as u


def test_unit_unit_isinstance_of_unitbase():
    u_m = UnitDB(unit=u.m)
    assert isinstance(u_m.unit, u.UnitBase)


def test_unit_unit_not_nullable(memory_session):
    u_fake = UnitDB()
    memory_session.add(u_fake)
    with pytest.raises(IntegrityError):
        memory_session.commit()


def test_unit_as_unique(memory_session):
    memory_session.bind.echo=True
    u_m = UnitDB.as_unique(memory_session, unit=u.m)
    u_m2 = UnitDB.as_unique(memory_session, unit=u.m)
    assert u_m is u_m2


def test_unit_unique_constraint(memory_session):
    u_m = UnitDB(unit=u.m)
    u_m2 = UnitDB(unit=u.m)
    memory_session.add_all([u_m, u_m2])
    with pytest.raises(IntegrityError):
        memory_session.commit()


def test_unit_as_unique_after_commit(memory_session):
    u_m = UnitDB.as_unique(memory_session, unit=u.m)
    memory_session.commit()
    u_m2 = UnitDB.as_unique(memory_session, unit=u.m)
    assert u_m is u_m2


