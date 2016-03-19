from .meta import Base, UnitType, UniqueMixin
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey


class UnitDB(UniqueMixin, Base):
    __tablename__ = 'unit_db'

    id = Column(Integer, primary_key=True)
    unit = Column(UnitType(150), unique=True, nullable=False)
    physical_type_id = Column(Integer, ForeignKey('physical_type.id'))

    physical_type = relationship("PhysicalType")

    @classmethod
    def unique_hash(cls, unit, *args, **kwargs):
        return unit

    @classmethod
    def unique_filter(cls, query, unit, *args, **kwargs):
        return query.filter(UnitDB.unit == unit)

    @classmethod
    def construct(cls, session, unit, *args, **kwargs):
        unit_db = UnitDB(unit=unit)
        unit_db._add_physical_type(session)
        return unit_db

    def _add_physical_type(self, session):
        self.physical_type = PhysicalType.as_unique(session, type=self.unit.physical_type)

    def __repr__(self):
        return "<Unit {0}>".format(self.unit)


class PhysicalType(UniqueMixin, Base):
    __tablename__ = 'physical_type'

    id = Column(Integer, primary_key=True)
    type = Column(String(150), unique=True, nullable=False)

    @classmethod
    def unique_hash(cls, type, *args, **kwargs):
        return type

    @classmethod
    def unique_filter(cls, query, type, *args, **kwargs):
        return query.filter(PhysicalType.type == type)

    def __repr__(self):
        return "<Physical Type: {0}>".format(self.type)