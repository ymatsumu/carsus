from .meta import Base, UnitType, UniqueMixin
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer,  ForeignKey


class UnitDB(UniqueMixin, Base):
    __tablename__ = 'unit_db'

    id = Column(Integer, primary_key=True)
    unit = Column(UnitType(150), unique=True, nullable=False)

    @classmethod
    def unique_hash(cls, unit, *args, **kwargs):
        return unit.to_string()

    @classmethod
    def unique_filter(cls, query, unit, *args, **kwargs):
        return query.filter(UnitDB.unit == unit)

    def __repr__(self):
        return "<Unit {0}>".format(self.unit)