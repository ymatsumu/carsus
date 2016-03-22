from .meta import Base, UnitType, UniqueMixin
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, event


class UnitDB(UniqueMixin, Base):
    __tablename__ = 'unit_db'

    id = Column(Integer, primary_key=True)
    unit = Column(UnitType(150), unique=True, nullable=False)
    physical_type_id = Column(Integer, ForeignKey('physical_type.id'))

    physical_type = relationship("PhysicalType")

    @classmethod
    def unique_hash(cls, unit, *args, **kwargs):
        return unit.to_string()

    @classmethod
    def unique_filter(cls, query, unit, *args, **kwargs):
        return query.filter(UnitDB.unit == unit)

    def add_physical_type(self, session):
        self.physical_type = PhysicalType.as_unique(session, type=self.unit.physical_type)

    def __repr__(self):
        return "<Unit {0}>".format(self.unit)


#@event.listens_for(Session, 'before_flush')
#def add_physical_type_before_flush(session, flush_context, instances):
#    for obj in session.new:
#        if isinstance(obj, UnitDB):
#            if obj.unit is not None:
#                obj.add_physical_type(session=session)


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