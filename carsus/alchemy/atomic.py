from .meta import Base, UniqueMixin
from .units import UnitDB

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint
from astropy import units as u

# __all__ = ['Base', 'Atom', 'AtomicQuantity', 'AtomicWeight', 'DataSource']


class Atom(Base):
    __tablename__ = "atom"
    atomic_number = Column(Integer, primary_key=True)
    symbol = Column(String(5), nullable=False)
    name = Column(String(150))
    group = Column(Integer)
    period = Column(Integer)
    quantities = relationship("AtomicQuantity",
                    backref='atom',
                    cascade='all, delete-orphan')

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)


class AtomicQuantity(UniqueMixin, Base):
    __tablename__ = "atomic_quantity"

    id = Column(Integer, primary_key=True)
    type = Column(String(20))
    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.id'), nullable=False)
    unit_id = Column(Integer, ForeignKey('unit_db.id'), nullable=False)
    value = Column(Float, nullable=False)
    std_dev = Column(Float)

    data_source = relationship("DataSource")
    unit_db = relationship("UnitDB")

    __table_args__ = (UniqueConstraint('atomic_number', 'data_source_id'),)
    __mapper_args__ = {
        'polymorphic_on':type,
        'polymorphic_identity':'atomic_quantity'
    }

    @classmethod
    def unique_hash(cls, data_source, atom, *args, **kwargs):
        return repr([atom_id, data_source_id])

    @classmethod
    def unique_filter(cls, query, short_name, *args, **kwargs):
        return query.filter(DataSource.short_name == short_name)

    def __repr__(self):
        return "<Quantity: {0}, value: {1}>".format(self.type, self.value)



class AtomicWeight(AtomicQuantity):

    # physical_type = u.u.physical_type

    __mapper_args__ = {
        'polymorphic_identity':'atomic_weight'
    }


class DataSource(UniqueMixin, Base):
    __tablename__ = "data_source"

    @classmethod
    def unique_hash(cls, short_name, *args, **kwargs):
        return short_name

    @classmethod
    def unique_filter(cls, query, short_name, *args, **kwargs):
        return query.filter(DataSource.short_name == short_name)

    id = Column(Integer, primary_key=True)
    short_name = Column(String(20), unique=True, nullable=False)
    name = Column(String(120))
    description = Column(String(800))
    data_source_quality = Column(Integer)

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)