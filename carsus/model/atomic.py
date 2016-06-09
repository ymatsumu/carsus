from .meta import Base, UniqueMixin, QuantityMixin, DataSourceMixin

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint, and_
from astropy import units as u


class Atom(Base):
    __tablename__ = "atom"
    atomic_number = Column(Integer, primary_key=True)
    symbol = Column(String(5), nullable=False)
    name = Column(String(150))
    group = Column(Integer)
    period = Column(Integer)

    weights = relationship("AtomWeight", back_populates='atom')

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)


class AtomQuantity(QuantityMixin, DataSourceMixin, Base):
    __tablename__ = "atom_quantity"

    atom_qty_id = Column(Integer, primary_key=True)
    atom_id = Column(Integer, ForeignKey("atom.atom_id"), nullable=False)
    type = Column(String(20))

    __table_args__ = (UniqueConstraint( 'data_source_id', 'atom_id', 'type'),)
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class AtomWeight(AtomQuantity):

    unit = u.u
    atom = relationship("Atom", back_populates='weights')

    __mapper_args__ = {
        'polymorphic_identity': 'weight'
    }


class DataSource(UniqueMixin, Base):
    __tablename__ = "data_source"

    @classmethod
    def unique_hash(cls, short_name, *args, **kwargs):
        return short_name

    @classmethod
    def unique_filter(cls, query, short_name, *args, **kwargs):
        return query.filter(DataSource.short_name == short_name)

    data_source_id = Column(Integer, primary_key=True)
    short_name = Column(String(20), unique=True, nullable=False)
    name = Column(String(120))
    description = Column(String(800))
    data_source_quality = Column(Integer)

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)