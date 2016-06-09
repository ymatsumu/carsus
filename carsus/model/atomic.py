from .meta import Base, UniqueMixin, QuantityMixin

from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint, and_
from astropy import units as u


class Atom(UniqueMixin, Base):
    __tablename__ = "atom"

    @classmethod
    def unique_hash(cls, atomic_number, data_source, *args, **kwargs):
        return "atom:{0}_{1}".format(atomic_number, data_source.short_name)

    @classmethod
    def unique_filter(cls, query, atomic_number, data_source, *args, **kwargs):
        return query.filter(and_(Atom.atomic_number == atomic_number,
                                 Atom.data_source == data_source))

    atom_id = Column(Integer, primary_key=True)
    atomic_number = Column(Integer, ForeignKey('basic_atom.atomic_number'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)

    weights = relationship("AtomWeight", back_populates='atom')
    data_source = relationship("DataSource", back_populates='atoms')

    basic_atom = relationship("BasicAtom")

    __table_args__ = (UniqueConstraint('atomic_number', 'data_source_id'),)

    def __repr__(self):
        return "<Atom Z={0}>".format(self.atomic_number)


class BasicAtom(Base):
    __tablename__ = "basic_atom"

    atomic_number = Column(Integer, primary_key=True)
    symbol = Column(String(3), nullable=False)
    name = Column(String(25))
    group = Column(Integer)
    period = Column(Integer)


class AtomQuantity(QuantityMixin, Base):
    __tablename__ = "atomic_quantity"

    atom_qty_id = Column(Integer, primary_key=True)
    atom_id = Column(Integer, ForeignKey("atom.atom_id"), nullable=False)
    type = Column(String(20))

    __table_args__ = (UniqueConstraint('type', 'atom_id'),)
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

    atoms = relationship("Atom", back_populates="data_source")

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)