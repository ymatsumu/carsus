from .meta import Base, UniqueMixin, QuantityMixin, DataSourceMixin

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey,\
    UniqueConstraint, ForeignKeyConstraint, and_
from astropy import units as u


class Atom(Base):
    __tablename__ = "atom"
    atomic_number = Column(Integer, primary_key=True)
    symbol = Column(String(5), nullable=False)
    name = Column(String(150))
    group = Column(Integer)
    period = Column(Integer)

    weights = relationship("AtomWeight", back_populates='atom')
    ions = relationship("Ion", back_populates='atom')

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)


class AtomQuantity(QuantityMixin, Base):
    __tablename__ = "atom_quantity"

    atom_qty_id = Column(Integer, primary_key=True)
    atomic_number= Column(Integer, ForeignKey("atom.atomic_number"), nullable=False)
    type = Column(String(20))

    # __table_args__ = (UniqueConstraint('data_source_id', 'atomic_number', 'type', 'method'),)
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


class Ion(UniqueMixin, Base):
    __tablename__ = "ion"

    @classmethod
    def unique_hash(cls, atomic_number, ion_charge, *args, **kwargs):
        return "ion:{0},{1}".format(atomic_number, ion_charge)

    @classmethod
    def unique_filter(cls, query, atomic_number, ion_charge, *args, **kwargs):
        return query.filter(and_(Ion.atomic_number == atomic_number,
                                 Ion.ion_charge == ion_charge))

    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), primary_key=True)
    ion_charge = Column(Integer, primary_key=True)

    ionization_energies = relationship("IonizationEnergy",
                                       back_populates='ion')
    atom = relationship("Atom", back_populates='ions')

    def __repr__(self):
        return "<Ion Z={0} +{1}>".format(self.atomic_number, self.ion_charge)


class IonQuantity(QuantityMixin, Base):
    __tablename__ = "ion_quantity"

    ion_qty_id = Column(Integer, primary_key=True)
    atomic_number= Column(Integer, nullable=False)
    ion_charge = Column(Integer, nullable=False)
    type = Column(String(20))

    __table_args__ = (ForeignKeyConstraint(['atomic_number', 'ion_charge'],
                                           ['ion.atomic_number', 'ion.ion_charge']),)
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class IonizationEnergy(IonQuantity):

    unit = u.eV
    ion = relationship("Ion", back_populates='ionization_energies')

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