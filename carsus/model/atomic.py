from .meta import Base, UniqueMixin, QuantityMixin
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint, and_
from astropy import units as u


class Atom(Base):
    __tablename__ = "atom"

    atom_id = Column(Integer, primary_key=True)
    atomic_number = Column(Integer, ForeignKey('basic_atom.atomic_number'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)

    ions = relationship("Ion", back_populates='atom')
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


class Ion(Base):
    __tablename__ = "ion"

    ion_id = Column(Integer, primary_key=True)
    atom_id = Column(Integer, ForeignKey('atom.atom_id'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)
    ion_charge = Column(Integer, nullable=False)

    levels = relationship("Level", back_populates='ion')
    atom = relationship("Atom", back_populates='ions')
    data_source = relationship("DataSource", back_populates='ions')

    __table_args__ = (UniqueConstraint('atom_id', 'ion_charge', 'data_source_id'),)

    def __repr__(self):
        return "<Ion {}>".format(self.ion_id)


class Level(Base):
    __tablename__ = "level"

    level_id = Column(Integer, primary_key=True)
    level_index = Column(Integer, nullable=False)  # Index of this level from the data source
    ion_id = Column(Integer, ForeignKey('ion.ion_id'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)
    type = Column(String(20))
    configuration = Column(String(50))
    L = Column(String(2))  # total orbital angular momentum
    J = Column(Float)  # total angular momentum
    spin_multiplicity = Column(Integer)  # 2*S + 1, where S is total spin
    parity = Column(Integer)  # 0 - even, 1 - odd
    # ToDo I think that term column can be derived from L, S, parity and configuration
    term = Column(String(20))

    energies = relationship("LevelEnergy", back_populates="level")
    ion = relationship("Ion", back_populates="levels")
    data_source = relationship("DataSource", back_populates="levels")

    __table_args__ = (UniqueConstraint('level_index', 'ion_id', 'data_source_id'),)

    __mapper_args__ = {
        'polymorphic_identity': 'level',
        'polymorphic_on': type
    }


class ChiantiLevel(Level):
    __tablename__ = "chianti_level"

    level_id = Column(Integer, ForeignKey('level.level_id'), primary_key=True)
    chianti_label = Column(String(10))

    __mapper_args__ = {
        'polymorphic_identity': 'chianti'
    }


class LevelQuantity(QuantityMixin, Base):
    __tablename__ = "level_quantity"

    level_qty_id = Column(Integer, primary_key=True)
    level_id = Column(Integer, ForeignKey('level.level_id'), nullable=False)
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class LevelEnergy(LevelQuantity):

    unit = u.eV
    equivalencies = u.spectral()

    level = relationship("Level", back_populates="energies")

    __mapper_args__ = {
        'polymorphic_identity': 'energy'
    }


class Transition(Base):
    __tablename__ = "transition"

    transition_id = Column(Integer, primary_key=True)
    type = Column(String(50))

    lower_level_id = Column(Integer, ForeignKey('level.level_id'), nullable=False)
    upper_level_id = Column(Integer, ForeignKey('level.level_id'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)

    lower_level = relationship("Level",
                                foreign_keys=[lower_level_id],
                                backref="l_transitions")

    upper_level = relationship("Level",
                                foreign_keys=[upper_level_id],
                                backref="u_transitions")

    data_source = relationship("DataSource", back_populates="transitions")

    __mapper_args__ = {
        'polymorphic_identity': 'transition',
        'polymorphic_on': type
    }


class Line(Transition):
    __tablename__ = "line"

    line_id = Column(Integer, ForeignKey('transition.transition_id'), primary_key=True)

    wavelengths = relationship("LineWavelength", back_populates="line")
    a_values = relationship("LineAValue", back_populates="line")
    gf_values = relationship("LineGFValue", back_populates="line")

    __mapper_args__ = {
        'polymorphic_identity': 'line'
    }


class LineQuantity(QuantityMixin, Base):
    __tablename__ = "line_quantity"

    line_qty_id = Column(Integer, primary_key=True)
    line_id = Column(Integer, ForeignKey("line.line_id"))
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'line_qty',
        'polymorphic_on': type
    }


class LineWavelength(LineQuantity):

    unit = u.Angstrom

    line = relationship("Line", back_populates="wavelengths")

    __mapper_args__ = {
        'polymorphic_identity': 'wavelength'
    }


class LineAValue(LineQuantity):

    unit = u.Unit("s-1")

    line = relationship("Line", back_populates="a_values")

    __mapper_args__ = {
        'polymorphic_identity': 'a_value'
    }


class LineGFValue(LineQuantity):

    unit = u.dimensionless_unscaled

    line = relationship("Line", back_populates="gf_values")

    __mapper_args__ = {
        'polymorphic_identity': 'gf_value'
    }


class ECollision(Transition):
    __tablename__ = "e_collision"

    e_collision_id = Column(Integer, ForeignKey('transition.transition_id'), primary_key=True)
    bt92_ttype = Column(Integer)  # BT92 Transition type
    bt92_cups = Column(Float)  # BT92 Scaling parameter

    energies = relationship("ECollisionEnergy",
                            backref="e_collision")
    gf_values = relationship("ECollisionGFValue",
                            backref="e_collision")
    temp_strengths = relationship("ECollisionTempStrength",
                            backref="e_collision")

    temp_strengths_tuple = association_proxy("temp_strengths", "as_tuple")

    __mapper_args__ = {
        'polymorphic_identity': 'e_collision'
    }


class ECollisionQuantity(QuantityMixin, Base):
    __tablename__ = "e_collision_qty"

    e_collision_qty_id = Column(Integer, primary_key=True)
    e_collision_id = Column(Integer, ForeignKey("e_collision.e_collision_id"))
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'e_collision_qty',
        'polymorphic_on': type
    }


class ECollisionEnergy(ECollisionQuantity):

    unit = u.eV

    __mapper_args__ = {
        'polymorphic_identity': 'energy'
    }

class ECollisionGFValue(ECollisionQuantity):

    unit = u.dimensionless_unscaled

    __mapper_args__ = {
        'polymorphic_identity': 'gf_value'
    }


class ECollisionTempStrength(Base):
    __tablename__ = "e_collision_temp_strength"

    temp_strength_id = Column(Integer, primary_key=True)
    temp = Column(Float)
    strength = Column(Float)
    e_collision_id = Column(Integer, ForeignKey("e_collision.e_collision_id"))

    @property
    def as_tuple(self):
        return self.temp, self.strength

    def __repr__(self):
        return "<Temp: {}, Strength: {}>".format(self.temp, self.strength)


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
    ions = relationship("Ion", back_populates="data_source")
    levels = relationship("Level", back_populates="data_source")
    transitions = relationship("Transition", back_populates="data_source")

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)