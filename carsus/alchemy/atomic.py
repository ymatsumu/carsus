from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint

__all__ = ['Base', 'Atom', 'AtomicQuantity', 'AtomicWeight', 'DataSource']

Base = declarative_base()

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


class AtomicQuantity(Base):
    __tablename__ = "atomic_quantity"
    id = Column(Integer, primary_key=True)
    type = Column(String(20))
    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'))
    data_source_id = Column(Integer, ForeignKey('data_source.id'))
    # unit_id = Column(Integer, ForeignKey('unit.id'))
    value = Column(Float, nullable=False)
    std_dev = Column(Float)
    data_source = relationship("DataSource")

    __table_args__ = (UniqueConstraint('atomic_number', 'data_source_id'),)

    __mapper_args__ = {
        'polymorphic_on':type,
        'polymorphic_identity':'atomic_quantity'
    }

    def __repr__(self):
        return "<Quantity: {0}, value: {1}>".format(self.type, self.value)



class AtomicWeight(AtomicQuantity):
    __mapper_args__ = {
        'polymorphic_identity':'atomic_weight'
    }


class DataSource(Base):
    __tablename__ = "data_source"
    id = Column(Integer, primary_key=True)
    short_name = Column(String(20), unique=True, nullable=False)
    name = Column(String(120))
    description = Column(String(800))
    data_source_quality = Column(Integer)

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)


#class Unit(Base):
#    __tablename__ = 'unit'
#
#    id = Column(Integer, primary_key=True)
#    unit = Column(String(150), unique=True, nullable=False)