from .meta import Base, UniqueMixin, UnitType

from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.util import symbol
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint, and_, event


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

    def merge_quantity(self, session, source_qty):
        """ Updates an existing quantity or creates a new one"""
        qty_cls = source_qty.__class__
        try:
            target_qty = session.query(qty_cls).\
                         filter(and_(qty_cls.atom==self,
                                     qty_cls.data_source==source_qty.data_source)).one()
            target_qty.value = source_qty.value
            target_qty.unit = source_qty.unit
            target_qty.std_dev = source_qty.std_dev

        except NoResultFound:

            self.quantities.append(source_qty)


class AtomicQuantity(Base):
    __tablename__ = "atomic_quantity"

    id = Column(Integer, primary_key=True)
    type = Column(String(20))
    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.id'), nullable=False)
    unit = Column(UnitType(150), nullable=False)
    value = Column(Float, nullable=False)
    std_dev = Column(Float)

    data_source = relationship("DataSource")

    __table_args__ = (UniqueConstraint('type', 'atomic_number', 'data_source_id'),)
    __mapper_args__ = {
        'polymorphic_on':type,
        'polymorphic_identity':'atomic_quantity',
        'with_polymorphic' : '*'
    }

    def __repr__(self):
        return "<Quantity: {0}, value: {1}>".format(self.type, self.value)

    def to(self, unit):
        """
        Converts a quantity to a new specified unit

        Parameters
        ----------
        unit : astropy.units.Unit

        """
        self.unit = unit  # ``unit_set`` event converts the value


@event.listens_for(AtomicQuantity.unit, "set", propagate=True)
def unit_set(target, value, oldvalue, initiator):
    """ Listen for the unit 'set' event and convert target's value when the new unit is different/"""
    if (oldvalue != symbol('NO_VALUE')) and (oldvalue != value):
        target.value = (target.value*oldvalue).to(value).value


class AtomicWeight(AtomicQuantity):
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