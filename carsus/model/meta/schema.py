""" Database schema generation/definition helpers """

from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declared_attr
from astropy.units import dimensionless_unscaled, UnitsError, set_enabled_equivalencies
from carsus.model.meta.types import DBQuantity
from carsus.util import convert_camel2snake


class IonListMixin(object):
    """
    Mixin for creating temporary tables for selecting ions from a list of ions.

    Because SQLite doesn't support composite IN expressions
    (you can't do WHERE atomic_number, ion_charge in some_list_of_ions)
    temporary tables are needed for selecting ions.

    This is needed because of ions not having a proper primary key but being
    linked to an atom. FIXME?
    """

    @declared_attr
    def __tablename__(cls):
        return convert_camel2snake(cls.__name__)

    atomic_number = Column(Integer, primary_key=True)
    ion_charge = Column(Integer, primary_key=True)

    __table_args__ = {'prefixes': ['TEMPORARY']}


class DataSourceMixin(object):
    '''
    Mixin that marks a model as datasource dependent by adding `data_source_id`
    and `data_source` class attributes.
    '''

    @declared_attr
    def data_source_id(cls):
        '''
        ID (in the database) of the DataSource
        '''
        return Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)

    @declared_attr
    def data_source(cls):
        '''
        Relationship to the DataSource
        '''
        return relationship("DataSource")


class QuantityMixin(DataSourceMixin):
    '''
    Mixin that marks a database model as a physical quantity.
    '''

    _value = Column(Float, nullable=False)
    #: uncertainty of the measurement
    uncert = Column(Float)
    #: experimental or theoretical data
    method = Column(String(15))
    reference = Column(String(50))

    #: Unit of data
    unit = dimensionless_unscaled
    equivalencies = None

    #: Public interface for value is via `.quantity` accessor
    @hybrid_property
    def quantity(self):
        return DBQuantity(self._value, self.unit)

    @quantity.setter
    def quantity(self, qty):
        try:
            with set_enabled_equivalencies(self.equivalencies):
                self._value = qty.to(self.unit).value
        except AttributeError:
            if self.unit is dimensionless_unscaled or qty == 0:
                self._value = qty
            else:
                raise UnitsError("Can only assign dimensionless values "
                                 "to dimensionless quantities "
                                 "(unless the value is 0)")

    def __repr__(self):
        return "<Quantity: {0} {1}>".format(self._value, self.unit)
