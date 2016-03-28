"""Types and SQL constructs specific to carsus"""

from sqlalchemy import Unicode
from sqlalchemy.types import TypeDecorator
from astropy import units as u


class UnitType(TypeDecorator):
    """Coerce astropy.Units to string representations for the database"""

    impl = Unicode

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value.to_string()


    def process_result_value(self, value, dialect):
        return u.Unit(value)