Commit Note 2
=============


Units
------

The mapping table for units is in the `alchemy.units <https://github.com/mishinma/carsus/blob/nist_comp/carsus/alchemy/units.py>`_
module. I created a custom type for storing units - ``UnitType``, that coerces astorpy ``Unit`` objects to unicode strings
with ``to_string`` method. This prevents user from creating units that represent the same physical unit yet are different
in the database, e.g "m" and "meter". I used the name ``UnitDB`` for the class to distinguish it from the astropy ``Unit`` class.
I also used a `unique object <https://bitbucket.org/zzzeek/sqlalchemy/wiki/UsageRecipes/UniqueObject>`_
"for instantiating an object that may or may not correspond to an existing row, kept unique on some field or set of fields".
Use ``as_unique`` function to create  an ``UnitDB`` object::

    from astropy import units as u
    u_m = UnitDB.as_unique(session, unit=u.m)
    session.commit()
    assert u_m.unit == u.m

You can see that ``UnitType`` also converts the result value back to astropy ``Unit`` object.
The ``AtomQuantity`` class from `alchemy.atomic <https://github.com/mishinma/carsus/blob/nist_comp/carsus/alchemy/atomic.py>`_
now has a ``unit_id`` foreign key attribute and a relationship to ``unit_db``. I also defined the ``PhysicalType`` class.
Each unit has a physical type, e.g length is the type for meter. One can use ``u.m.physical_type`` to get the type for a given astropy
Unit object. This is not implemented yet, but I think that a physical type also should be assigned to each quantity
table. Then, when storing a new quantity in the database we can check that its unit has a valid physical type.

