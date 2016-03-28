.. _quickstart:

Quickstart
==========

This page provides an introduction how to use Carsus.

Initialize a database
---------------------

Initializing a database is a matter of calling the ``init_db`` function.
Let's initialize a SQLite memory database:

.. code:: python

    from carsus import init_db
    init_db("sqlite://")


.. parsed-literal::

    Initializing the database
    Ingesting basic atomic data


Because the database was empty, basic atomic data (atomic numbers, symbols, etc.)
was added to it. Now let's create a session and query the database.
Firstly, we need to import a sessionmaker (that is already bound to our engine):

.. code:: python

    from carsus.model import Session
    print Session


.. parsed-literal::

    sessionmaker(class_='Session',autoflush=True, bind=Engine(sqlite://), autocommit=False, expire_on_commit=True)

We use the sessionmaker to create a session; after that we can work with the database:

.. code:: python

    from carsus.model import Atom
    session = Session()
    q = session.query(Atom).all()
    for atom in q[:5]:
        print atom



.. parsed-literal::

    <Atom H, Z=1>
    <Atom He, Z=2>
    <Atom Li, Z=3>
    <Atom Be, Z=4>
    <Atom B, Z=5>


.. code:: python

    session.query(Atom).count()




.. parsed-literal::

    118


The figure below illustrates the database schema. Atoms have some fundamental quantities, like
atomic numbers and groups, and quantites that can depend on a data source. The latter are stored in
the ``AtomicQuantities`` table. The ``AtomicWeights`` table is a *subset* table
of ``AtomicQuantities`` and it represents a specific *type* of quantities - atomic weights.
Although there is only one quantity type in this schema, generally there can be many.

.. image:: images/atomic_schema.png
    :width: 600
    :align: center

Ingest data
-------------

To ingest data from a source you need to create an ingestor for that source.
In this example we will ingest atomic weights from the
`NIST Atomic Weight and Isotopic Compositions <http://www.nist.gov/pml/data/comp.cfm>`_ database.
After you have created the ingestor, you need to call two methods: ``download`` and ``ingest``.
The first one will download data from the source  and the second one will ingest it
into the database. You **must** pass a Session object to the ``ingest`` method!
You should commit the session after the data have been ingested.

.. code:: python

    from carsus.io.nist import NISTWeightsCompIngester
    ingester = NISTWeightsCompIngester()
    ingester.download()
    ingester.ingest(session)
    session.commit()


.. parsed-literal::

    Downloading the data from http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
    Ingesting atomic weights

Query the database
-------------------

Let's do some queries. To select both atoms and atomic weights we need to join the ``Atoms`` table on
the ``AtomicWeights`` table. We use ``join()`` to create an explicit JOIN.
To specify the ON parameter we provide the relationship-bound attribute of the ``Atom`` class - ``Atom.quantities`` -
and then use the ``of_type()`` helper method to narrow the criterion to atomic weights.
This query selects the first five atoms with the values of their atomic weights:

.. code:: python

    from carsus.model import AtomicWeight, DataSource
    session.query(Atom, AtomicWeight.value).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        filter(Atom.atomic_number <= 5).all()




.. parsed-literal::

    [(<Atom H, Z=1>, 1.007975),
     (<Atom He, Z=2>, 4.002602),
     (<Atom Li, Z=3>, 6.967499999999999),
     (<Atom Be, Z=4>, 9.0121831),
     (<Atom B, Z=5>, 10.8135)]

Before it ingested data the ``ingester`` created a ``DataSource`` object:

.. code:: python

    nist = session.query(DataSource).filter(DataSource.short_name=="nist").one()
    print nist


.. parsed-literal::

    <Data Source: nist>

To create a new data source you should use ``as_unique()`` method.
The method queries the database first to check if the object already exists; if it does the method returns
the object, in the other case it creates a new one.
You should pass a Session object as the first positional argument and then the same key word arguments as you would pass
to a class constructor. Also, you don't need to add the objects to a session after you've created
them using ``as_unique`` method (it's done behind the scenes).

.. code:: python

    ku = DataSource.as_unique(session, short_name="ku")
    session.query(DataSource).filter(DataSource.short_name=="ku").one()


.. parsed-literal::

    <Data Source: ku>


This way you won't create multiple records in the database for the same data source.
For example, lets try to create another NIST data source:

.. code:: python

    nist2 = DataSource.as_unique(session, short_name="nist")
    assert nist2 is nist

To create new units use the ``astropy.units`` module. The mapping class is named ``UnitDB`` so it's
distinguishable from the ``astropy.units.Unit`` class. You should use ``as_unique()`` to create new
units for the same reasons as with data sources:

.. code:: python

    from astropy import units as u
    from carsus.model import UnitDB
    u_u = UnitDB.as_unique(session, unit=u.u)

Works with complex units as well:

.. code:: python

    u_complex = UnitDB.as_unique(session, unit=u.Unit("kg*m/s"))
    u_complex2 = UnitDB.as_unique(session, unit=u.Unit("meter*kilogram*s**-1"))
    assert u_complex is u_complex2



To build more interesting queries lets create new atomic quantites from another
data source. You should use the ``merge_quantity()`` helper method to create new quantities.
Basically, it works the same way as the ``as_unique()`` method: queries the database first and either returns
the existing quantity or creates a new one.

.. code:: python


    atomic_weights = [(1, 1.00769), (2, 4.0033), (3, 6.987), (4, 9.012), (5, 10.733), (14, 28.095)]
    for atomic_number, value in atomic_weights:
        atom = session.query(Atom).filter(Atom.atomic_number == atomic_number).one()
        atom.merge_quantity(session, AtomicWeight(data_source=ku, unit_db=u_u, value=value))
    session.commit()

Let's see what we got now:

.. code:: python

    q = session.query(Atom, AtomicWeight.value, DataSource.short_name).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        join(AtomicWeight.data_source)

    for atom, value, short_name in q.all()[:20]:
        print atom, value, short_name


.. parsed-literal::

    <Atom H, Z=1> 1.007975 nist
    <Atom H, Z=1> 1.00769 ku
    <Atom He, Z=2> 4.002602 nist
    <Atom He, Z=2> 4.0033 ku
    <Atom Li, Z=3> 6.9675 nist
    <Atom Li, Z=3> 6.987 ku
    <Atom Be, Z=4> 9.0121831 nist
    <Atom Be, Z=4> 9.012 ku
    <Atom B, Z=5> 10.8135 nist
    <Atom B, Z=5> 10.733 ku
    <Atom C, Z=6> 12.0106 nist
    <Atom N, Z=7> 14.006855 nist
    <Atom O, Z=8> 15.9994 nist
    <Atom F, Z=9> 18.998403163 nist
    <Atom Ne, Z=10> 20.1797 nist
    <Atom Na, Z=11> 22.98976928 nist
    <Atom Mg, Z=12> 24.3055 nist
    <Atom Al, Z=13> 26.9815385 nist
    <Atom Si, Z=14> 28.085 nist
    <Atom Si, Z=14> 28.095 ku


Imagine that the ku source is better than the nist and we want to use
it whenever it's available. We first define define our rating using the CASE statement.
Then we use the min function to select the best source for each atom. Records in a
GROUP BY query are guaranteed to come from the record in a group that matches a
MIN for that group.

.. code:: python

    from sqlalchemy import case, func

    stmt = case([
            (DataSource.short_name == "ku", 1),
            (DataSource.short_name == "nist", 2)
        ])

    q = session.query(Atom, AtomicWeight.value, DataSource.short_name, func.min(stmt)).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        join(AtomicWeight.data_source).\
        group_by(Atom.atomic_number)

    for atom, value, short_name, t in q.all()[:20]:
        print atom, value, short_name, t


.. parsed-literal::

    <Atom H, Z=1> 1.00769 ku 1
    <Atom He, Z=2> 4.0033 ku 1
    <Atom Li, Z=3> 6.987 ku 1
    <Atom Be, Z=4> 9.012 ku 1
    <Atom B, Z=5> 10.733 ku 1
    <Atom C, Z=6> 12.0106 nist 2
    <Atom N, Z=7> 14.006855 nist 2
    <Atom O, Z=8> 15.9994 nist 2
    <Atom F, Z=9> 18.998403163 nist 2
    <Atom Ne, Z=10> 20.1797 nist 2
    <Atom Na, Z=11> 22.98976928 nist 2
    <Atom Mg, Z=12> 24.3055 nist 2
    <Atom Al, Z=13> 26.9815385 nist 2
    <Atom Si, Z=14> 28.095 ku 1
    <Atom P, Z=15> 30.973761998 nist 2
    <Atom S, Z=16> 32.0675 nist 2
    <Atom Cl, Z=17> 35.4515 nist 2
    <Atom Ar, Z=18> 39.948 nist 2
    <Atom K, Z=19> 39.0983 nist 2
    <Atom Ca, Z=20> 40.078 nist 2

