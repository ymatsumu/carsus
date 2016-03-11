Commit Note 1
=============

Grammar revisited
-----------------

The output formats are available for the `NIST Atomic Weights and Isotopic Compositions <http://www.nist.gov/pml/data/comp.cfm>`_
database: HTML table, preformatted ASCII table and linearized ASCII output. I wrote the new grammar for the
linearized output because it is the easiest one to parse. I won't go into much details here,
refer to the `compositions_grammar <https://github.com/mishinma/carsus/blob/nist_comp/carsus/io/nist/grammars/compositions_grammar.py>`_
module.
The new idea is to use pyparsing ParseActions to extract all needed information from the source and then create a
Base DataFrame from the parsed results. The Base DataFrame columns correspond to the parsed results named tokens.
Consider this isotope entry::

    Atomic Number = 18
    Atomic Symbol = Ar
    Mass Number = 38
    Relative Atomic Mass = 37.96273211(21)
    Isotopic Composition = 0.000629(7)
    Standard Atomic Weight = 39.948(1)
    Notes = g,r

If you parse it with the ``isotope`` grammar element and inspect the nested tokens list with the ``dump()`` function
you will see this output::

    - atomic_mass: ['37.96273211', '(', '21', ')']
        - nominal_value: 37.96273211
        - std_dev: 2.1e-07
        - theoretical: False
    - atomic_number: 18
    - atomic_weight: ['39.948', '(', '1', ')']
        - nominal_value: 39.948
        - std_dev: 0.001
        - type: 0
    - isotopic_comp: ['0.000629', '(', '7', ')']
        - nominal_value: 0.000629
        - std_dev: 7e-06
    - mass_number: 38
    - notes: g r
    - symbol: Ar

The Base DataFrame row that contains this isotope  would look like this::

    symbol                                    Ar
    atomic_mass_nominal_value            37.9627
    atomic_mass_std_dev                  2.1e-07
    atomic_mass_theoretical                False
    isotopic_comp_nominal_value         0.000629
    isotopic_comp_std_dev                  7e-06
    atomic_weight_type                         0
    atomic_weight_nominal_value           39.948
    atomic_weight_std_dev                  0.001
    atomic_weight_lwr_bnd                    NaN
    atomic_weight_upr_bnd                    NaN
    atomic_weight_stable_mass_number         NaN
    notes                                    g r

This columns are generated from the ParsedResults object using the ``to_flat_dict`` function from the
`utils <https://github.com/mishinma/carsus/blob/nist_comp/carsus/io/util.py>`_ module.

So, the idea here is to do additional text processing while parsing. A nice example is the ``ufloat`` parse element; its parse
action uses the `<uncertainties <http://pythonhosted.org/uncertainties/index.html>`_ package to extract nominal values
and standard deviations from strings like "1.2345(23)".

The BasePyparser Class
----------------------

All parsers that use pyparsing grammar inherit from this class defined in the
`io.base <https://github.com/mishinma/carsus/blob/nist_comp/carsus/io/base.py>`_ module.
You should know about three attributes of the instances of this class: ``grammar``, ``columns`` and ``base_df`` (The Base DataFrame).
The grammar is a pyparsing.ParserElement object that will be used to scan input. The ``columns`` are the columns name
of the ``base_df``; remember that they are closely related to the named tokens of the grammar element (they specify the order of
the columns in the ``base_df``). Now, ``base_df`` is very important - the parser will use it to prepare other dataframes for
ingestion. You have already seen the columns of the ``base_df`` for the NIST Atomic Weights and Isotopic Compositions database.
You can notice that it contains data related to atoms (like atomic weight) and data related to isotopes (like isotopic compostions).
The ``base_df`` will be used by an instance of the ``NISTCompositionsPyparser`` class
(from the `compositions <https://github.com/mishinma/carsus/blob/nist_comp/carsus/io/nist/compositions.py>`_ module)
when it is asked to prepare ``atomic_df`` with the atomic related data or ``isotopes_df`` with
the isotopes related data. For example, the  ``atomic_df`` has the following structure::


                   atomic_weight_nominal_value  atomic_weight_std_dev
    atomic_number
    35                               79.904000               0.003000
    38                               87.620000               0.010000
    43                               97.907212               0.000004

Note that atomic weights of different types (nominal value and standard deviation, interval or stable isotope
mass number) are all converted to the first type.

Database structure
------------------

I have indicated some problems with the current database structure in `this <https://www.overleaf.com/4487510sdycrg>`_
document. I will describe the proposed structure there in more detail soon. Here is a brief into.
The sqlalchemy classes are defined in the
`alchemy.atomic <https://github.com/mishinma/carsus/blob/nist_comp/carsus/alchemy/atomic.py>` module.
The ``Atom`` class has fields that describes an atom and are *not* data source depended. I have chosen an
atomic number as a primary key instead of a surrogate primary key. The ``DataSource`` class is also
straightforward. For storing information about data dependent atomic quantities the ``AtomicQuantity`` class is used.
It is a base class for all other atomic quantities, so the ``AtomicWeight`` class inherit from it. This is
single table inheritance scheme that is described `here <http://docs.sqlalchemy.org/en/latest/orm/inheritance.html#single-table-inheritance>`_.
Refer to the tests in the `test_atomic <https://github.com/mishinma/carsus/blob/nist_comp/carsus/alchemy/tests/test_atomic.py>`_
to see some queries. Also note the Unique constrain in the ``AtomicQuantity`` class. There cannot be two different quantites of the
same type from the same datasource related to the same atom.

The BaseIngester Class
-----------------------

This is an abstract class for all ingesters defined in the `io.base <https://github.com/mishinma/carsus/blob/nist_comp/carsus/io/base.py>`_
module. Some methods and properties must be overridden to instantiate this class. It is easier to explain things using the
``NISTCompositionsIngester`` class from the `compositions <https://github.com/mishinma/carsus/blob/nist_comp/carsus/io/nist/compositions.py>`_ module.
Instances of this class have again three important attributes: ``atomic_db``, ``parser`` and ``downloader``.
``atomic_db`` is an instance of the ``AtomicDatabase`` class from the `base <https://github.com/mishinma/carsus/blob/nist_comp/carsus/base.py>`_ module.
It contains the session and all. ``parser`` is an instance of the familiar ``NISTCompositionsPyparser`` class which is
initially not loaded with data (so its ``base_df`` is empty). You download data with  ``downloader`` function and then
load ``parser`` with it. Now you are ready to ingest data to the database. For now the implementation only will ingest atomic data.
There is an important aspect: all general atomic data that is not related to the database (atomic number, symbol and all)
is loaded automatically from `this <https://github.com/mishinma/carsus/blob/nist_comp/carsus/data/basic_atomic_data.csv>` file.
The only data that is going to be ingested is the atomic weights data. The ingester will check if there is already atomic data from
this source and update it appropriately.

Next Steps
----------
1. command line arguments with the argparse module
2. ingest isotopic data

Please give feedback!





