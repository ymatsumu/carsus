from pandas import read_sql_query
from abc import ABCMeta
from carsus.model import Atom, AtomWeight, Ion, IonizationEnergy,\
    Line, LineWavelength, LineGFValue, LineAValue, Level, DataSource
from sqlalchemy import and_, or_, tuple_, not_, union_all, literal
from sqlalchemy.orm import aliased, joinedload, subqueryload
from sqlalchemy.orm.exc import NoResultFound
from astropy import constants
from tardis.util import species_string_to_tuple
import numpy as np
import pandas as pd


def create_basic_atom_df(session, max_atomic_number=30):
    """
        Create a DataFrame with basic atomic data.
        Parameters
        ----------
        session : SQLAlchemy session
        max_atomic_number: int
            The maximum atomic number to be stored in basic_atom_df

        Returns
        -------
        basic_atom_df : pandas.DataFrame
           DataFrame with columns: atomic_number, symbol, name, weight[u]
    """
    basic_atom_q = session.query(Atom).\
        filter(Atom.atomic_number <= max_atomic_number).\
        order_by(Atom.atomic_number)

    basic_atom_data = list()
    for atom in basic_atom_q.options(joinedload(Atom.weights)):
        weight = atom.weights[0].quantity.value if atom.weights else None  # Get the first weight from the collection
        basic_atom_data.append((atom.atomic_number, atom.symbol, atom.name, weight))

    basic_atom_dtype = [("atomic_number", np.int), ("symbol", "|S5"), ("name", "|S150"),
                        ("weight", np.float)]
    basic_atom_data = np.array(basic_atom_data, dtype=basic_atom_dtype)

    basic_atom_df = pd.DataFrame.from_records(basic_atom_data, index="atomic_number")

    return basic_atom_df


def create_ionization_df(session):
    """
        Create a DataFrame with ionization data.

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        ionization_df : pandas.DataFrame
           DataFrame with columns: atomic_number, ion_number, ionization_energy[eV]
    """
    ionization_q = session.query(Ion).\
        order_by(Ion.atomic_number, Ion.ion_charge)

    ionization_data = list()
    for ion in ionization_q.options(joinedload(Ion.ionization_energies)):
        ionization_energy = ion.ionization_energies[0].quantity.value if ion.ionization_energies else None
        ionization_data.append((ion.atomic_number, ion.ion_number, ionization_energy))

    ionization_dtype = [("atomic_number", np.int), ("ion_number", np.int), ("ionization_energy", np.float)]
    ionization_data = np.array(ionization_data, dtype=ionization_dtype)

    ionization_df = pd.DataFrame.from_records(ionization_data, index=["atomic_number", "ion_number"])

    return ionization_df


def create_levels_df(session, chianti_species=None, chianti_short_name=None, kurucz_short_name=None,
                     create_metastable_flags=True, metastable_loggf_threshold=-3):
    """
        Create a DataFrame with levels data.
        Parameters
        ----------
        session : SQLAlchemy session
        chianti_species: list of str in format <element_symbol> <ion_number>, eg. Fe 2
            The levels data for these ions will be taken from the CHIANTI database
            (default: None)
        chianti_short_name: str
            The short name of the CHIANTI database, is set to None the latest version will be used
            (default: None)
        kurucz_short_name: str
            The short name of the Kurucz database, is set to None the latest version will be used
            (default: None)
        create_metastable_flags: bool
            Create the `metastable` column containing flags for metastable levels (levels that take a long time to de-excite)
            (default: True)
        metastable_loggf_threshold: int
            log(gf) threshold for flagging metastable levels
            (default: -3)
        Returns
        -------
        levels_df : pandas.DataFrame
            DataFrame with columns: atomic_number, ion_number, level_number, energy[eV], g[1], metastable
    """

    if chianti_short_name is None:
        chianti_short_name = "chianti_v8.0.2"

    if kurucz_short_name is None:
        kurucz_short_name = "ku_latest"

    try:
        ch_ds = session.query(DataSource).filter(DataSource.short_name == chianti_short_name).one()
        ku_ds = session.query(DataSource).filter(DataSource.short_name == kurucz_short_name).one()
    except NoResultFound:
        print "Requested data sources does not exist!"
        raise

    # Get a list of tuples (atomic_number, ion_charge) for the chianti species
    chianti_species = [tuple(species_string_to_tuple(species_str)) for species_str in chianti_species]

    # To select ions we create a CTE (Common Table Expression), because sqlite doesn't support composite IN statements
    chianti_species_cte = union_all(
        *[session.query(
            literal(atomic_number).label("atomic_number"),
            literal(ion_charge).label("ion_charge"))
          for atomic_number, ion_charge in chianti_species]
    ).cte("chianti_species_cte")

    # To select chianti ions we join on the CTE
    chianti_levels_q = session.query(Level).\
        join(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                       Level.ion_charge == chianti_species_cte.c.ion_charge)).\
        filter(Level.data_source == ch_ds)

    # To select kurucz ions we do an outerjoin on the CTE and select rows that don't have a match from the CTE
    kurucz_levels_q = session.query(Level).\
        outerjoin(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                       Level.ion_charge == chianti_species_cte.c.ion_charge)).\
        filter(chianti_species_cte.c.atomic_number.is_(None)).\
        filter(Level.data_source == ku_ds)

    levels_q = kurucz_levels_q.union(chianti_levels_q)

    # Get the levels data
    levels_data = list()
    for lvl in levels_q.options(joinedload(Level.energies)):
        try:
            energy = None
            # Try to find the measured energy for this level
            for nrg in lvl.energies:
                if nrg.method == "meas":
                    energy = nrg.quantity
                    break
            # If the measured energy is not available, try to get the first one
            if energy is None:
                energy = lvl.energies[0].quantity
        except IndexError:
            print "No energy is available for level {0}".format(lvl.level_id)
            continue
        levels_data.append((lvl.level_id, lvl.atomic_number, lvl.ion_charge, energy.value, lvl.g, lvl.data_source_id))

    # Create a dataframe with the levels data
    levels_dtype = [("level_id", np.int), ("atomic_number", np.int),
                    ("ion_charge", np.int), ("energy", np.float), ("g", np.int), ("ds_id", np.int)]
    levels_data = np.array(levels_data, dtype=levels_dtype)
    levels_df = pd.DataFrame.from_records(levels_data, index="level_id")

    # Replace ion_charge with ion_number in the spectroscopic notation
    levels_df["ion_number"] = levels_df["ion_charge"] + 1
    levels_df.drop("ion_charge", axis=1, inplace=True)

    # Create level numbers
    levels_df.sort_values(["atomic_number", "ion_number", "energy", "g"], inplace=True)
    levels_df["level_number"] = levels_df.groupby(['atomic_number', 'ion_number'])['energy']. \
        transform(lambda x: np.arange(len(x))).values
    levels_df["level_number"] = levels_df["level_number"].astype(np.int)

    if create_metastable_flags:
        # Create metastable flags
        # ToDO: It is assumed that all lines are ingested. That may not always be the case

        levels_subq = session.query(Level).\
            filter(Level.level_id.in_(levels_df.index.values)).subquery()
        metastable_q = session.query(Line).\
            join(levels_subq, Line.upper_level)

        metastable_data = list()
        for line in metastable_q.options(joinedload(Line.gf_values)):
            try:
                # Currently it is assumed that each line has only one gf value
                gf = line.gf_values[0].quantity  # Get the first gf value
            except IndexError:
                print "No gf value is available for line {0}".format(line.line_id)
                continue
            metastable_data.append((line.line_id, line.upper_level_id, gf))

        metastable_dtype = [("line_id", np.int), ("upper_level_id", np.int), ("gf", np.float)]
        metastable_data = np.array(metastable_data, dtype=metastable_dtype)
        metastable_df = pd.DataFrame.from_records(metastable_data, index="line_id")

        # Filter loggf on the threshold value
        metastable_df["loggf"] = np.log10(metastable_df["gf"])
        metastable_df = metastable_df.loc[metastable_df["loggf"] > metastable_loggf_threshold]

        # Count the remaining strong transitions
        metastable_df_grouped = metastable_df.groupby("upper_level_id")
        metastable_flags = metastable_df_grouped["upper_level_id"].count()
        metastable_flags.name = "metastable"

        # If there are no strong transitions for a level (the count is NaN) then the metastable flag is True
        # else (the count is a natural number) the metastable flag is False
        levels_df = levels_df.join(metastable_flags)
        levels_df["metastable"] = levels_df["metastable"].isnull()

    # Create multiindex
    levels_df.reset_index(inplace=True)
    levels_df.set_index(["atomic_number", "ion_number", "level_number"], inplace=True)

    return levels_df
