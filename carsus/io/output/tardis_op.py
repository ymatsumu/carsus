from pandas import read_sql_query
from carsus.model import Atom, AtomWeight, Ion, IonizationEnergy,\
    Line, LineWavelength, LineGFValue, LineAValue, Level, DataSource
from sqlalchemy import and_, or_, tuple_, not_, union_all, literal
from sqlalchemy.orm import aliased, joinedload, subqueryload
from sqlalchemy.orm.exc import NoResultFound
from astropy import constants
from tardis.util import species_string_to_tuple
import numpy as np
import pandas as pd


def create_basic_atom_df(session):
    """
        This function creates a DataFrame with basic atomic data.

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        basic_atom_data : pandas.DataFrame
           DataFrame with columns: atomic_number, symbol, name, weight[u]
    """
    basic_atom_q = session.query(Atom).\
        options(joinedload(Atom.weights))

    basic_atom_data = list()
    for atom in basic_atom_q:
        weight = atom.weights[0].quantity.value if atom.weights else None  # Get the first weight from the collection
        basic_atom_data.append((atom.atomic_number, atom.symbol, atom.name, weight))

    basic_atom_dtype = [("atomic_number", np.int), ("symbol", "|S5"), ("name", "|S150"),
                        ("weight", np.float)]
    basic_atom_data = np.array(basic_atom_data, dtype=basic_atom_dtype)

    basic_atom_df = pd.DataFrame.from_records(basic_atom_data, index="atomic_number")

    return basic_atom_df


def create_ionization_df(session):
    """
        This function created a DataFrame with ionization data.

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        ionization_data_df : pandas.DataFrame
           DataFrame with columns: atomic_number, ion_number, ionization_energy[eV]
    """
    ionization_q = session.query(Ion).\
        options(joinedload(Ion.ionization_energies))

    ionization_data = list()
    for ion in ionization_q:
        ionization_energy = ion.ionization_energies[0].quantity.value if ion.ionization_energies else None
        ionization_data.append((ion.atomic_number, ion.ion_number, ionization_energy))

    ionization_dtype = [("atomic_number", np.int), ("ion_number", np.int), ("ionization_energy", np.float)]
    ionization_data = np.array(ionization_data, dtype=ionization_dtype)

    ionization_df = pd.DataFrame.from_records(ionization_data, index=["atomic_number", "ion_number"])

    return ionization_df


def create_levels_df(session, chianti_species=None, chianti_short_name=None, kurucz_short_name=None):
    """
        This function creates a DataFrame with levels data.

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

    chianti_species_cte = union_all(
        *[session.query(
            literal(atomic_number).label("atomic_number"),
            literal(ion_charge).label("ion_charge"))
          for atomic_number, ion_charge in chianti_species]
    ).cte("chianti_species_cte")

    chianti_levels_q = session.query(Level).\
        join(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                       Level.ion_charge == chianti_species_cte.c.ion_charge)).\
        filter(Level.data_source == ch_ds)

    kurucz_levels_q = session.query(Level).\
        outerjoin(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                       Level.ion_charge == chianti_species_cte.c.ion_charge)).\
        filter(chianti_species_cte.c.atomic_number.is_(None)).\
        filter(Level.data_source == ku_ds)

    levels_q = kurucz_levels_q.union(chianti_levels_q)

    levels_data = list()
    for lvl in levels_q.options(joinedload(Level.energies)):
        try:
            energy = lvl.energies[0].quantity
        except IndexError:
            print "No energy is available for level {0}".format(lvl.level_id)
            continue
        levels_data.append((lvl.level_id, lvl.atomic_number, lvl.ion_charge, energy.value, lvl.g, lvl.data_source_id))

    levels_dtype = [("level_id", np.int), ("atomic_number", np.int),
                    ("ion_charge", np.int), ("energy", np.float), ("g", np.int), ("ds_id", np.int)]
    levels_data = np.array(levels_data, dtype=levels_dtype)

    levels_df = pd.DataFrame.from_records(levels_data, index="level_id")

    # Create metastable flags
    # ToDO: It is assumed that all lines are ingested. That may not always be the case

    metastable_lines_data = list()

    levels_subq = levels_q.subquery()
    metastable_lines_q = session.query(Line).\
        join(levels_subq, Line.upper_level)

    for line in metastable_lines_q.options(joinedload(Line.gf_values)):
        try:
            gf = line.gf_values[0].quantity  # Get the first gf value
        except IndexError:
            print "No gf value is available for line {0}".format(line.line_id)
            continue
        metastable_lines_data.append((line.line_id, line.upper_level_id, gf))

    metastable_lines_dtype = [("line_id", np.int), ("upper_level_id", np.int), ("gf", np.float)]
    metastable_lines_data = np.array(metastable_lines_data, dtype=metastable_lines_dtype)

    metastable_lines_df = pd.DataFrame.from_records(metastable_lines_data, index="line_id")

    metastable_lines_df["loggf"] = np.log10(metastable_lines_df["gf"])

    import pdb; pdb.set_trace()


    return levels_df


def create_lines_df(session):
    """
        This function creates a DataFrame with lines data.

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------

        lines_df: pandas.DataFrame
            DataFrame with columns: wavelength[angstrom], atomic_number, ion_number, f_ul, f_lu, level_id_lower, level_id_upper.
    """

    lines_q = session.query(Line).\
        options(subqueryload(Line.lower_level)).\
        options(subqueryload(Line.upper_level)).\
        options(joinedload(Line.wavelengths)).\
        options(joinedload(Line.gf_values))

    line_data = list()
    for line in lines_q:
        try:
            wavelength = line.wavelengths[0].quantity
            gf = line.gf_values[0].quantity  # g_lf_lu
        except IndexError:
            print "No wavelength or gf value for line {0}".format(line.line_id)
            continue

        atomic_number = line.lower_level.atomic_number
        ion_number = line.lower_level.ion_charge + 1  # the hybrid property ion_number is not available here

        line_data.append((line.line_id, wavelength, gf, atomic_number, ion_number,
                          line.lower_level.g, line.upper_level.g))


class AtomData(object):
    """
    Class for storing atomic data


    Attributes
    ----------

    basic_atom_df : pandas.DataFrame
        containing the basic atom data: atomic_number[1], symbol, name, weight[u]

    ionization_df : pandas.DataFrame
        containing the ionization data: atomic_number[1], ion_number[1], ionization_energy[eV]
        ::important to note here that ion_number is ionization state in spectroscopic notation,
            e.g. for Fe+ (singly ionized Fe) the spectroscopic notation is Fe II and ion_number=2

    levels_df : pandas.DataFrame
        containing the levels data: atomic_number, ion_number, level_number, energy[eV], g

    lines_df : pandas.DataFrame
        containing the lines data: wavelength, z, ion, levels_number_lower,
        levels_number_upper, f_lu, f_ul
    """

    @classmethod
    def from_database(cls, session):

        basic_atom_df = create_basic_atom_df(session)
        ionization_df = create_ionization_df(session)

        return cls(basic_atom_df=basic_atom_df, ionization_df=ionization_df)

    def __init__(self, basic_atom_df=None, ionization_df=None, levels_df=None, lines_df=None):

        self.basic_atom_df = basic_atom_df
        self.ionization_df = ionization_df
        self.levels_df = levels_df
        self.lines_df = lines_df