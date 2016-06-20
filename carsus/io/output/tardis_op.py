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


class BaseData(object):
    """
        Base class for creating DataFrames with atomic data.

        Parameters
        -----------
        session : SQLAlchemy session

        Attributes
        -----------
        session : SQLAlchemy session=
        config : dict
            The configuration parameters

    """
    __metaclass__ = ABCMeta

    def __init__(self, session, **kwargs):
        self.session = session
        self.config = kwargs


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


class BasicAtomData(BaseData):
    """
        Class for creating DataFrames with basic atomic data.

        Parameters
        -----------
        session : SQLAlchemy session
        max_atomic_number : int
            The maximum atomic number to be stored in basic_atom_df

        Attributes
        -----------
        session : SQLAlchemy session
        config : dict
            The configuration parameters
        basic_atom_df : pandas.DataFrame
           DataFrame with columns: atomic_number, symbol, name, weight[u]
    """
    def __init__(self, session, max_atomic_number=30):
        self._basic_atom_df = None
        super(BasicAtomData, self).__init__(session, max_atomic_number=max_atomic_number)

    @property
    def basic_atom_df(self):
        if self._basic_atom_df is None:
            self._basic_atom_df = create_basic_atom_df(self.session, **self.config)
        return self._basic_atom_df


def create_ionization_df(session):
    """
        Create a DataFrame with ionization data.

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        ionization_data_df : pandas.DataFrame
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


class IonData(BaseData):
    """
        Class for creating DataFrames with ion data

        Parameters
        -----------
        session : SQLAlchemy session

        Attributes
        -----------
        session : SQLAlchemy session
        config : dict
            The configuration parameters
        ionization_data_df : pandas.DataFrame
           DataFrame with columns: atomic_number, ion_number, ionization_energy[eV]
    """
    def __init__(self, session):
        self._ionization_df = None
        super(IonData, self).__init__(session)

    @property
    def ionization_df(self):
        if self._ionization_df is None:
            self._ionization_df = create_ionization_df(self.session, **self.config)
        return self._ionization_df