import pandas as pd
import numpy as np
import pickle
import os
import re
from pandas import read_sql_query
from numpy.testing import assert_almost_equal
from astropy import units as u
from sqlalchemy import and_
from sqlalchemy.orm.exc import NoResultFound
from carsus.io.base import IngesterError
from carsus.util import atomic_number2symbol
from carsus.model import DataSource, Ion, Level, LevelEnergy,\
    Line,LineGFValue, LineAValue, LineWavelength, \
    ECollision, ECollisionEnergy, ECollisionGFValue, ECollisionTempStrength

if os.getenv('XUVTOP'):
    import chianti.core as ch
    masterlist_ions_path = os.path.join(
        os.getenv('XUVTOP'), "masterlist", "masterlist_ions.pkl"
    )

    masterlist_ions_file = open(masterlist_ions_path, 'rb')
    masterlist_ions = pickle.load(masterlist_ions_file).keys()
    # Exclude the "d" ions for now
    masterlist_ions = [_ for _ in masterlist_ions
                       if re.match("[a-z]+_\d+", _)]

else:
    print "Chianti database is not installed!"
    masterlist_ions = list()


class ReaderError(ValueError):
    pass


class ChiantiIonReader(object):
    """
        Class for reading ion data from the CHIANTI database

        Attributes
        ----------
        ion: chianti.core.ion instance

        Methods
        -------
        levels_df
            Return a DataFrame with the data for ion's levels

        lines_df
            Return a DataFrame with the data for ion's lines

        collisions_df
            Return a DataFrame with the data for ion's electron collisions

        bound_levels_df
            Same as `levels_df`, but only for bound levels (with energy < ionization_potential)

        bound_lines_df
            Same as `lines_df`, but only for bound levels (with energy < ionization_potential)

        bound_collisions_df
            Same as `collisions_df`, but only for bound levels (with energy < ionization_potential)
    """

    elvlc_dict = {
        'lvl': 'level_index',
        'ecm': 'energy',  # cm-1
        'ecmth': 'energy_theoretical',  # cm-1
        'j': 'J',
        'spd': 'L',
        'spin': 'spin_multiplicity',
        'pretty': 'pretty',  # configuration + term
        'label': 'label'
    }

    wgfa_dict = {
        'avalue': 'a_value',
        'gf': 'gf_value',
        'lvl1': 'lower_level_index',
        'lvl2': 'upper_level_index',
        'wvl': 'wavelength'
    }

    scups_dict = {
        'btemp': 'temperatures',
        'bscups': 'collision_strengths',
        'gf': 'gf_value',
        'de': 'energy',  # Rydberg
        'lvl1': 'lower_level_index',
        'lvl2': 'upper_level_index',
        'ttype': 'ttype',  # BT92 Transition type
        'cups': 'cups'  # BT92 scaling parameter
    }

    def __init__(self, ion_name):

        self.ion = ch.ion(ion_name)
        self._levels_df = None
        self._lines_df = None
        self._collisions_df = None

    @property
    def levels_df(self):
        if self._levels_df is None:
            self._read_levels()
        return self._levels_df.copy()

    @property
    def lines_df(self):
        if self._lines_df is None:
            self._read_lines()
        return self._lines_df.copy()

    @property
    def collisions_df(self):
        if self._collisions_df is None:
            self._read_collisions()
        return self._collisions_df.copy()

    @property
    def last_bound_level(self):
        ionization_potential = u.eV.to(u.Unit("cm-1"), value=self.ion.Ip, equivalencies=u.spectral())
        last_row = self.levels_df.loc[self.levels_df['energy'] < ionization_potential].tail(1)
        return last_row.index[0]

    @property
    def bound_levels_df(self):
        return self.levels_df.loc[:self.last_bound_level]

    def filter_bound_transitions(self, transition_df):
        """ Filter transitions DataFrames on bound levels.

            The most succinct and accurate way to do this is to use slicing on multi index,
            but due to some bug in pandas out-of-range rows are included in the resulting DataFrame.
        """
        transition_df.reset_index(inplace=True)
        transition_df = transition_df.loc[transition_df["upper_level_index"] <= self.last_bound_level]
        transition_df.set_index(["lower_level_index", "upper_level_index"], inplace=True)
        transition_df.sort_index(inplace=True)
        return transition_df

    @property
    def bound_lines_df(self):
        bound_lines_df = self.filter_bound_transitions(self.lines_df)
        return bound_lines_df

    @property
    def bound_collisions_df(self):
        bound_collision_df = self.filter_bound_transitions(self.collisions_df)
        return bound_collision_df

    def _read_levels(self):

        if not hasattr(self.ion, 'Elvlc'):
            raise ReaderError("No levels data is available for ion {}".format(self.ion.Spectroscopic))

        levels_dict = {}

        for key, col_name in self.elvlc_dict.iteritems():
            levels_dict[col_name] = self.ion.Elvlc.get(key)

        # Check that ground level energy is 0
        try:
            for key in ['energy', 'energy_theoretical']:
                assert_almost_equal(levels_dict[key][0], 0)
        except AssertionError:
            raise ValueError('Level 0 energy is not 0.0')

        self._levels_df = pd.DataFrame(levels_dict)

        # Replace empty labels with NaN
        self._levels_df["label"].replace(r'\s+', np.nan, regex=True, inplace=True)

        # Extract configuration and term from the "pretty" column
        self._levels_df[["term", "configuration"]] = self._levels_df["pretty"].str.rsplit(' ', expand=True, n=1)
        self._levels_df.drop("pretty", axis=1, inplace=True)

        self._levels_df.set_index("level_index", inplace=True)
        self._levels_df.sort_index(inplace=True)

    def _read_lines(self):
        if not hasattr(self.ion, 'Wgfa'):
            raise ReaderError("No lines data is available for ion {}".format(self.ion.Spectroscopic))

        lines_dict = {}

        for key, col_name in self.wgfa_dict.iteritems():
            lines_dict[col_name] = self.ion.Wgfa.get(key)

        self._lines_df = pd.DataFrame(lines_dict)

        # two-photon transitions are given a zero wavelength and we ignore them for now
        self._lines_df = self._lines_df.loc[~(self._lines_df["wavelength"] == 0)]

        # theoretical wavelengths have negative values
        def parse_wavelength(row):
            if row["wavelength"] < 0:
                wvl = -row["wavelength"]
                method = "th"
            else:
                wvl = row["wavelength"]
                method = "m"
            return pd.Series([wvl, method])

        self._lines_df[["wavelength", "method"]] = self._lines_df.apply(parse_wavelength, axis=1)

        self._lines_df.set_index(["lower_level_index", "upper_level_index"], inplace=True)
        self._lines_df.sort_index(inplace=True)

    def _read_collisions(self):
        if not hasattr(self.ion, 'Scups'):
            raise ("No collision data is available for ion {}".format(self.ion.Spectroscopic))

        collisions_dict = {}

        for key, col_name in self.scups_dict.iteritems():
            collisions_dict[col_name] = self.ion.Scups.get(key)

        self._collisions_df = pd.DataFrame(collisions_dict)

        self._collisions_df.set_index(["lower_level_index", "upper_level_index"], inplace=True)
        self._collisions_df.sort_index(inplace=True)


class ChiantiIngester(object):
    """
        Class for ingesting data from the CHIANTI database

        Attributes
        ----------
        session: SQLAlchemy session

        data_source: DataSource instance
            The data source of the ingester

        ion_readers : list of ChiantiIonReader instances
            (default value = masterlist_ions)

        Methods
        -------
        ingest(session)
            Persists data into the database
    """

    masterlist_ions = masterlist_ions

    def __init__(self, session, ions_list=masterlist_ions, ds_short_name="chianti_v8.0.2"):
        self.session = session
        # ToDo write a parser for Spectral Notation
        self.ion_readers = list()
        for ion in ions_list:
            if ion in self.masterlist_ions:
                self.ion_readers.append(ChiantiIonReader(ion))
            else:
                print("Ion {0} is not available".format(ion))

        self.data_source = DataSource.as_unique(self.session, short_name=ds_short_name)
        if self.data_source.data_source_id is None:  # To get the id if a new data source was created
            self.session.flush()

    def get_lvl_index2id_df(self, ion):
        """ Return a DataFrame that maps levels indexes to ids """

        q_ion_lvls = self.session.query(Level.level_id.label("id"),
                                        Level.level_index.label("index")). \
            filter(and_(Level.ion == ion,
                        Level.data_source == self.data_source))

        lvl_index2id_data = list()
        for id, index in q_ion_lvls:
            lvl_index2id_data.append((index, id))

        lvl_index2id_dtype = [("index", np.int), ("id", np.int)]
        lvl_index2id_data = np.array(lvl_index2id_data, dtype=lvl_index2id_dtype)
        lvl_index2id_df = pd.DataFrame.from_records(lvl_index2id_data, index="index")

        return lvl_index2id_df

    def ingest_levels(self):

        print("Ingesting levels from {}".format(self.data_source.short_name))

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion -1

            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting levels for {} +{}".format(atomic_number2symbol(atomic_number), ion_charge))

            # ToDo: Determine parity from configuration

            for index, row in rdr.bound_levels_df.iterrows():

                level = Level(ion=ion, data_source=self.data_source, level_index=index,
                                     configuration=row["configuration"], term=row["term"],
                                     L=row["L"], J=row["J"], spin_multiplicity=row["spin_multiplicity"])

                level.energies = []
                for column, method in [('energy', 'meas'), ('energy_theoretical', 'theor')]:
                    if row[column] != -1:  # check if the value exists
                        level.energies.append(
                            LevelEnergy(quantity=row[column]*u.Unit("cm-1"),
                                        data_source=self.data_source,
                                        method=method),
                        )
                self.session.add(level)

    def ingest_lines(self):

        print("Ingesting lines from {}".format(self.data_source.short_name))

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion - 1

            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting lines for {} +{}".format(atomic_number2symbol(atomic_number), ion_charge))

            lvl_index2id_df = self.get_lvl_index2id_df(ion)

            for index, row in rdr.bound_lines_df.iterrows():

                # index: (lower_level_index, upper_level_index)
                lower_level_index, upper_level_index = index

                try:
                    lower_level_id = int(lvl_index2id_df.loc[lower_level_index])
                    upper_level_id = int(lvl_index2id_df.loc[upper_level_index])
                except KeyError:
                    raise IngesterError("Levels from this source have not been found."
                                        "You must ingest levels before transitions")

                # Create a new line
                line = Line(
                    lower_level_id=lower_level_id,
                    upper_level_id=upper_level_id,
                    data_source=self.data_source,
                    wavelengths=[
                        LineWavelength(quantity=row["wavelength"]*u.AA,
                                       data_source=self.data_source,
                                       method=row["method"])
                    ],
                    a_values=[
                        LineAValue(quantity=row["a_value"]*u.Unit("s**-1"),
                                   data_source=self.data_source)
                    ],
                    gf_values=[
                        LineGFValue(quantity=row["gf_value"],
                                    data_source=self.data_source)
                    ]
                )

                self.session.add(line)

    def ingest_collisions(self):

        print("Ingesting collisions from {}".format(self.data_source.short_name))

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion - 1

            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting collisions for {} +{}".format(atomic_number2symbol(atomic_number), ion_charge))

            lvl_index2id_df = self.get_lvl_index2id_df(ion)

            for index, row in rdr.bound_collisions_df.iterrows():

                # index: (lower_level_index, upper_level_index)
                lower_level_index, upper_level_index = index

                try:
                    lower_level_id = int(lvl_index2id_df.loc[lower_level_index])
                    upper_level_id = int(lvl_index2id_df.loc[upper_level_index])
                except KeyError:
                    raise IngesterError("Levels from this source have not been found."
                                        "You must ingest levels before transitions")

                # Create a new electron collision
                e_col = ECollision(
                    lower_level_id=lower_level_id,
                    upper_level_id=upper_level_id,
                    data_source=self.data_source,
                    bt92_ttype=row["ttype"],
                    bt92_cups=row["cups"],
                    energies=[
                        ECollisionEnergy(quantity=row["energy"]*u.rydberg,
                                         data_source=self.data_source)
                    ],
                    gf_values=[
                        ECollisionGFValue(quantity=row["gf_value"],
                                          data_source=self.data_source)
                    ]
                )

                e_col.temp_strengths = [
                    ECollisionTempStrength(temp=temp, strength=strength)
                    for temp, strength in zip(row["temperatures"], row["collision_strengths"])
                    ]

                self.session.add(e_col)


    def ingest(self, levels=True, lines=False, collisions=False):

        if levels:
            self.ingest_levels()
            self.session.flush()

        if lines:
            self.ingest_lines()
            self.session.flush()

        if collisions:
            self.ingest_collisions()
            self.session.flush()
