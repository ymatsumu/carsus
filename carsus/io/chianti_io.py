import chianti.core as ch
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
from carsus.model import DataSource, Ion, Level, LevelEnergy

if os.getenv('XUVTOP'):
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

    def __init__(self, ion_name):

        self.ion = ch.ion(ion_name)
        self._levels_df = None

    @property
    def levels_df(self):
        if self._levels_df is None:
            self._read_levels()
        return self._levels_df

    @property
    def last_bound_level(self):
        ionization_potential = u.eV.to(u.Unit("cm-1"), value=self.ion.Ip, equivalencies=u.spectral())
        last_row = self.levels_df.loc[self.levels_df['energy'] < ionization_potential].tail(1)
        return last_row.index[0]

    @property
    def bound_levels_df(self):
        return self.levels_df.loc[:self.last_bound_level]

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


class ChiantiIngester(object):
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

    def get_index2id_df(self, ion):
        """ Return a DataFrame that maps levels indexes to ids """

        q_ion_lvls = self.session.query(Level.id.label("id"),
                                        Level.index.label("index")). \
                                  filter(and_(Level.ion == ion,
                                              Level.data_source == self.ds))

        index2id_df = read_sql_query(q_ion_lvls.selectable, self.session.bind,
                                       index_col="index")

        return index2id_df

    def ingest_levels(self):

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion -1

            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            # ToDo: Determine parity from configuration

            for index, row in rdr.bound_levels_df.iterrows():

                level = Level(ion=ion, data_source=self.data_source, level_index=index,
                                     configuration=row["configuration"], term=row["term"],
                                     L=row["L"], J=row["J"], spin_multiplicity=row["spin_multiplicity"])

                level.energies = []
                for column, method in [('energy', 'meas'), ('energy_theoretical', 'theor')]:
                    if row[column] != -1:  # check if the value exists
                        level.energies.append(
                            LevelEnergy(quantity=row[column] * u.Unit("cm-1"), method=method),
                        )
                self.session.add(level)

    def ingest(self, levels=True, lines=False, collisions=False):

        if levels:
            self.ingest_levels()

