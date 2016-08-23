import re
import numpy as np
import pandas as pd

from astropy import units as u
from sqlalchemy import and_
from pyparsing import ParseException
from carsus.model import DataSource, Ion, Level, LevelEnergy,\
    Line, LineWavelength, LineGFValue, MEDIUM_VACUUM, MEDIUM_AIR
from carsus.io.base import IngesterError
from carsus.util import convert_atomic_number2symbol, parse_selected_species


GFALL_AIR_THRESHOLD = 200  # [nm], wavelengths above this value are given in air


class GFALLReader(object):
    """
        Class for extracting lines and levels data from kurucz gfall files

        Attributes
        ----------
        fname: path to gfall.dat

        Methods
        --------
        gfall_raw:
            Return pandas DataFrame representation of gfall

    """
    def __init__(self, fname):
        self.fname = fname
        self._gfall_raw = None
        self._gfall = None
        self._levels = None
        self._lines = None

    @property
    def gfall_raw(self):
        if self._gfall_raw is None:
            self._gfall_raw = self.read_gfall_raw()
        return self._gfall_raw

    @property
    def gfall(self):
        if self._gfall is None:
            self._gfall = self.parse_gfall()
        return self._gfall

    @property
    def levels(self):
        if self._levels is None:
            self._levels = self.extract_levels()
        return self._levels

    @property
    def lines(self):
        if self._lines is None:
            self._lines = self.extract_lines()
        return self._lines

    def read_gfall_raw(self, fname=None):
        """
        Reading in a normal gfall.dat

        Parameters
        ----------
        fname: ~str
            path to gfall.dat

        Returns
        -------
            pandas.DataFrame
                pandas Dataframe represenation of gfall
        """

        if fname is None:
            fname = self.fname

        # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
        # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)

        kurucz_fortran_format = ('F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,'
                                 'A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,I5,I5,'
                                 '1X,I1,A1,1X,I1,A1,I1,A3,I5,I5,I6')

        number_match = re.compile(r'\d+(\.\d+)?')
        type_match = re.compile(r'[FIXA]')
        type_dict = {'F': np.float64, 'I': np.int64, 'X': 'S1', 'A': 'S10'}
        field_types = tuple([type_dict[item] for item in number_match.sub(
            '', kurucz_fortran_format).split(',')])

        field_widths = type_match.sub('', kurucz_fortran_format)
        field_widths = map(int, re.sub(r'\.\d+', '', field_widths).split(','))

        def read_remove_empty(fname):
            """ Generator to remove empty lines from the gfall file"""
            with open(fname, "r") as f:
                for line in f:
                    if not re.match(r'^\s*$', line):
                        yield line

        gfall = np.genfromtxt(read_remove_empty(fname), dtype=field_types, delimiter=field_widths)

        columns = ['wavelength', 'loggf', 'element_code', 'e_first', 'j_first',
                   'blank1', 'label_first', 'e_second', 'j_second', 'blank2',
                   'label_second', 'log_gamma_rad', 'log_gamma_stark',
                   'log_gamma_vderwaals', 'ref', 'nlte_level_no_first',
                   'nlte_level_no_second', 'isotope', 'log_f_hyperfine',
                   'isotope2', 'log_iso_abundance', 'hyper_shift_first',
                   'hyper_shift_second', 'blank3', 'hyperfine_f_first',
                   'hyperfine_note_first', 'blank4', 'hyperfine_f_second',
                   'hyperfine_note_second', 'line_strength_class', 'line_code',
                   'lande_g_first', 'lande_g_second', 'isotopic_shift']

        gfall = pd.DataFrame(gfall)
        gfall.columns = columns

        return gfall

    def parse_gfall(self, gfall_raw=None):
        """
        Parse raw gfall DataFrame

        Parameters
        ----------
        gfall_raw: pandas.DataFrame

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """
        gfall = gfall_raw if gfall_raw is not None else self.gfall_raw.copy()
        
        double_columns = [item.replace('_first', '') for item in gfall.columns if
                          item.endswith('first')]

        # due to the fact that energy is stored in 1/cm
        order_lower_upper = (gfall["e_first"].abs() <
                             gfall["e_second"].abs())

        for column in double_columns:
            data = pd.concat([gfall['{0}_first'.format(column)][order_lower_upper],
                              gfall['{0}_second'.format(column)][~order_lower_upper]])

            gfall['{0}_lower'.format(column)] = data

            data = pd.concat([gfall['{0}_first'.format(column)][~order_lower_upper], \
                              gfall['{0}_second'.format(column)][order_lower_upper]])

            gfall['{0}_upper'.format(column)] = data

            del gfall['{0}_first'.format(column)]
            del gfall['{0}_second'.format(column)]

        # Clean labels
        gfall["label_lower"] = gfall["label_lower"].str.strip()
        gfall["label_upper"] = gfall["label_upper"].str.strip()

        gfall["label_lower"] = gfall["label_lower"].str.replace('\s+', ' ')
        gfall["label_upper"] = gfall["label_upper"].str.replace('\s+', ' ')

        # Ignore lines with the labels "AVARAGE ENERGIES" and "CONTINUUM"
        ignored_labels = ["AVERAGE", "ENERGIES", "CONTINUUM"]
        gfall = gfall.loc[~((gfall["label_lower"].isin(ignored_labels)) |
                            (gfall["label_upper"].isin(ignored_labels)))].copy()

        gfall['e_lower_predicted'] = gfall["e_lower"] < 0
        gfall["e_lower"] = gfall["e_lower"].abs()
        gfall['e_upper_predicted'] = gfall["e_upper"] < 0
        gfall["e_upper"] = gfall["e_upper"].abs()

        gfall['atomic_number'] = gfall.element_code.astype(int)
        gfall['ion_charge'] = ((gfall.element_code.values -
                                gfall.atomic_number.values) * 100).round().astype(int)

        del gfall['element_code']

        return gfall

    def extract_levels(self, gfall=None, selected_columns=None):
        """
        Extract levels from `gfall`

        Parameters
        ----------
        gfall: pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """

        if gfall is None:
            gfall = self.gfall

        if selected_columns is None:
            selected_columns = ['atomic_number', 'ion_charge', 'energy', 'j',
                                'label', 'theoretical']

        column_renames = {'e_{0}': 'energy', 'j_{0}': 'j', 'label_{0}': 'label',
                          'e_{0}_predicted': 'theoretical'}

        e_lower_levels = gfall.rename(
            columns=dict([(key.format('lower'), value)
                          for key, value in column_renames.items()]))

        e_upper_levels = gfall.rename(
            columns=dict([(key.format('upper'), value)
                          for key, value in column_renames.items()]))

        levels = pd.concat([e_lower_levels[selected_columns],
                            e_upper_levels[selected_columns]])

        levels = levels.sort_values(['atomic_number', 'ion_charge', 'energy', 'j', 'label']).\
            drop_duplicates(['atomic_number', 'ion_charge', 'energy', 'j', 'label'])

        levels["method"] = levels["theoretical"].\
            apply(lambda x: "theor" if x else "meas")  # Theoretical or measured
        levels.drop("theoretical", 1, inplace=True)

        levels["level_index"] = levels.groupby(['atomic_number', 'ion_charge'])['j'].\
            transform(lambda x: np.arange(len(x))).values
        levels["level_index"] = levels["level_index"].astype(int)

        # ToDo: The commented block below does not work with all lines. Find a way to parse it.
        # levels[["configuration", "term"]] = levels["label"].str.split(expand=True)
        # levels["configuration"] = levels["configuration"].str.strip()
        # levels["term"] = levels["term"].str.strip()

        levels.set_index(["atomic_number", "ion_charge", "level_index"], inplace=True)
        return levels

    def extract_lines(self, gfall=None, levels=None, selected_columns=None):
        """
        Extract lines from `gfall`

        Parameters
        ----------
        gfall: pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """
        if gfall is None:
            gfall = self.gfall

        if levels is None:
            levels = self.levels

        if selected_columns is None:
            selected_columns = ['wavelength', 'loggf', 'atomic_number', 'ion_charge']

        levels_idx = levels.reset_index()
        levels_idx = levels_idx.set_index(['atomic_number', 'ion_charge', 'energy', 'j', 'label'])

        lines = gfall[selected_columns].copy()
        lines["gf"] = np.power(10, lines["loggf"])
        lines = lines.drop(["loggf"], 1)

        level_lower_idx = gfall[['atomic_number', 'ion_charge', 'e_lower', 'j_lower', 'label_lower']].values.tolist()
        level_lower_idx = [tuple(item) for item in level_lower_idx]

        level_upper_idx = gfall[['atomic_number', 'ion_charge', 'e_upper', 'j_upper', 'label_upper']].values.tolist()
        level_upper_idx = [tuple(item) for item in level_upper_idx]

        lines['level_index_lower'] = levels_idx.loc[level_lower_idx, "level_index"].values
        lines['level_index_upper'] = levels_idx.loc[level_upper_idx, "level_index"].values

        lines.set_index(['atomic_number', 'ion_charge', 'level_index_lower', 'level_index_upper'], inplace=True)

        return lines


class GFALLIngester(object):
    """
        Class for ingesting data from kurucz dfall files

        Attributes
        ----------
        session: SQLAlchemy session
        fname: str
            The name of the gfall file to read
        ions: str
            Ingest levels and lines only for these ions. If set to None then ingest all.
            (default: None)
        data_source: DataSource instance
            The data source of the ingester

        gfall_reader : GFALLReaderinstance

        Methods
        -------
        ingest(session)
            Persists data into the database
    """
    def __init__(self, session, fname, ions=None, ds_short_name="ku_latest"):
        self.session = session
        self.gfall_reader = GFALLReader(fname)
        if ions is not None:
            try:
                ions = parse_selected_species(ions)
            except ParseException:
                raise ValueError('Input is not a valid species string {}'.format(ions))
            ions = pd.DataFrame.from_records(ions, columns=["atomic_number", "ion_charge"])
            self.ions = ions.set_index(['atomic_number', 'ion_charge'])
        else:
            self.ions = None

        self.data_source = DataSource.as_unique(self.session, short_name=ds_short_name)
        if self.data_source.data_source_id is None:  # To get the id if a new data source was created
            self.session.flush()

    def get_lvl_index2id(self, ion):
        """ Return a DataFrame that maps levels indexes to ids """

        q_ion_lvls = self.session.query(Level.level_id.label("id"),
                                        Level.level_index.label("index")). \
            filter(and_(Level.ion == ion,
                        Level.data_source == self.data_source))

        lvl_index2id = list()
        for id, index in q_ion_lvls:
            lvl_index2id.append((index, id))

        lvl_index2id_dtype = [("index", np.int), ("id", np.int)]
        lvl_index2id = np.array(lvl_index2id, dtype=lvl_index2id_dtype)
        lvl_index2id = pd.DataFrame.from_records(lvl_index2id, index="index")

        return lvl_index2id

    def ingest_levels(self, levels=None):

        if levels is None:
            levels = self.gfall_reader.levels

        # Select ions
        if self.ions is not None:
            levels = levels.reset_index().\
                                  join(self.ions, how="inner",
                                       on=["atomic_number", "ion_charge"]).\
                                  set_index(["atomic_number", "ion_charge", "level_index"])

        print("Ingesting levels from {}".format(self.data_source.short_name))

        for ion_index, ion_levels in levels.groupby(level=["atomic_number", "ion_charge"]):

            atomic_number, ion_charge = ion_index
            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting levels for {} {}".format(convert_atomic_number2symbol(atomic_number), ion_charge))

            for index, row in ion_levels.iterrows():

                level_index = index[2]  # index: (atomic_number, ion_charge, level_index)

                ion.levels.append(
                    Level(level_index=level_index,
                          data_source=self.data_source,
                          J=row["j"],
                          energies=[
                              LevelEnergy(quantity=row["energy"]*u.Unit("cm-1"),
                                          method=row["method"],
                                          data_source=self.data_source)
                          ])
                )

    def ingest_lines(self, lines=None):

        if lines is None:
            lines = self.gfall_reader.lines

        # Select ions
        if self.ions is not None:
            lines = lines.reset_index(). \
                join(self.ions, how="inner",
                     on=["atomic_number", "ion_charge"]). \
                set_index(["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"])

        print("Ingesting lines from {}".format(self.data_source.short_name))

        for ion_index, ion_lines in lines.groupby(level=["atomic_number", "ion_charge"]):

            atomic_number, ion_charge = ion_index
            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting lines for {} {}".format(convert_atomic_number2symbol(atomic_number), ion_charge))

            lvl_index2id = self.get_lvl_index2id(ion)

            for index, row in ion_lines.iterrows():

                # index: (atomic_number, ion_charge, lower_level_index, upper_level_index)
                lower_level_index, upper_level_index = index[2:]

                try:
                    lower_level_id = int(lvl_index2id.loc[lower_level_index])
                    upper_level_id = int(lvl_index2id.loc[upper_level_index])
                except KeyError:
                    raise IngesterError("Levels from this source have not been found."
                                        "You must ingest levels before transitions")

                medium = MEDIUM_VACUUM if row["wavelength"] <= GFALL_AIR_THRESHOLD else MEDIUM_AIR

                # Create a new line
                line = Line(
                    lower_level_id=lower_level_id,
                    upper_level_id=upper_level_id,
                    data_source=self.data_source,
                    wavelengths=[
                        LineWavelength(quantity=row["wavelength"] * u.nm,
                                       medium=medium,
                                       data_source=self.data_source)
                    ],
                    gf_values=[
                        LineGFValue(quantity=row["gf"],
                                    data_source=self.data_source)
                    ]
                )

                self.session.add(line)

    def ingest(self, levels=True, lines=True):

        if levels:
            self.ingest_levels()
            self.session.flush()

        if lines:
            self.ingest_lines()
            self.session.flush()


