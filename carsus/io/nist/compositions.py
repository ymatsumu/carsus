from carsus.io.base import BasePyparser, BaseIngester
from carsus.io.util import to_nom_val_and_std_dev
from carsus.io.nist.grammars.compositions_grammar import *
from carsus.alchemy import AtomicWeight, Atom, UnitDB
from astropy import units as u
import requests
import pandas as pd
from bs4 import BeautifulSoup

COMPOSITIONS_URL = "http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
DEFAULT_PARAMS = {'ascii': 'ascii2', 'isotype': 'some'}
NIST = "nist"

def download_compositions(url=COMPOSITIONS_URL, params=DEFAULT_PARAMS):
    """
    Makes a GET request to download data; then extracts preformatted text
    Parameters
    ----------
    url: ~str -- request url, (default="http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl")
    params: ~dict -- get request parameters (default={'ascii':'ascii2', 'isotype': 'some'})

    Returns: ~str preformatted text data
    -------

    """
    r = requests.get(url, params=params)
    soup = BeautifulSoup(r.text, 'html5lib')
    pre_text_data = soup.pre.get_text()
    pre_text_data = pre_text_data.replace(u'\xa0', u' ')  # replace non-breaking spaces with spaces
    return pre_text_data


class NISTCompositionsPyparser(BasePyparser):
    """ Class for parsers for the NIST Atomic Weights and Isotopic Compositions database """

    def __init__(self, input_data=None):
        super(NISTCompositionsPyparser, self).\
            __init__(grammar=isotope,
                     columns=COLUMNS,
                     input_data=input_data)

    def load(self, input_data):
        super(NISTCompositionsPyparser, self).load(input_data)
        self.base_df.set_index([ATOM_NUM_COL, MASS_NUM_COL], inplace=True)  # set multiindex "atomic_number","mass_number"

    def _prepare_atomic_weights(self, atomic_df):
        grouped_df = atomic_df.groupby([AW_TYPE_COL])
        interval_gr = grouped_df.get_group(INTERVAL).copy()
        stable_mass_num_gr = grouped_df.get_group(STABLE_MASS_NUM).copy()

        def atomic_weight_interval_to_nom_val_and_std(row):
            nom_val, std_dev = to_nom_val_and_std_dev([row[AW_LWR_BND_COL], row[AW_UPR_BND_COL]])
            return pd.Series([nom_val, std_dev])

        interval_gr[[AW_VAL_COL, AW_SD_COL]] = interval_gr.\
            apply(atomic_weight_interval_to_nom_val_and_std, axis=1)

        def atomic_weight_find_stable_atom_mass(row):
            stable_isotope = self.base_df.loc[row.name, row[AW_STABLE_MASS_NUM_COL]]
            return stable_isotope[[AM_VAL_COL, AM_SD_COL]]

        stable_mass_num_gr[[AW_VAL_COL, AW_SD_COL]] = stable_mass_num_gr.\
            apply(atomic_weight_find_stable_atom_mass, axis=1)

        atomic_df.update(interval_gr)
        atomic_df.update(stable_mass_num_gr)
        return atomic_df.drop([AW_TYPE_COL, AW_LWR_BND_COL, AW_UPR_BND_COL, AW_STABLE_MASS_NUM_COL], axis=1)

    def prepare_atomic_dataframe(self):
        """ Returns a new dataframe created from the base_df and containing data *only* related to atoms """
        atomic_df = self.base_df[ATOM_WEIGHT_COLS].reset_index(level=MASS_NUM_COL, drop=True)
        atomic_df = atomic_df[~atomic_df.index.duplicated()]
        atomic_df = self._prepare_atomic_weights(atomic_df)
        return atomic_df

    def prepare_isotope_dataframe(self):
        """ Returns a new dataframe created from the base_df and containing data *only* related to isotopes """
        pass


class NISTCompositionsIngester(BaseIngester):
    """ Class for ingesters for the NIST Atomic Weights and Isotopic Compositions database """

    ds_short_name = NIST  # Data source short name

    def __init__(self, atomic_db):
        super(NISTCompositionsIngester, self).\
            __init__(atomic_db,
                     parser=NISTCompositionsPyparser(),
                     downloader=download_compositions)

    def download(self):
        data = self.downloader()
        self.parser(data)

    def ingest(self):
        """ *Only* ingests atomic weights *for now* """
        atomic_df = self.parser.prepare_atomic_dataframe()
        atomic_df = atomic_df[pd.notnull(atomic_df[AW_VAL_COL])]

        u_u = UnitDB.as_unique(self.session, unit=u.u)

        for atom_num, row in atomic_df.iterrows():
            atom = self.session.query(Atom).filter(Atom.atomic_number==atom_num).one()
            atom.merge_quantity(self.session,
                AtomicWeight(data_source=self.data_source, value=row[AW_VAL_COL], std_dev=row[AW_SD_COL], unit_db=u_u))
