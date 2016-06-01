import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
from astropy import units as u
from sqlalchemy import and_
from carsus.io.nist import NISTWeightsCompIngester, NISTWeightsCompPyparser
from carsus.io.nist.weightscomp_grammar import *
from carsus.model import Atom, AtomWeight, DataSource

test_input = """
Atomic Number = 35
Atomic Symbol = Br
Mass Number = 79
Relative Atomic Mass = 78.9183376(14)
Isotopic Composition = 0.5069(7)
Standard Atomic Weight = [79.901,79.907]
Notes =

Atomic Number = 38
Atomic Symbol = Sr
Mass Number = 84
Relative Atomic Mass = 83.9134191(13)
Isotopic Composition = 0.0056(1)
Standard Atomic Weight = 87.62(1)
Notes = g,r

Atomic Number = 38
Atomic Symbol = Sr
Mass Number = 86
Relative Atomic Mass = 85.9092606(12)
Isotopic Composition = 0.0986(1)
Standard Atomic Weight = 87.62(1)
Notes = g,r

Atomic Number = 43
Atomic Symbol = Tc
Mass Number = 98
Relative Atomic Mass = 97.9072124(36)
Isotopic Composition =
Standard Atomic Weight = [98]
Notes =

Atomic Number = 43
Atomic Symbol = Tc
Mass Number = 99
Relative Atomic Mass = 98.9062508(10)
Isotopic Composition =
Standard Atomic Weight = [98]
Notes =
"""

expected_dict=[
    {'atomic_number': 35, 'atomic_weight_nominal_value': 79.904, 'atomic_weight_std_dev': 3e-3},
    {'atomic_number': 38, 'atomic_weight_nominal_value': 87.62, 'atomic_weight_std_dev': 1e-2},
    {'atomic_number': 43, 'atomic_weight_nominal_value': 97.9072124, 'atomic_weight_std_dev': 36e-7}
]

expected_tuples = [
    (43, 97.9072124, 36e-7),
    (35, 79.904, 3e-3),
    (38, 87.62, 1e-2)
]


@pytest.fixture
def weightscomp_pyparser():
    return NISTWeightsCompPyparser(input_data=test_input)

@pytest.fixture
def atomic_df(weightscomp_pyparser):
    return weightscomp_pyparser.prepare_atomic_dataframe()


@pytest.fixture
def expected_df():
    return pd.DataFrame(data=expected_dict, columns=[ATOM_NUM_COL, AW_VAL_COL, AW_SD_COL]).set_index(ATOM_NUM_COL)


@pytest.fixture
def weightscomp_ingester():
    ingester = NISTWeightsCompIngester()
    ingester.parser(test_input)
    return ingester


def test_weightscomp_pyparser_base_df_index(weightscomp_pyparser):
    assert weightscomp_pyparser.base_df.index.names == [ATOM_NUM_COL, MASS_NUM_COL]


def test_weightscomp_pyparser_prepare_atomic_df_index(atomic_df):
    assert atomic_df.index.name == ATOM_NUM_COL


def test_weightscomp_pyparser_prepare_atomic_df_(atomic_df, expected_df):
    assert_frame_equal(atomic_df, expected_df, check_names=False)


@pytest.mark.parametrize("atomic_number,value,uncert", expected_tuples)
def test_weightscomp_ingest_atomic_weights(atomic_number, value, uncert, weightscomp_ingester, test_session):
    weightscomp_ingester.ingest(test_session)
    nist = DataSource.as_unique(test_session, short_name="nist")
    atom, aw = test_session.query(Atom, AtomWeight).\
            filter(and_(Atom.atomic_number==atomic_number,
                        Atom.data_source==nist)).\
            join(Atom.weights).one()
    assert_almost_equal(aw.quantity.value, value)
    assert_almost_equal(aw.uncert, uncert)


@pytest.mark.remote_data
def test_weightscomp_ingest_default_count(weightscomp_ingester, test_session):
    weightscomp_ingester.download()
    weightscomp_ingester.ingest(test_session)
    assert test_session.query(AtomWeight).count() == 94
