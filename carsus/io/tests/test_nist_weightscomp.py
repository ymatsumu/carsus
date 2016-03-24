import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
from astropy import units as u
from carsus.io.nist import NISTWeightsCompIngester, NISTWeightsCompPyparser
from carsus.io.nist.weightscomp_grammar import *
from carsus.alchemy import Atom, AtomicWeight, UnitDB

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
def weightscomp_ingester(atomic_db):
    session = atomic_db.session_maker()
    ingester = NISTWeightsCompIngester(session)
    ingester.parser(test_input)
    return ingester


def test_weightscomp_pyparser_base_df_index(weightscomp_pyparser):
    assert weightscomp_pyparser.base_df.index.names == [ATOM_NUM_COL, MASS_NUM_COL]


def test_weightscomp_pyparser_prepare_atomic_df_index(atomic_df):
    assert atomic_df.index.name == ATOM_NUM_COL


def test_weightscomp_pyparser_prepare_atomic_df_(atomic_df, expected_df):
    assert_frame_equal(atomic_df, expected_df, check_names=False)



@pytest.mark.parametrize("atomic_number,nom_val,std_dev", expected_tuples)
def test_weithscomp_ingest_existing_atomic_weights(atomic_number, nom_val, std_dev, weightscomp_ingester):
    u_u = UnitDB.as_unique(weightscomp_ingester.session, unit=u.u)
    atom = weightscomp_ingester.session.query(Atom).filter(Atom.atomic_number==atomic_number).one()
    atom.quantities = [
        AtomicWeight(data_source=weightscomp_ingester.data_source, value=9.9999, unit_db=u_u),
    ]
    weightscomp_ingester.session.commit()

    weightscomp_ingester.ingest()
    q = weightscomp_ingester.session.query(Atom, AtomicWeight).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        filter(AtomicWeight.data_source==weightscomp_ingester.data_source).\
        filter(Atom.atomic_number==atomic_number).one()

    assert q.AtomicWeight.atomic_number == atomic_number
    assert_almost_equal(q.AtomicWeight.value, nom_val)
    assert_almost_equal(q.AtomicWeight.std_dev, std_dev)


def test_weightscomp_ingest_nonexisting_atomic_weights(weightscomp_ingester):
    weightscomp_ingester.ingest()
    for t in expected_tuples:
        aw = weightscomp_ingester.session.query(AtomicWeight).\
            filter(AtomicWeight.atomic_number==t[0]).\
            filter(AtomicWeight.data_source==weightscomp_ingester.data_source).one()
        assert_almost_equal(aw.value, t[1])
        assert_almost_equal(aw.std_dev, t[2])


@pytest.mark.remote_data
def test_weightscomp_ingest_default_count(weightscomp_ingester):
    weightscomp_ingester()
    assert weightscomp_ingester.session.query(AtomicWeight).\
               filter(AtomicWeight.data_source==weightscomp_ingester.data_source).count() == 94
