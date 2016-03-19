import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from carsus.alchemy import DataSource
from carsus.io.nist.grammars.compositions_grammar import AW_SD_COL, AW_VAL_COL


@pytest.mark.parametrize("test_input,expected",[
    ("""atomic_weight = 6.8083492038(23)
         atomic_weight = 8.8239833(11)
         atomic_weight = 2.19802(3)""",
     [{AW_VAL_COL: "6.8083492038", AW_SD_COL: '23'},
      {AW_VAL_COL: "8.8239833", AW_SD_COL: '11'},
      {AW_VAL_COL: "2.19802", AW_SD_COL: '3'}])
])
def test_pyparser_load(test_input, expected, aw_pyparser):
    aw_pyparser.load(test_input)
    assert_frame_equal(aw_pyparser.base_df,
        pd.DataFrame(data=expected, columns=[AW_VAL_COL, AW_SD_COL]), check_names=False)


def test_pyparser_callable(aw_pyparser):
    aw_pyparser(input_data="atomic_weight = 6.8083492038(23)")
    assert aw_pyparser.base_df.loc[0, AW_VAL_COL] == "6.8083492038"
    assert aw_pyparser.base_df.loc[0, AW_SD_COL] == "23"


def test_base_ingester_add_data_source(ingester):
    ingester.atomic_db.session.query(DataSource).\
        filter_by(short_name=ingester.ds_short_name).one()


def test_base_ingester_ingest(ingester):
    ingester()