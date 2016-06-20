import pytest

from carsus.io.output.tardis_op import create_basic_atom_df, BasicAtomData
from carsus.model import AtomWeight, Atom, Line, LineGFValue, LineAValue, Level


def test_create_basic_atomic_df(test_session):
    basic_atom_data = BasicAtomData(test_session, max_atomic_number=15)
    basic_atom_df = basic_atom_data.basic_atom_df
    basic_atom_df.reset_index(inplace=True)
    assert basic_atom_df["atomic_number"].max() == 15