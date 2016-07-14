import pytest

from carsus.util import convert_camel2snake


@pytest.mark.parametrize("input_camel_case, expected_snake_case", [
    ("Atom", "atom"),
    ("CHIANTIIon", "chianti_ion"),
    ("LevelJJTerm", "level_jj_term")
])
def test_convert_camel2snake(input_camel_case, expected_snake_case):
    assert convert_camel2snake(input_camel_case) == expected_snake_case