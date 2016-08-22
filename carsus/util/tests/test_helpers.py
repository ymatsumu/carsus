import pytest

from carsus.util.helpers import convert_camel2snake, \
    convert_atomic_number2symbol, convert_symbol2atomic_number


@pytest.mark.parametrize("input_camel_case, expected_snake_case", [
    ("Atom", "atom"),
    ("CHIANTIIon", "chianti_ion"),
    ("LevelJJTerm", "level_jj_term")
])
def test_convert_camel2snake(input_camel_case, expected_snake_case):
    assert convert_camel2snake(input_camel_case) == expected_snake_case


@pytest.mark.parametrize("atomic_number, expected_symbol", [
    (1, "H"),
    (14, "Si"),
    (30, "Zn"),
    (118, "Uuo")
])
def test_convert_atomic_number2symbol(atomic_number, expected_symbol):
    assert convert_atomic_number2symbol(atomic_number) == expected_symbol


@pytest.mark.parametrize("symbol, expected_atomic_number", [
    ("H", 1),
    ("Si", 14),
    ("Zn", 30),
    ("Uuo", 118)
])
def test_convert_symbol2atomic_number(symbol, expected_atomic_number):
    assert convert_symbol2atomic_number(symbol) == expected_atomic_number