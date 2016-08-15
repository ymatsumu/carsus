import pytest

from carsus.util.helpers import convert_camel2snake, \
    atomic_number2symbol, symbol2atomic_number,\
    parse_selected_atoms


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
def test_atomic_number2symbol(atomic_number, expected_symbol):
    assert atomic_number2symbol[atomic_number] == expected_symbol


@pytest.mark.parametrize("symbol, expected_atomic_number", [
    ("H", 1),
    ("Si", 14),
    ("Zn", 30),
    ("Uuo", 118)
])
def test_symbol2atomic_number(symbol, expected_atomic_number):
    assert symbol2atomic_number[symbol] == expected_atomic_number


@pytest.mark.parametrize("selected_atoms, expected_list", [
    ("H", [1]),
    ("H-Li", [1, 2, 3]),
    ("H, Be-B", [1, 4, 5]),
    ("h, be-b", [1, 4, 5]),
    (" h ,  be - b ", [1, 4, 5])
])
def test_parse_selected_atoms(selected_atoms, expected_list):
    assert parse_selected_atoms(selected_atoms) == expected_list