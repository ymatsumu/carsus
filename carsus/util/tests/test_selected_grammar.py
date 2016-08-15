import pytest

from carsus.util.selected_grammar import element, element_range, \
    selected_atomic_numbers


@pytest.mark.parametrize("test_input, exp_atomic_number",[
    ("H", 1),
    ("Zn", 30),
    ("Uuo", 118),
    ("h", 1)
])
def test_element(test_input, exp_atomic_number):
    tokens = element.parseString(test_input)
    assert tokens[0] == exp_atomic_number


@pytest.mark.parametrize("test_input, exp_atomic_numbers",[
    ("H-Li", [1, 2, 3]),
    ("H-Zn", range(1,31)),
    ("si-s", [14, 15, 16])
])
def test_element_range(test_input, exp_atomic_numbers):
    tokens = element_range.parseString(test_input)
    assert tokens[:] == exp_atomic_numbers


@pytest.mark.parametrize("test_input, exp_atomic_numbers",[
    ("H", [1,]),
    ("H-Zn", range(1,31)),
    ("h, si-s", [1, 14, 15, 16])
])
def test_selected_atomic_numbers(test_input, exp_atomic_numbers):
    tokens = selected_atomic_numbers.parseString(test_input)
    assert tokens[:] == exp_atomic_numbers