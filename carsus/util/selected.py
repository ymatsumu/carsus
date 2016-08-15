"""
BNF grammar for parsing spectra

element ::= 'H' .. 'Uuh'
element_range ::= element + '-' + element

selected_atoms = element | element_range + [',' + element | element_range]*
"""

from helpers import symbol2atomic_number
from pyparsing import Literal, Suppress, delimitedList, Word, alphas

hyphen = Suppress(Literal('-'))

element = Word(alphas)

def parse_element(tokens):
    symbol = tokens[0]
    symbol = symbol[:1].upper() + symbol[1:].lower()
    try:
        atomic_number = symbol2atomic_number[symbol]
    except KeyError:
        raise ValueError("Unrecognized atomic symbol {}".format(symbol))

    return atomic_number

element.setParseAction(parse_element)

element_range = element + hyphen + element
element_range.setParseAction(lambda x: range(x[0], x[1] + 1))

selected_atomic_numbers = delimitedList(element ^ element_range)
selected_atomic_numbers.setParseAction(lambda x: sorted(set(x)))


def parse_selected_atoms(selected_atoms):
    """
    Parse the sting specifying selected atoms to the list of atomic numbers.

    Parameters
    ----------
    selected_atoms: str
        Sting that specifies selected atoms. It should consist of comma-separated entries
        that are either single atoms (e.g. "H") or ranges (indicated by using a hyphen between, e.g "H-Zn").
        Element symbols need **not** to be capitalized.

    Returns
    -------
    list of int
        List of atomic numbers

    Examples
    --------

    >>> parse_selected_atoms("H")
    [1]

    >>> parse_selected_atoms("H, Li-N")
    [1, 3, 4, 5, 6, 7]

    >>> parse_selected_atoms("H, Li-N, Si, S")
    [1, 3, 4, 5, 6, 7, 14, 16]

    """
    return selected_atomic_numbers.parseString(selected_atoms).asList()

