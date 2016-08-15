"""
BNF grammar for parsing spectra

element ::= 'H' .. 'Uuh'
element_range ::= element + '-' + element

selected_atoms = element | element_range + [',' + element | element_range]*
"""

from helpers import symbol2atomic_number
from pyparsing import oneOf, Literal, Suppress, delimitedList

hyphen = Suppress(Literal('-'))

element = oneOf(symbol2atomic_number.keys(), caseless=True)
element.setParseAction(lambda x: symbol2atomic_number[x[0]])

element_range = element + hyphen + element
element_range.setParseAction(lambda x: range(x[0], x[1] + 1))

selected_atomic_numbers = delimitedList(element ^ element_range)
