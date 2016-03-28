"""This module defines base classes for parsers and ingesters."""

import pandas as pd
from util import to_flat_dict
from abc import ABCMeta, abstractmethod, abstractproperty


class ParserError(ValueError):
    pass


class BaseParser(object):
    """
    Abstract base class for parsers.

    Attributes
    ----------
    base_df : pandas.DataFrame
        Contains parsed results from the provided input data.

    Methods
    -------
    load(input_data)
        Parses the input data and stores the results in the `base_df` attribute

    Notes
    -----
    Instead of invoking the `load` method you can just "call" an instance with input data.

    """
    __metaclass__ = ABCMeta

    def __init__(self, input_data=None):
        self.base_df = pd.DataFrame()
        if input_data is not None:
            self.load(input_data)

    @abstractmethod
    def load(self, input_data):
        pass

    def __call__(self, input_data):
        self.load(input_data)


class BasePyparser(BaseParser):
    """
    Abstract base class for parsers that use pyparsing grammar.

    Attributes
    ----------
    base_df : pandas.DataFrame
        Contains parsed results from the provided input data.

    grammar : pyparsing.ParseElement
        The grammar used to parse input.
        Its labeled tokens correspond to the columns of the `base_df`

    columns : list of str
        The column names of the `base_df`

    Methods
    -------
    load(input_data)
        Parses the input data and stores the results in the `base_df` attribute

    Notes
    -----
    Rationale: pyparsers have a specific load workflow illustrated below.

    Suppose a `base_df` of some parser has three columns::

        atomic_mass_nominal_value | atomic_mass_std_dev | notes

    The `load` method scans the input data with parser's `grammar`.
    The returned matches have nested labeled tokens that correspond to the columns.
    Say, one of the matches has the following nested tokens list::

        - atomic_mass: ['37.96273211', '(', '21', ')']
            - nominal_value: 37.96273211
            - std_dev: 2.1e-07

    The `load` method then infers the columns' values from
    the nested labels and adds the following row to the `base_df`::

        atomic_mass_nominal_value            37.9627
        atomic_mass_std_dev                  2.1e-07
        notes                                NaN

    """
    __metaclass__ = ABCMeta

    def __init__(self, grammar, columns, input_data=None):
        self.grammar = grammar
        self.columns = columns
        super(BasePyparser, self).__init__(input_data)

    def load(self, input_data):
        results = self.grammar.scanString(input_data)
        base_df_data = list()  # list of dicts that will be passed to the base_df
        for tokens, start, end in results:
            tokens_dict = to_flat_dict(tokens)  # make a flattened dict with the column names as keys
            base_df_data.append(tokens_dict)
        self.base_df = pd.DataFrame(data=base_df_data, columns=self.columns)



class IngesterError(ValueError):
    pass


class BaseIngester(object):
    """
    Abstract base class for ingesters.

    Attributes
    ----------
    parser : BaseParser instance
        Parses the downloaded data

    downloader : function
        Downloads the data

    ds_short_name : str
        The short name of the data source

    Methods
    -------
    download()
        Downloads the data with the 'downloader' and loads the `parser` with it

    ingest(session)
        Persists the downloaded data into the database

    """
    __metaclass__ = ABCMeta

    def requirements_satisfied(self):
        return True

    def __init__(self, parser, downloader):
        self.parser = parser
        self.downloader = downloader

        if not self.requirements_satisfied():
            raise IngesterError('Requirements for ingest are not satisfied!')

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def ingest(self, session):
        pass

    #def __call__(self, session):
    #    self.download()
    #    self.ingest(session)

    @abstractproperty
    def ds_short_name(self):  # Data source short name
        pass