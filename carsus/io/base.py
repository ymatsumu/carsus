import pandas as pd
from util import to_flat_dict
from carsus.alchemy import DataSource
from abc import ABCMeta, abstractmethod, abstractproperty
from sqlalchemy.orm.exc import NoResultFound

class ParserError(ValueError):
    pass


class BasePyparser(object):
    """
    Base class for parsers that use pyparsing grammar.

    Attributes:
    grammar: ~pyparsing.ParseElement -- the grammar used to parse input

    columns: ~list -- the column names corresponding to the grammar named tokens

    base_df: ~pandas.DataFrame -- contains parsed results from provided input data

    """
    __metaclass__ = ABCMeta

    def __init__(self, grammar, columns, input_data=None):
        self.grammar = grammar
        self.columns = columns
        self.base_df = pd.DataFrame()
        if input_data is not None:
            self.load(input_data)

    def load(self, input_data):
        """ Parses the input data and stores the results in the base_df """
        results = self.grammar.scanString(input_data)
        base_df_data = list()  # list of dicts that will be passed to the base_df
        for tokens, start, end in results:
            tokens_dict = to_flat_dict(tokens)  # make a flattaned dict with the column names as keys
            base_df_data.append(tokens_dict)
        self.base_df = pd.DataFrame(data=base_df_data, columns=self.columns)

    def __call__(self, input_data):
        self.load(input_data)


class IngesterError(ValueError):
    pass


class BaseIngester(object):

    __metaclass__ = ABCMeta

    def requirements_satisfied(self):
        return True

    def __init__(self, session, parser, downloader):
        self.session = session
        self.parser = parser
        self.downloader = downloader
        self.data_source = DataSource.as_unique(self.session, short_name=self.ds_short_name)

        if not self.requirements_satisfied():
            raise IngesterError('Requirements for ingest are not satisfied!')

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def ingest(self):
        pass

    def __call__(self):
        self.download()
        self.ingest()

    @abstractproperty
    def ds_short_name(self):  # Data source short name
        pass