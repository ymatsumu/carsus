from carsus.model import Atom, AtomicWeight, DataSource
from pandas import read_sql_query, HDFStore


class Dataset(object):

    def __init__(self, name, data=None):
        self.name = name
        self._data = data

    @property
    def data(self):
        if self._data is None:
            raise ValueError('Data is not available please load first')
        else:
            return self._data

    @data.setter
    def data(self, value):
        self._data = value


class AtomsDataset(Dataset):

    def __init__(self, name="atoms", data=None):
        super(AtomsDataset, self).__init__(name, data)

    def load_sql(self, session, load_atomic_weights=False):
        q = session.query(Atom.atomic_number.label("atomic_number"),
                              Atom.symbol.label("symbol"),
                              Atom.name.label("name"),
                              Atom.group.label("group"),
                              Atom.period.label("period"),
                              )
        if load_atomic_weights:

            q = q.add_columns(AtomicWeight.quantity.value.label("atomic_weight_value"),
                              AtomicWeight.std_dev.label("atomic_weight_uncert"),
                              DataSource.short_name.label("atomic_weight_data_source")).\
                              join(Atom.quantities.of_type(AtomicWeight)).\
                              join(AtomicWeight.data_source)

        df = read_sql_query(q.selectable, session.bind, index_col="atomic_number")
        self.data = df


def create_hdf(fname, datasets):
    """
    Store datasets in a HDF file

    Parameters
    ----------
    fname : str
        The name of the HDF file

    datasets : list of Dataset objects
        The datasets that will be put in the HDF file

    """
    with HDFStore(fname, "w") as store:
        for dataset in datasets:
            store[dataset.name] = dataset.data


