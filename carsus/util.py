import os

def carsus_data_dir(fname):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', fname)