"""fundamental units like declarative_base"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import engine_from_config
from contextlib import contextmanager

Base = declarative_base()

Session = sessionmaker()

engine = None


def setup(config):
    """Setup the application given a config dictionary."""

    global engine
    engine = engine_from_config(config, prefix='')
    Base.metadata.create_all(engine)
    Session.configure(bind=engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
