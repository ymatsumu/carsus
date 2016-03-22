""" Object-relational mapping helpers """


class UniqueMixin(object):
    """Unique object mixin.

    Allows an object to be returned or created as needed based on
    criterion.

    .. seealso::

        http://www.sqlalchemy.org/trac/wiki/UsageRecipes/UniqueObject

    """
    @classmethod
    def unique_hash(cls, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def unique_filter(cls, query, *args, **kwargs):
        raise NotImplementedError()


    @classmethod
    def as_unique(cls, session, *args, **kwargs):

        hashfunc = cls.unique_hash
        queryfunc = cls.unique_filter
        constructor = cls

        if 'unique_cache' not in session.info:
            session.info['unique_cache'] = cache = {}
        else:
            cache = session.info['unique_cache']

        key = (cls, hashfunc(*args, **kwargs))
        if key in cache:
            return cache[key]
        else:
            with session.no_autoflush:
                q = session.query(cls)
                q = queryfunc(q, *args, **kwargs)
                obj = q.first()
                if not obj:
                    obj = constructor(*args, **kwargs)
                    session.add(obj)
            cache[key] = obj
            return obj
