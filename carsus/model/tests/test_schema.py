
from sqlalchemy import and_
from carsus.model import Ion
from carsus.model.meta import IonListMixin, Base


def test_ion_list_mixin_insert_values(memory_session):

    class ChiantiIonList(Base, IonListMixin):
        pass

    ChiantiIonList.__table__.create(memory_session.connection())


    h0 = Ion(atomic_number=1, ion_charge=0)
    he0 = Ion(atomic_number=2, ion_charge=0)
    he1 = Ion(atomic_number=2, ion_charge=1)

    memory_session.add_all([h0, he0, he1])

    chianti_ions = [(1,0), (2,0)]

    for atomic_number, ion_charge in chianti_ions:
        chianti_ion = ChiantiIonList(atomic_number=atomic_number, ion_charge=ion_charge)
        memory_session.add(chianti_ion)

    chianti_ions_query = memory_session.query(Ion).\
        join(ChiantiIonList, and_(Ion.atomic_number == ChiantiIonList.atomic_number,
                                  Ion.ion_charge == ChiantiIonList.ion_charge)). \
        order_by(Ion.atomic_number, Ion.ion_charge)

    assert [(ion.atomic_number, ion.ion_charge) for ion in chianti_ions_query] == chianti_ions
