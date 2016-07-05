import numpy as np
import pandas as pd
import hashlib
import uuid

from pandas import HDFStore
from carsus.model import Atom, Ion, Line, Level, DataSource, ECollision
from carsus.model.meta import yield_limit
from sqlalchemy import and_, union_all, literal
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.exc import NoResultFound
from astropy import constants as const
from astropy import units as u
from scipy import interpolate
from tardis.util import species_string_to_tuple

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

LINES_MAXRQ = 10000  # for yield_limit


class AtomData(object):
    """
    Class for creating the atomic dataframes for TARDIS

    Parameters:
    ------------
    session: SQLAlchemy session
    atom_masses_max_atomic_number: int
        The maximum atomic number to be stored in atom_masses
            (default: 30)
    levels_create_metastable_flags: bool
        Create the `metastable` column containing flags for metastable levels (levels that take a long time to de-excite)
        (default: True)
    lines_loggf_threshold: int
        log(gf) threshold for lines
    levels_metastable_loggf_threshold: int
        log(gf) threshold for flagging metastable levels
        (default: -3)
    chianti_species: list of str in format <element_symbol> <ion_number>, eg. Fe 2
            The levels data for these ions will be taken from the CHIANTI database
            (default: None)
    chianti_short_name: str
        The short name of the CHIANTI database, if set to None the latest version will be used
        (default: None)
    kurucz_short_name: str
        The short name of the Kurucz database, if set to None the latest version will be used
        (default: None)
    temperatures: np.array
        The temperatures for calculating collision strengths

    Attributes:
    ------------
    session: SQLAlchemy session
    atom_masses_param: dict
        The parameters for creating the `atom_masses` DataFrame
    levels_param: dict
        The parameters for creating the `levels` DataFrame
    lines_param: dict
        The parameters for creating the `lines` DataFrame
    collisions_param: dict
        The parameters for creating the `collisions` DataFrame

    ku_ds: carsus.model.atomic.DataSource instance
        The Kurucz datasource
    cu_ds: carsus.model.atomic.DataSource instance
        The CHIANTI datasource

    atom_masses
    ionization_energies
    levels_all
    lines_all
    levels
    lines
    collisions
    macro_atom
    macro_atom_references

    atom_masses_prepared
    ionization_energies_prepared
    levels_prepared
    lines_prepared
    collisions_prepared
    macro_atom_prepared
    macro_atom_references_prepared

    Methods:
    ---------
    create_atom_masses
    create_ionization_energies
    create_levels
    create_lines
    create_collisions
    create_macro_atom
    create_macro_atom_references

    prepare_atom_masses
    prepare_ionization_energies
    prepare_levels
    prepare_lines
    prepare_collisions
    prepare_macro_atom
    prepare_macro_atom_references

    """

    def __init__(self, session,
                 atom_masses_max_atomic_number=30, levels_create_metastable_flags=True,
                 lines_loggf_threshold=-3, levels_metastable_loggf_threshold=-3, chianti_species=None,
                 chianti_short_name=None, kurucz_short_name=None, collisions_temperatures=None):

        self.session = session

        # Set the parameters for the dataframes
        self.atom_masses_param = {
            "max_atomic_number": atom_masses_max_atomic_number
        }

        self.levels_param = {
            "create_metastable_flags": levels_create_metastable_flags,
            "metastable_loggf_threshold": levels_metastable_loggf_threshold
        }

        self.lines_param = {
            "loggf_threshold": lines_loggf_threshold
        }

        self.collisions_param = {
            "temperatures": collisions_temperatures
        }

        self.ku_ds = None
        self.ch_ds = None
        self.chianti_species = None

        # Query the data sources
        if kurucz_short_name is None:
            kurucz_short_name = "ku_latest"
        try:
            self.ku_ds = session.query(DataSource).filter(DataSource.short_name == kurucz_short_name).one()
        except NoResultFound:
            print "Kurucz data source does not exist!"
            raise

        if chianti_species is not None:

            # Get a list of tuples (atomic_number, ion_charge) for the chianti species
            chianti_species = [tuple(species_string_to_tuple(species_str)) for species_str in chianti_species]
            self.chianti_species = chianti_species

            if chianti_short_name is None:
                chianti_short_name = "chianti_v8.0.2"
            try:
                self.ch_ds = session.query(DataSource).filter(DataSource.short_name == chianti_short_name).one()
            except NoResultFound:
                print "Chianti data source does not exist!"
                raise

        self._atom_masses = None
        self._ionization_energies = None
        self._levels_all = None
        self._levels = None
        self._lines_all = None
        self._lines = None
        self._collisions = None
        self._macro_atom = None
        self._macro_atom_references = None

    @property
    def atom_masses_df(self):
        if self._atom_masses is None:
            self._atom_masses = self.create_atom_masses(**self.atom_masses_param)
        return self._atom_masses

    def create_atom_masses(self, max_atomic_number=30):
        """
        Create a DataFrame containing *atomic masses*.

        Parameters
        ----------
        max_atomic_number: int
            The maximum atomic number to be stored in `atom_masses`
            (default: 30)

        Returns
        -------
        atom_masses : pandas.DataFrame
            DataFrame with:
                index: none;
                columns: atom_masses, symbol, name, mass[u].
        """
        atom_masses_q = self.session.query(Atom). \
            filter(Atom.atomic_number <= max_atomic_number).\
            order_by(Atom.atomic_number)

        atom_masses = list()
        for atom in atom_masses_q.options(joinedload(Atom.weights)):
            weight = atom.weights[0].quantity.value if atom.weights else None  # Get the first weight from the collection
            atom_masses.append((atom.atomic_number, atom.symbol, atom.name, weight))

        atom_masses_dtype = [("atomic_number", np.int), ("symbol", "|S5"), ("name", "|S150"), ("mass", np.float)]
        atom_masses = np.array(atom_masses, dtype=atom_masses_dtype)
        atom_masses = pd.DataFrame.from_records(atom_masses)

        return atom_masses

    @property
    def atom_masses_prepared(self):
        return self.prepare_atom_masses()

    def prepare_atom_masses(self):
        """
        Prepare the DataFrame with atomic masses for TARDIS.

        Returns
        -------
        atom_masses_prepared : pandas.DataFrame
            DataFrame with:
                index: atomic_number;
                columns: symbol, name, mass[u].
        """
        atom_masses_prepared = self.atom_masses.set_index("atomic_number")
        return atom_masses_prepared

    @property
    def ionization_energies(self):
        if not self._ionization_energies:
            self._ionization_energies = self.create_ionization_energies()
        return self._ionization_energies

    def create_ionization_energies(self):
        """
        Create a DataFrame containing *ionization energies*.

        Returns
        -------
        ionization_energies : pandas.DataFrame
            DataFrame with:
                index: none;
                columns: atomic_number, ion_number, ionization_energy[eV]
        """
        ionization_energies_q = self.session.query(Ion).\
            order_by(Ion.atomic_number, Ion.ion_charge)

        ionization_energies = list()
        for ion in ionization_energies_q.options(joinedload(Ion.ionization_energies)):
            ionization_energy = ion.ionization_energies[0].quantity.value if ion.ionization_energies else None
            ionization_energies.append((ion.atomic_number, ion.ion_number, ionization_energy))

        ionization_dtype = [("atomic_number", np.int), ("ion_number", np.int), ("ionization_energy", np.float)]
        ionization_energies = np.array(ionization_energies, dtype=ionization_dtype)

        ionization_energies = pd.DataFrame.from_records(ionization_energies)

        return ionization_energies

    @property
    def ionization_energies_prepared(self):
        return self.prepare_ionization_energies()

    def prepare_ionization_energies(self):
        """
        Prepare the DataFrame with ionization energies for TARDIS

        Returns
        -------
        ionization_energies : pandas.DataFrame
            DataFrame with:
                index: atomic_number, ion_number;
                columns: ionization_energy[eV].

        Notes
        ------
        In TARDIS `ion_number` describes the final ion state,
        e.g. H I - H II is described with ion_number = 1
        On the other hand, in carsus `ion_number` describes the lower ion state,
        e.g. H I - H II is described with ion_number = 0
        For this reason we add 1 to `ion_number` in this prepare method.
        """
        ionization_energies = self.ionization_energies.copy()
        ionization_energies["ion_number"] += 1
        ionization_energies.set_index(["atomic_number", "ion_number"], inplace=True)

        return ionization_energies

    @property
    def levels_df(self):
        if self._levels_df is None:
            self._levels_df = self.create_levels_df()
        return self._levels_df

    def create_levels_df(self, create_metastable_flags=True, metastable_loggf_threshold=-3):
        """
            Create a DataFrame with levels data.

            Parameters
            ----------
            create_metastable_flags: bool
                Create the `metastable` column containing flags for metastable levels (levels that take a long time to de-excite)
                (default: True)
            metastable_loggf_threshold: int
                log(gf) threshold for flagging metastable levels
                (default: -3)

            Returns
            -------
            levels_df : pandas.DataFrame
                DataFrame with index: level_id
                         and columns: atomic_number, ion_number, level_number, energy[eV], g[1]
        """

        if self.chianti_species is None:
            kurucz_levels_q = self.session.query(Level).\
                filter(Level.data_source == self.ku_ds)

            levels_q = kurucz_levels_q

        else:

            # To select ions we create a CTE (Common Table Expression), because sqlite doesn't support composite IN statements
            chianti_species_cte = union_all(
                *[self.session.query(
                    literal(atomic_number).label("atomic_number"),
                    literal(ion_charge).label("ion_charge"))
                  for atomic_number, ion_charge in self.chianti_species]
            ).cte("chianti_species_cte")

            # To select chianti ions we join on the CTE
            chianti_levels_q = self.session.query(Level).\
                join(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                               Level.ion_charge == chianti_species_cte.c.ion_charge)).\
                filter(Level.data_source == self.ch_ds)

            # To select kurucz ions we do an outerjoin on the CTE and select rows that don't have a match from the CTE
            kurucz_levels_q = self.session.query(Level).\
                outerjoin(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                               Level.ion_charge == chianti_species_cte.c.ion_charge)).\
                filter(chianti_species_cte.c.atomic_number.is_(None)).\
                filter(Level.data_source == self.ku_ds)

            levels_q = kurucz_levels_q.union(chianti_levels_q)

        # Get the levels data
        levels_data = list()
        for lvl in levels_q.options(joinedload(Level.energies)):
            try:
                energy = None
                # Try to find the measured energy for this level
                for nrg in lvl.energies:
                    if nrg.method == "meas":
                        energy = nrg.quantity
                        break
                # If the measured energy is not available, try to get the first one
                if energy is None:
                    energy = lvl.energies[0].quantity
            except IndexError:
                print "No energy is available for level {0}".format(lvl.level_id)
                continue
            levels_data.append((lvl.level_id, lvl.atomic_number, lvl.ion_charge, energy.value, lvl.g, lvl.data_source_id))

        # Create a dataframe with the levels data
        levels_dtype = [("level_id", np.int), ("atomic_number", np.int),
                        ("ion_charge", np.int), ("energy", np.float), ("g", np.int), ("ds_id", np.int)]
        levels_data = np.array(levels_data, dtype=levels_dtype)
        levels_df = pd.DataFrame.from_records(levels_data, index="level_id")

        # Replace ion_charge with ion_number in the spectroscopic notation
        levels_df["ion_number"] = levels_df["ion_charge"] + 1
        levels_df.drop("ion_charge", axis=1, inplace=True)

        # Create level numbers
        levels_df.sort_values(["atomic_number", "ion_number", "energy", "g"], inplace=True)
        levels_df["level_number"] = levels_df.groupby(['atomic_number', 'ion_number'])['energy']. \
            transform(lambda x: np.arange(len(x))).values
        levels_df["level_number"] = levels_df["level_number"].astype(np.int)

        if create_metastable_flags:
            # Create metastable flags
            # ToDO: It is assumed that all lines are ingested. That may not always be the case

            levels_subq = self.session.query(Level). \
                filter(Level.level_id.in_(levels_df.index.values)).subquery()
            metastable_q = self.session.query(Line). \
                join(levels_subq, Line.upper_level)

            metastable_data = list()
            for line in yield_limit(metastable_q.options(joinedload(Line.gf_values)),
                                    Line.line_id, maxrq=LINES_MAXRQ):
                try:
                    # Currently it is assumed that each line has only one gf value
                    gf = line.gf_values[0].quantity  # Get the first gf value
                except IndexError:
                    print "No gf value is available for line {0}".format(line.line_id)
                    continue
                metastable_data.append((line.line_id, line.upper_level_id, gf.value))

            metastable_dtype = [("line_id", np.int), ("upper_level_id", np.int), ("gf", np.float)]
            metastable_data = np.array(metastable_data, dtype=metastable_dtype)
            metastable_df = pd.DataFrame.from_records(metastable_data, index="line_id")

            # Filter loggf on the threshold value
            metastable_df["loggf"] = np.log10(metastable_df["gf"])
            metastable_df = metastable_df.loc[metastable_df["loggf"] > metastable_loggf_threshold]

            # Count the remaining strong transitions
            metastable_df_grouped = metastable_df.groupby("upper_level_id")
            metastable_flags = metastable_df_grouped["upper_level_id"].count()
            metastable_flags.name = "metastable"

            # If there are no strong transitions for a level (the count is NaN) then the metastable flag is True
            # else (the count is a natural number) the metastable flag is False
            levels_df = levels_df.join(metastable_flags)
            levels_df["metastable"] = levels_df["metastable"].isnull()

        return levels_df

    @property
    def levels_df_prepared(self):
        return self.prepare_levels_df()

    def prepare_levels_df(self):
        """
        Prepare levels_df for TARDIS

        Returns
        -------
        levels_df : pandas.DataFrame
            DataFrame with columns: atomic_number, ion_number, level_number, energy[eV], g[1], metastable
        """

        levels_df = self.levels_df.copy()

        # Create multiindex
        levels_df.reset_index(inplace=True)
        levels_df.set_index(["atomic_number", "ion_number", "level_number"], inplace=True)

        # Drop the unwanted columns
        levels_df.drop(["level_id", "ds_id"], axis=1, inplace=True)

        return levels_df

    @property
    def lines_df(self):
        if self._lines_df is None:
            self._lines_df = self.create_lines_df()
        return self._lines_df

    def create_lines_df(self):
        """
            Create a DataFrame with lines data.

            Returns
            -------
            lines_df : pandas.DataFrame
                DataFrame with index: line_id
                and columns: atomic_number, ion_number, level_number_lower, level_number_upper,
                             wavelength[AA], nu[Hz], f_lu, f_ul, B_ul, B_ul, A_ul
        """
        levels_df = self.levels_df.copy()

        levels_subq = self.session.query(Level.level_id.label("level_id")). \
            filter(Level.level_id.in_(levels_df.index.values)).subquery()

        lines_q = self.session.query(Line).\
            join(levels_subq, Line.lower_level_id == levels_subq.c.level_id)

        lines_data = list()
        for line in yield_limit(lines_q.options(joinedload(Line.wavelengths), joinedload(Line.gf_values)),
                                Line.line_id, maxrq=LINES_MAXRQ):
            try:
                # Try to get the first gf value
                gf = line.gf_values[0].quantity
            except IndexError:
                print "No gf value is available for line {0}".format(line.line_id)
                continue
            try:
                # Try to get the first wavelength
                wavelength = line.wavelengths[0].quantity
            except IndexError:
                print "No wavelength is available for line {0}".format(line.line_id)
                continue
            lines_data.append((line.line_id, line.lower_level_id, line.upper_level_id,
                               line.data_source_id,  wavelength.value, gf.value))

        lines_dtype = [("line_id", np.int), ("lower_level_id", np.int), ("upper_level_id", np.int),
                       ("ds_id", np.int), ("wavelength", np.float), ("gf", np.float)]
        lines_data = np.array(lines_data, dtype=lines_dtype)
        lines_df = pd.DataFrame.from_records(lines_data, index="line_id")

        # Join atomic_number, ion_number, level_number_lower, level_number_upper and set multiindex
        ions_df = levels_df[["atomic_number", "ion_number"]]

        lower_levels_df = levels_df.rename(columns={"level_number": "level_number_lower", "g": "g_l"}).\
            loc[:,["level_number_lower", "g_l"]]
        upper_levels_df = levels_df.rename(columns={"level_number": "level_number_upper", "g": "g_u"}).\
            loc[:,["level_number_upper", "g_u"]]

        lines_df = lines_df.join(ions_df, on="lower_level_id")
        lines_df = lines_df.join(lower_levels_df, on="lower_level_id")
        lines_df = lines_df.join(upper_levels_df, on="upper_level_id")

        # Calculate absorption oscillator strength f_lu and emission oscillator strength f_ul
        lines_df["f_lu"] = lines_df["gf"]/lines_df["g_l"]
        lines_df["f_ul"] = -lines_df["gf"]/lines_df["g_u"]

        # Calculate frequency
        lines_df['nu'] = u.Unit('angstrom').to('Hz', lines_df['wavelength'], u.spectral())

        # Calculate Einstein coefficients
        einstein_coeff = (4 * np.pi**2 * const.e.gauss.value**2) / (const.m_e.cgs.value * const.c.cgs.value)
        lines_df['B_lu'] = einstein_coeff * lines_df['f_lu'] / (const.h.cgs.value * lines_df['nu'])
        lines_df['B_ul'] = einstein_coeff * lines_df['f_ul'] / (const.h.cgs.value * lines_df['nu'])
        lines_df['A_ul'] = -2 * einstein_coeff * lines_df['nu']**2 / const.c.cgs.value**2 * lines_df['f_ul']

        return lines_df

    @property
    def lines_df_prepared(self):
        return self.prepare_lines_df()

    def prepare_lines_df(self):
        """
            Prepare lines_df for TARDIS
            Parameters
            ----------
            session : SQLAlchemy session
            chianti_species: list of str in format <element_symbol> <ion_number>, eg. Fe 2
                The lines data for these ions will be taken from the CHIANTI database
                (default: None)
            chianti_short_name: str
                The short name of the CHIANTI database, if set to None the latest version will be used
                (default: None)
            kurucz_short_name: str
                The short name of the Kurucz database, if set to None the latest version will be used
                (default: None)
            Returns
            -------
            lines_df : pandas.DataFrame
                DataFrame with multiindex: atomic_number, ion_number, level_number_lower, level_number_upper
                and columns: line_id, wavelength[AA], nu[Hz], f_lu, f_ul, B_ul, B_ul, A_ul
        """

        #Set the multiindex
        lines_df = self.lines_df.reset_index()
        lines_df.set_index(["atomic_number", "ion_number", "level_number_lower", "level_number_upper"], inplace=True)

        # Drop the unwanted columns
        lines_df.drop(["g_l", "g_u", "gf", "lower_level_id", "upper_level_id", "ds_id"], axis=1, inplace=True)

        return lines_df

    @property
    def collisions_df(self):
        if self._collisions_df is None:
            self._collistions_df = self.create_collisions_df(**self.collisions_param)
        return self._collistions_df

    def create_collisions_df(self, temperatures=None):
        """
            Create a DataFrame with collisions data.

            Parameters
            -----------
            temperatures: np.array
                The temperatures for calculating collision strengths
                (default: None)

            Returns
            -------
            collisions_df : pandas.DataFrame
                DataFrame with indes: e_col_id,
                and columns:
        """

        if temperatures is None:
            temperatures = np.linspace(2000, 50000, 20)
        else:
            temperatures = np.array(temperatures)

        levels_df = self.levels_df.copy()

        levels_subq = self.session.query(Level.level_id.label("level_id")). \
            filter(Level.level_id.in_(levels_df.index.values)).\
            filter(Level.data_source == self.ch_ds).subquery()

        collisions_q = self.session.query(ECollision). \
            join(levels_subq, ECollision.lower_level_id == levels_subq.c.level_id)

        collisions_data = list()
        for e_col in collisions_q.options(joinedload(ECollision.gf_values),
                                          joinedload(ECollision.temp_strengths)):

            # Try to get the first gf value
            try:
                gf = e_col.gf_values[0].quantity
            except IndexError:
                print "No gf is available for electron collision {0}".format(e_col.e_col_id)
                continue

            btemp, bscups = (list(ts) for ts in zip(*e_col.temp_strengths_tuple))

            collisions_data.append((e_col.e_col_id, e_col.lower_level_id, e_col.upper_level_id,
                e_col.data_source_id, btemp, bscups, e_col.bt92_ttype, e_col.bt92_cups, gf.value))

        collisions_dtype = [("e_col_id", np.int), ("lower_level_id", np.int), ("upper_level_id", np.int),
                            ("ds_id", np.int),  ("btemp", 'O'), ("bscups", 'O'), ("ttype", np.int),
                            ("cups", np.float), ("gf", np.float)]

        collisions_data = np.array(collisions_data, dtype=collisions_dtype)
        collisions_df = pd.DataFrame.from_records(collisions_data, index="e_col_id")

        # Join atomic_number, ion_number, level_number_lower, level_number_upper and set multiindex
        ions_df = levels_df[["atomic_number", "ion_number"]]

        lower_levels_df = levels_df.rename(columns={"level_number": "level_number_lower", "g": "g_l", "energy": "energy_lower"}). \
                              loc[:, ["level_number_lower", "g_l", "energy_lower"]]
        upper_levels_df = levels_df.rename(columns={"level_number": "level_number_upper", "g": "g_u", "energy": "energy_upper"}). \
                              loc[:, ["level_number_upper", "g_u", "energy_upper"]]

        collisions_df = collisions_df.join(ions_df, on="lower_level_id")
        collisions_df = collisions_df.join(lower_levels_df, on="lower_level_id")
        collisions_df = collisions_df.join(upper_levels_df, on="upper_level_id")

        # Calculate delta_e
        kb_ev = const.k_B.cgs.to('eV / K').value
        collisions_df["delta_e"] = (collisions_df["energy_upper"] - collisions_df["energy_lower"])/kb_ev

        def calculate_collisional_strength(row, temperatures):
            """
                Function to calculation upsilon from Burgess & Tully 1992 (TType 1 - 4; Eq. 23 - 38)
            """
            c = row["cups"]
            x_knots = np.linspace(0, 1, len(row["btemp"]))
            y_knots = row["bscups"]
            delta_e = row["delta_e"]
            g_u = row["g_u"]

            ttype = row["ttype"]
            if ttype > 5: ttype -= 5

            kt = kb_ev * temperatures

            spline_tck = interpolate.splrep(x_knots, y_knots)

            if ttype == 1:
                x = 1 - np.log(c) / np.log(kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func * np.log(kt / delta_e + np.exp(1))

            elif ttype == 2:
                x = (kt / delta_e) / (kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func

            elif ttype == 3:
                x = (kt / delta_e) / (kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func / (kt / delta_e + 1)

            elif ttype == 4:
                x = 1 - np.log(c) / np.log(kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func * np.log(kt / delta_e + c)

            elif ttype == 5:
                raise ValueError('Not sure what to do with ttype=5')

            #### 1992A&A...254..436B Equation 20 & 22 #####

            c_ul = 8.63e-6 * upsilon / (g_u * temperatures**.5)
            return tuple(c_ul)

        collisions_df["c_ul"] = collisions_df.apply(calculate_collisional_strength, axis=1, args=(temperatures,))

        # Calculate g_ratio
        collisions_df["g_ratio"] = collisions_df["g_l"] / collisions_df["g_u"]

        return collisions_df

    @property
    def collisions_df_prepared(self):
        return self.prepare_collisions_df()

    def prepare_collisions_df(self):
        """
            Prepare collisions_df for TARDIS

            Returns
            -------
            collisions_df : pandas.DataFrame
                DataFrame with multiindex: atomic_number, ion_number, level_number_lower, level_number_upper
                and columns: e_col_id, delta_e, g_ratio, c_ul
        """

        collisions_df = self.collisions_df.copy()

        # Drop the unwanted columns
        collisions_df.drop(["lower_level_id", "upper_level_id", "ds_id", "btemp", "bscups",
                            "ttype", "energy_lower", "energy_upper", "gf", "g_l", "g_u", "cups"],  axis=1, inplace=True)

        # Set multiindex
        collisions_df.reset_index(inplace=True)
        collisions_df.set_index(["atomic_number", "ion_number", "level_number_lower", "level_number_upper"], inplace=True)

        return collisions_df

    @property
    def macro_atom_df(self):
        if self._macro_atom_df is None:
            self._macro_atom_df = self.create_macro_atom_df()
        return self._macro_atom_df

    def create_macro_atom_df(self):
        """
            Create a DataFrame with macro atom data.

            Returns
            -------
            macro_atom_df : pandas.DataFrame
                DataFrame with columns: atomic_number, ion_number, source_level_number, target_level_number,
                transition_line_id, transition_type, transition_probability

            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html

        """

        levels_df = self.levels_df.copy()
        lines_df = self.lines_df.copy()

        lvl_energy_lower_df = levels_df.rename(columns={"energy": "energy_lower"}).loc[:, ["energy_lower"]]
        lvl_energy_upper_df = levels_df.rename(columns={"energy": "energy_upper"}).loc[:, ["energy_upper"]]

        lines_df = lines_df.join(lvl_energy_lower_df, on="lower_level_id")
        lines_df = lines_df.join(lvl_energy_upper_df, on="upper_level_id")

        macro_atom_data = list()
        macro_atom_dtype = [("atomic_number", np.int), ("ion_number", np.int),
                            ("source_level_number", np.int), ("target_level_number", np.int),
                            ("transition_line_id", np.int), ("transition_type", np.int), ("transition_probability", np.float)]

        for line_id, row in lines_df.iterrows():
            atomic_number, ion_number = row["atomic_number"], row["ion_number"]
            level_number_lower, level_number_upper = row["level_number_lower"], row["level_number_upper"]
            nu = row["nu"]
            f_ul, f_lu = row["f_ul"], row["f_lu"]
            e_lower, e_upper = row["energy_lower"], row["energy_upper"]

            transition_probabilities_dict = dict()  # type : probability
            transition_probabilities_dict[P_EMISSION_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * (e_upper - e_lower)
            transition_probabilities_dict[P_INTERNAL_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * e_lower
            transition_probabilities_dict[P_INTERNAL_UP] = f_lu * e_lower / (const.h.cgs.value * nu)

            macro_atom_data.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_EMISSION_DOWN, transition_probabilities_dict[P_EMISSION_DOWN]))
            macro_atom_data.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_INTERNAL_DOWN, transition_probabilities_dict[P_INTERNAL_DOWN]))
            macro_atom_data.append((atomic_number, ion_number, level_number_lower, level_number_upper,
                                    line_id, P_INTERNAL_UP, transition_probabilities_dict[P_INTERNAL_UP]))

        macro_atom_data = np.array(macro_atom_data, dtype=macro_atom_dtype)
        macro_atom_df = pd.DataFrame(macro_atom_data)

        return macro_atom_df

    @property
    def macro_atom_df_prepared(self):
        return self.prepare_macro_atom_df()

    def prepare_macro_atom_df(self):
        """
            Prepare macro_atom_df for TARDIS

            Returns
            -------
            macro_atom_df : pandas.DataFrame
                DataFrame with muliindex: atomic_number, ion_number, source_level_number, target_level_number
                and columns: transition_line_id, transition_type, transition_probability

            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html

        """
        macro_atom_df = self.macro_atom_df.set_index(["atomic_number", "ion_number", "source_level_number", "target_level_number"])
        macro_atom_df.sort_index(level=["atomic_number", "ion_number", "source_level_number"], inplace=True)
        return macro_atom_df

    @property
    def macro_atom_ref_df(self):
        if self._macro_atom_ref_df is None:
            self._macro_atom_ref_df = self.create_macro_atom_ref_df()
        return self._macro_atom_ref_df

    def create_macro_atom_ref_df(self):
        """
            Create a DataFrame with macro atom reference data.

            Returns
            -------
            macro_atom_ref_df : pandas.DataFrame
                DataFrame with index: level_id,
                and columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total
        """

        levels_df = self.levels_df.copy()
        lines_df = self.lines_df.copy()

        macro_atom_ref_df = levels_df.rename(columns={"level_number": "source_level_number"}).\
                                       loc[:, ["atomic_number", "ion_number", "source_level_number"]]

        count_down = lines_df.groupby("upper_level_id").size()
        count_down.name = "count_down"

        count_up = lines_df.groupby("lower_level_id").size()
        count_up.name = "count_up"

        macro_atom_ref_df = macro_atom_ref_df.join(count_down).join(count_up)
        macro_atom_ref_df.fillna(0, inplace=True)
        macro_atom_ref_df["count_total"] = 2*macro_atom_ref_df["count_down"] + macro_atom_ref_df["count_up"]

        return macro_atom_ref_df

    @property
    def macro_atom_ref_df_prepared(self):
        return self.prepare_macro_atom_ref_df()

    def prepare_macro_atom_ref_df(self):
        """
            Prepare macro_atom_ref_df for TARDIS

            Returns
            -------
            macro_atom_ref_df : pandas.DataFrame
                DataFrame with multiindex: atomic_number, ion_number, source_level_number
                and columns: level_id, count_down, count_up, count_total
        """
        macro_atom_ref_df = self.macro_atom_ref_df.copy()

        macro_atom_ref_df.reset_index(inplace=True)
        macro_atom_ref_df.set_index(["atomic_number", "ion_number", "source_level_number"], inplace=True)

        return macro_atom_ref_df

    def to_hdf(self, hdf5_path, store_basic_atom=True, store_ionization=True,
               store_levels=True, store_lines=True, store_collisions=True, store_macro_atom=True,
               store_macro_atom_ref=True):
        """
            Store the dataframes in an HDF5 file

            Parameters
            ------------
            hdf5_path: str
                The path of the HDF5 file
            store_basic_atom: bool
                Store the basic atom DataFrame
                (default: True)
            store_ionization: bool
                Store the ionzation DataFrame
                (default: True)
            store_levels: bool
                Store the levels DataFrame
                (default: True)
            store_lines: bool
                Store the lines DataFrame
                (default: True)
            store_collisions: bool
                Store the electron collisions DataFrame
                (default: True)
            store_macro_atom: bool
                Store the macro_atom DataFrame
                (default: True)
            store_macro_atom_ref: bool
                Store the macro_atom_references DataFrame
                (default: True)
        """

        with HDFStore(hdf5_path) as store:

            if store_basic_atom:
                store.put("basic_atom_df", self.basic_atom_df_prepared)

            if store_ionization:
                store.put("ionization_df", self.ionization_df_prepared)

            if store_levels:
                store.put("levels_df", self.levels_df_prepared)

            if store_lines:
                store.put("lines_df", self.lines_df_prepared)

            if store_collisions:
                store.put("collisions_df", self.collisions_df_prepared)

            if store_macro_atom:
                store.put("macro_atom_df", self.macro_atom_df_prepared)

            if store_macro_atom_ref:
                store.put("macro_atom_ref_df", self.macro_atom_ref_df_prepared)

            # Set the root attributes
            # It seems that the only way to set the root attributes is to use `_v_attrs`
            store.root._v_attrs["database_version"] = "v0.9"

            print "Signing AtomData with MD5 and UUID1"

            md5_hash = hashlib.md5()
            for key in store.keys():
                md5_hash.update(store[key].values.data)

            uuid1 = uuid.uuid1().hex

            store.root._v_attrs['md5'] = md5_hash.hexdigest()
            store.root._v_attrs['uuid1'] = uuid1