"""The module holds implementation of classes that handle identification and selection of right allomteric parameters and equations."""

# Importing Python Libraries
import os
import json
import functools
from typing import Dict, List, Optional, Tuple

import numpy as np
from fuzzywuzzy import process
from scipy.optimize import curve_fit

import logging
from scipy.optimize import OptimizeWarning
import warnings
logger = logging.getLogger(__name__)

FITTED_CURVES = {}
root_to_shoot = 0.26

def mass_1_biomass_function(dbh, A, B, C):
    return (np.exp(A + B * np.log(dbh) + C / 2)) / (1 - Species.root_to_shoot_ratio)

def mass_2_biomass_function(dbh, A, B, C):
    return (A * pow(dbh, B + C)) / (1 - Species.root_to_shoot_ratio)

def polynomial_fit(dbh, B0, B1, B2, DBHMin, DBHMax, should_convert):
    # converting the dbh in cm into inches
    if should_convert:
        dbh = min(DBHMax, max(DBHMin, 0.393700787 * dbh))
    estimate = B0 + (B1 * dbh) + (B2 * dbh * dbh)
    # converting into meters
    if should_convert:
        return max(0, 0.3048 * estimate)
    else:
        return max(0, estimate)
    
def exponential_fit(dbh, B0, B1, DBHMin, DBHMax):
    # converting the dbh in cm into inche
    dbh = min(DBHMax, max(DBHMin, 0.393700787 * dbh))
    estimate = np.exp(B0 + B1 * np.log(dbh))
    # converting into meters
    return max(0, 0.3048 * estimate)

def parametric_fit(dbh, B0, B1, B2):
    dbh = max(0, dbh)
    estimate = B0 + B1 * (dbh**B2)
    return max(0, estimate)

def log_linear_fit(dbh, B0, B1, DBHMin, DBHMax):
    # converting the dbh in cm into inche
    dbh = min(DBHMax, max(DBHMin, 0.393700787 * dbh))
    estimate = B0 + B1 * np.log(dbh)
    # converting into meters
    return max(0, 0.3048 * estimate)

def linear_fit(dbh, B0, B1, DBHMin, DBHMax):
    # converting the dbh in cm into inche
    dbh = min(DBHMax, max(DBHMin, 0.393700787 * dbh))
    estimate = B0 + (B1 * dbh)
    # converting into meters
    return max(0, 0.3048 * estimate)

class Species:
    """Object that holds standard tree growth rate by species at different sites.
    Source: https://database.itreetools.org/#/speciesSearch

    Todo:
        Consider to convert this into a db or API.
    """

    # Ref: Root to shoot ratio (Cairns et al. 1997)
    root_to_shoot_ratio = 0.26
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Species, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, species_db: Optional[str] = None):
        if self._initialized and species_db is None: return
        print("input file: {}".format(species_db))
        self._load_species_data(species_db)
        self._initialized = True

    def _load_species_data(self, species_db: Optional[str]):
        """Loads species data from a file."""
        if species_db is None:
            species_db = os.path.join(os.path.dirname(__file__), "inputs", "allometrics_extended.json")
        species_filename = species_db
        try:
            f = open(species_filename)
        except IOError as e:
            try:
                species_filename = os.path.join(os.path.dirname(__file__), "inputs", "allometrics.json")
                f = open(species_filename)
            except IOError as e:
                raise NameError("Error: {}".format(str(e)))
            
        print("Found allometrics file! Using: ", species_filename)
        
        self.species_data: Dict = json.loads(f.read())
        f.close()

    def get_diameter_growth(self, species_name):
        """Retrieve annual avg diameter growth rate for the site for the given species.

        Args:
            species: (:obj:`string`): name of the species in 'genusName_speciesName' format. Ex
            'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            (:obj:`float`): the standard growth per year in cm.
        """
        out = 0.8382
        if species_name in self.species_data.keys():
            out = self.species_data[species_name]["diameter_growth"]
        return out
    
    def get_regeneration_probability(self, species_name):
        """Retrieve regenaration probability for the given species at the site.
        This probability can be used as regeneration liklihood naturally while the site is a forest.
        In case of urban trees this the information about the likelihood of replantation of a dead tree
        at the site. 

        Args:
            species_name: (:obj:`string`): name of the species in 'genusName_speciesName' format. Ex
            'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            (:obj:`float`): the replacement/regeneration probability.
        """
        probability = 1.0
        if species_name in self.species_data.keys():
            specs = self.species_data[species_name]
            if "regeneration_probability" in specs.keys():
                probability = specs["regeneration_probability"]
        return probability
        
    def get_eqn_sapling_height(self, species_name):
        is_evergreen = self.is_evergreen(species_name)

        if is_evergreen:
            return 3.302, self.fit_polynomial({
                # c + bx + ax^2
                'B0': 0.17109132068269153, 'B1': 1.307318659081205, 'B2': -0.18231865908120493,
                'DBHMin': 0.1, 'DBHMax': 3.302
            }, should_convert=False)
        else:
            return 2.54, self.fit_polynomial({
                'B0': 0.05885752915328711, 'B1': 2.4466621598783167, 'B2': -0.35237451411188114, 
                'DBHMin': 0.1, 'DBHMax': 2.54
            }, should_convert=False)

    INCH_TO_CM = 2.54 # 1 inch = 2.54 cm
    MIN_HEIGHT = 0.3 #m
    MIN_CROWN_HEIGHT = 0.3 #m

    def get_eqn_sapling(self, species_name, func_type="crown_height"):
        
        # general fitting function
        def poly_func(x, a, b, c):
            return a + b * x + c * x**2

        func, params = self.get_eqn(species_name, func_type)
        if self.is_evergreen(species_name):
            DBHMin = params.get("DBHMin", 1.3) * self.INCH_TO_CM
        else:
            DBHMin = params.get("DBHMin", 1.0) * self.INCH_TO_CM

        # Checking the cache: FITTED_CURVES
        if species_name in FITTED_CURVES.keys():
            if func_type in FITTED_CURVES[species_name].keys():
                return DBHMin, FITTED_CURVES[species_name][func_type]

        # Calculate the midpoint
        midpoint = (DBHMin / 2, func(DBHMin) / 1.5) ################### Important for calibration ###################

        # Generate the data points
        dbh_range = np.array([0.1, midpoint[0], DBHMin])
        crownH_range = np.array([self.MIN_CROWN_HEIGHT, midpoint[1], func(DBHMin)])

        # Fit the quadratic polynomial function to the data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(poly_func, dbh_range, crownH_range)
            
        out = {
            "B0": params[0],
            "B1": params[1],
            "B2": params[2],
            "DBHMin": 0.1,
            "DBHMax": DBHMin,
        }
        # print(f"{species_name} - {func_type}: {out['B0']} + {out['B1']}x + {out['B2']}x^2")
        
        # a + bx + cx^2
        sapling_function = self.fit_polynomial(
            {
                "B0": params[0],
                "B1": params[1],
                "B2": params[2],
                "DBHMin": 0.1,
                "DBHMax": DBHMin,
            },
            should_convert=False,
        )

        # Update the cache
        if species_name not in FITTED_CURVES.keys():
            FITTED_CURVES[species_name] = {}
        FITTED_CURVES[species_name][func_type] = sapling_function
        return DBHMin, sapling_function

    def get_height_at_maturity(self, species_name):
        """Observed avg total height of the tree for the given species.

        Args:
            species: (:obj:`string`): name of the species in 'genusName_speciesName' format. Ex
            'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            (:obj:`int`): Avg height in meters.
        """
        if species_name in self.species_data.keys():
            if "height_at_maturity" not in self.species_data[species_name].keys():
                return 20
            return self.species_data[species_name]["height_at_maturity"]
        else:
            return 20
        
    def get_maturity_age(self, species_name):
        """Observed avg maturity age of the given species."""
        if species_name in self.species_data.keys():
            if "maturity_age" not in self.species_data[species_name].keys():
                return 20
            return self.species_data[species_name]["maturity_age"]
        else:
            return 20

    def list_species_names(self) -> List[Tuple[str, str]]:
        """
        List existing species names (both Latin and common) with association to Latin names.

        Returns:
            List[Tuple[str, str]]: List of tuples where each tuple is (name, Latin name).
                                Name can be either a Latin name or a common name.
        """
        species_pairs = []
        for latin_name, species_info in self.species_data.items():
            species_pairs.append((latin_name, latin_name))
            if "common_names" in species_info.keys():
                for common_name in species_info["common_names"]:
                    species_pairs.append((common_name, latin_name))
            
            if "common_name" in species_info.keys():
                species_pairs.append((species_info["common_name"], latin_name))

        return species_pairs


    def fuzzymatch(self, species_input):
        """
        Fuzzy matching species name.

        Returns:
            string: Latin name of the species that has the highest matching score.
                    If the best matching has a poor score (<40% similarity), European Birch is returned as a default growth model
        """
        name_pairs = self.list_species_names()
        names, latin_names = zip(*name_pairs)  # Separate the names and their corresponding Latin names

        highest, score = process.extractOne(species_input, names)
        
        if score < 60:
            #print("Found {} to be the best match for the input {} with a score of {}.".format(highest, species_input, score))
            return "decidu"
        
        # Get the Latin name corresponding to the highest match
        index = names.index(highest)
        return latin_names[index]
    
    def is_evergreen(self, species_name):
        """Returns True if the species is evergreen, False if it is deciduous or unknown.
        """
        if species_name in self.species_data.keys():
            if "is_evergreen" not in self.species_data[species_name].keys():
                return False
            return self.species_data[species_name]["is_evergreen"]
        else:
            return False

    def get_leaf_type(self, species_name):
        """Returns the leaf type of the species: broadleaf or needles
        """
        if species_name in self.species_data.keys():
            if "leaf_type" not in self.species_data[species_name].keys():
                return "broadleaf"
            return self.species_data[species_name]["leaf_type"]
        else:
            return "broadleaf"

    def get_eqn(self, species_name, allometry_type):
        """The method retrieves parameters of a given growth function
        and sets its constant paramters and returns a function to be used
        by the tree agents.

        Args:
            species_name: (:obj:`string`): name of the species
            allometry_type: (:obj:`string`): type of of growth can be height, canopy_width
                canopy_height.

        Returns:
            (:obj:`f(string)->float`): the growth function

        """
        eq_type, params = self.get_form_and_constants(species_name, allometry_type)
        if eq_type == "exponential":
            return Species.fit_exponential(params), params
        elif eq_type == "polynomial":
            return Species.fit_polynomial(params), params
        elif eq_type == "parametric":
            return Species.fit_parametric(params), params
        elif eq_type == "linear":
            return Species.fit_linear(params), params
        elif eq_type == "log_linear":
            return Species.fit_log_linear(params), params
        else:
            raise NameError(
                "Equation {} for {} type is not implemented.".format(
                    eq_type, allometry_type
                )
            )

    def get_eqn_biomass(self, species_name):
        """The method retrieves constants of a bimomass function for the given species
        and returns the species specific function.

        Args:
            species_name: (:obj:`string`): name of the species

        Returns:
            (:obj:`f(string)->float`): the biomass function

        Note:
            Refactoring note: Consider to re implement this by the generic function above.
        """
        eq_type, params = self.get_form_and_constants(species_name, "biomass")
        A = params["A"]
        B = params["B"]
        C = params["C"]

        if eq_type == "mass_1":
            return functools.partial(mass_1_biomass_function, A=A, B=B, C=C)
        elif eq_type == "mass_2":
            return functools.partial(mass_2_biomass_function, A=A, B=B, C=C)
        else:
            raise NameError("Equation {} for biomass is not implemented.".format(eq_type))
        
    def get_form_and_constants(self, species_name, allometry_type):
        """The method retrieves parameters and type of a growth function for
        the given species.

        Args:
            species_name: (:obj:`string`): name of the species
            allometry_type: (:obj:`string`): type of of growth can be height, canopy_width
                canopy_height.

        Returns:
            (:obj:`(string, dict`)
        """
        form = self.species_data[species_name]["equations"][allometry_type][
            "equation_type"
        ]
        params = self.species_data[species_name]["equations"][allometry_type]["params"]
        return (form, params)

    @staticmethod
    def filter_dbh_size(dbh, minv, maxv):
        """Utility function to assure the range of dbh that can be used by the growth functions"""
        # converting the dbh in cm into inche
        dbh = max(minv, 0.393700787 * dbh)
        return min(maxv, dbh)

    @staticmethod
    def fit_polynomial(params, should_convert=True):
        """Static method that sets the constant of a second degree polynomial function."""
        B0 = params["B0"]
        B1 = params["B1"]
        B2 = params["B2"]
        DBHMin = params["DBHMin"]
        DBHMax = params["DBHMax"]

        return functools.partial(polynomial_fit, B0=B0, B1=B1, B2=B2, DBHMin=DBHMin, DBHMax=DBHMax, should_convert=should_convert)

    @staticmethod
    def fit_exponential(params):
        """Static method that sets the constant of the exponential function."""
        B0 = params["B0"]
        B1 = params["B1"]
        DBHMin = params["DBHMin"]
        DBHMax = params["DBHMax"]

        return functools.partial(exponential_fit, B0=B0, B1=B1, DBHMin=DBHMin, DBHMax=DBHMax)

    @staticmethod
    def fit_parametric(params):
        """Static method that sets the constant of the exponential function."""
        B0 = params["B0"]
        B1 = params["B1"]
        B2 = params["B2"]

        return functools.partial(parametric_fit, B0=B0, B1=B1, B2=B2)

    @staticmethod
    def fit_linear(params):
        """Static method that sets the constant of the linear function."""
        B0 = params["B0"]
        B1 = params["B1"]
        DBHMin = params["DBHMin"]
        DBHMax = params["DBHMax"]

        return functools.partial(linear_fit, B0=B0, B1=B1, DBHMin=DBHMin, DBHMax=DBHMax)
    
    @staticmethod
    def fit_log_linear(params):
        """Static method that sets the constant of the linear function."""
        B0 = params["B0"]
        B1 = params["B1"]
        DBHMin = params["DBHMin"]
        DBHMax = params["DBHMax"]

        return functools.partial(log_linear_fit, B0=B0, B1=B1, DBHMin=DBHMin, DBHMax=DBHMax)