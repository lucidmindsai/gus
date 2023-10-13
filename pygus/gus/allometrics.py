"""The module holds implementation of classes that handle identification and selection of right allomteric parameters and equations."""

# Importing Python Libraries
import numpy as np
import json
from fuzzywuzzy import process


class Species:
    """Object that holds standard tree growth rate by species at different sites.
    Source: https://database.itreetools.org/#/speciesSearch

    Todo:
        Consider to convert this into a db or API.
    """

    # Ref: Root to shoot ratio (Cairns et al. 1997)
    root_to_shoot_ratio = 0.26

    def __init__(self, species_db):
        """The constructor method.

        Args:
            species_db: (:obj:`str`): File name that holds parameters for allometrics of
            the species used in the models. Eg:'./gus/inpurs/allometrics.json'

        Returns:
            None
        """

        species_filename = species_db
        f = open(species_filename)
        self.parameters = json.loads(f.read())

    def get_diameter_growth(self, species):
        """Retrieve annual avg diameter growth rate for the site for the given species.

        Args:
            species: (:obj:`string`): name of the species in 'genusName_speciesName' format. Ex
            'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            (:obj:`float`): the standard growth per year in cm.
        """
        if species in self.parameters.keys():
            return self.parameters[species]["diameter_growth"]
        else:
            # Return moderate growth rate.
            return 0.8382

    def get_height_at_maturity(self, species):
        """Observed avg total height of the tree for the given species.

        Args:
            species: (:obj:`string`): name of the species in 'genusName_speciesName' format. Ex
            'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            (:obj:`int`): Avg height in meters.
        """
        if species in self.parameters.keys():
            return self.parameters[species]["height_at_maturity"]
        else:
            # Return moderate growth rate.
            return 25

    def list_species(self):
        """List existing species whose carbon related parameters exist within the library.

        Args:
            None

        Returns:
            (:obj:`list` of `String`): List of species names.
        """
        return list(self.parameters.keys())

    def fuzzymatching(self, species):
        """Fuzzy matching species name.

        Args:
            species: (:obj:`string`): name of the species

        Returns:
            (:obj:`string`): Species name in 'genusName_speciesName' format
            that has highest matching score.

        Note:
            If the best matching has a poor score (<10% similarity) then
            betula_pendula is passed on as default.
        """
        highest, score = process.extractOne(species, self.list_species())
        if score < 10:
            highest = "betula_pendula"
        return highest

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
            return Species.fit_exponential(params)
        elif eq_type == "polynomial":
            return Species.fit_polynomial(params)
        elif eq_type == "parametric":
            return Species.fit_parametric(params)
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
            return (
                lambda dbh: 1.0
                * (np.e ** (A + B * np.log(dbh) + C / 2))
                / (1 - Species.root_to_shoot_ratio)
            )
        elif eq_type == "mass_2":
            return (
                lambda dbh: 1.0
                * (A * pow(dbh, B + C))
                / (1 - Species.root_to_shoot_ratio)
            )

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
        form = self.parameters[species_name]["equations"][allometry_type][
            "equation_type"
        ]
        params = self.parameters[species_name]["equations"][allometry_type]["params"]
        return (form, params)

    @staticmethod
    def filter_dbh_size(dbh, minv, maxv):
        """Utility function to assure the range of dbh that can be used by the growth functions"""
        # converting the dbh in cm into inche
        dbh = max(minv, 0.393700787 * dbh)
        return min(maxv, dbh)

    @staticmethod
    def fit_polynomial(params):
        """Static method that sets the constant of a second degree polynomial function."""
        B0 = params["B0"]
        B1 = params["B1"]
        B2 = params["B2"]
        DBHMin = params["DBHMin"]
        DBHMax = params["DBHMax"]

        def fit_pol(dbh):
            # converting the dbh in cm into inche
            dbh = min(DBHMax, max(DBHMin, 0.393700787 * dbh))
            estimate = B0 + (B1 * dbh) + (B2 * dbh * dbh)
            # converting into meters
            return max(0, 0.3048 * estimate)

        return fit_pol

    @staticmethod
    def fit_exponential(params):
        """Static method that sets the constant of the exponential function."""
        B0 = params["B0"]
        B1 = params["B1"]
        DBHMin = params["DBHMin"]
        DBHMax = params["DBHMax"]

        def fit_exp(dbh):
            # converting the dbh in cm into inche
            dbh = min(DBHMax, max(DBHMin, 0.393700787 * dbh))
            estimate = np.exp(B0 + B1 * np.log(dbh))
            # converting into meters
            return max(0, 0.3048 * estimate)

        return fit_exp

    @staticmethod
    def fit_parametric(params):
        """Static method that sets the constant of the exponential function."""
        B0 = params["B0"]
        B1 = params["B1"]
        B2 = params["B2"]

        def fit_parametric(dbh):
            estimate = B0 + B1 * (dbh**B2)
            return max(0, estimate)

        return fit_parametric
