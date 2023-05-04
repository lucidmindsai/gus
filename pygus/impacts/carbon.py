""" The module that implements biomass and carbon storage. """

# Carbon storage estimation.

from ..gus.allometrics import Species


class Carbon:
    """Species specific biomass and carbon storage estimator."""

    #  Unit: Kg, Ref: iTree, 2020
    storage_cap = 7500

    # The coeff is used to estimate carbon storage through biomass.
    # The stock of carbon is estimated by multiplying tree biomass by 0.5 (Chow and Rolfe 1989).
    coeff = 0.5

    def __init__(self, allometrics, species_name=None):
        """Constructor method.

        Args:
            allometrics (:obj:`str`): The name of the file that keeps allometrics of the tree species for the site.
            species_name: (:obj:`string`): name of the species in 'genusName_speciesName' format.
                Ex: 'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            None
        """
        self.Allometrics = Species(allometrics)
        if species_name:
            self.species_name = self.Allometrics.fuzzymatching(species_name)
            self.f_biomass = self.Allometrics.get_eqn_biomass(species_name)

    def compute_carbon_storage(self, dbh, species_name="decidu", height=None) -> float:
        """The carbon storage calculation for a tree, in KG.

        Args:
            dbh: (:obj:`float`): the DBH measure which is diameter in cm of the trunk usually measured at 1.3m from the ground.
            species_name: (:obj:`string`): name of the species in 'genusName_speciesName' format.
                Ex: 'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch
            height: (:obj:`float`): The tree height in meters.

        Returns:
            (:obj:`float`): Carbon in Kg.

        TODO:
            Update the method to allow receive the height data and use the allometric equations
            with height parameter.

        """
        # Load species composition and their allometrics
        self.species_name = self.Allometrics.fuzzymatching(species_name)
        self.f_biomass = self.Allometrics.get_eqn_biomass(species_name)
        biomass = self.f_biomass(dbh)
        carbon_estimate = biomass * Carbon.coeff
        # min() function prevents carbon storage from over estimation for very large trees
        carbon_storage = min(carbon_estimate, Carbon.storage_cap)
        return carbon_storage

    def compute_biomass(self, dbh, species_name="decidu", height=None) -> float:
        """The carbon storage calculation for a tree, in KG.

        Args:
            species_name: (:obj:`string`): name of the species in 'genusName_speciesName' format.
                    Ex: 'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch
            dbh: (:obj:`float`): the DBH measure which is diameter in cm of the trunk usually measured at 1.3m from the ground.
            height: (:obj:`float`): The tree height in meters.

        Returns:
            (:obj:`float`): Carbon in Kg.

        TODO:
            Convert and extend this into carbon estimation object.

        """
        carbon_estimate = self.compute_carbon_storage(dbh, species_name)
        return carbon_estimate / Carbon.coeff
