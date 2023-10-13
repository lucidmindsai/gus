"""Module that holds implementation of Tree agents."""

import numpy as np

# Mesa Packages
from mesa import Agent


class Tree(Agent):
    """A generic Tree agent with basic structural attributes and a growth model.

    It inherits the generic mesa.Agent class.
    """

    # Def: Carbon storage cap in sequestration calculations. A tree with that amount of CARBON or above this level
    # has a constant yearly sequestration which is 25kg/year.
    # Unit: Kg
    # Ref: iTree, 2020
    carbon_storage_cap = 7500
    sequestration_at_maturity = 25

    # The coeff is used to estimate carbon storage through biomass.
    # The stock of carbon is estimated by multiplying tree biomass by 0.5
    # (Chow and Rolfe 1989).
    carbon_coeff = 0.5

    # The coeff is used as the generic decomposition rate of dead portion of a tree.
    # For more accurate estimates on decompostion revise the methods at Nowak et al. (2002b, 2008)
    decomposition_coeff = 0.1

    # Fixed rates for crown light exposure to sunlight (CLE).
    # 0.44: Forest conditions with a closed, or nearly closed canopy,
    # 0.56: Park conditions
    # 1.0:  Open-grown conditions, street trees.
    # Source: iTree
    sun_exposure_rates = {"forest": 0.44, "park": 0.56, "street": 1.0, "pocket": 0.56}

    # Condition multipliers. Used at adjusting growth rates.
    # Ref: Fleming, 1988, and Nowak 2002b
    condition_multiplier = {
        "excellent": 1,
        "good": 1,
        "fair": 1,
        "poor": 0.76,
        "critical": 0.42,
        "dying": 0.15,
        "dead": 0,
    }

    # Ref: Root to shoot ratio (Cairns et al. 1997)
    root_to_shoot_ratio = 0.26

    # Biomass ratio at crown to trunk: Needs validation
    crown_to_trunk_ratio = 0.05

    def __init__(
        self,
        unique_id,
        model,
        dbh,
        species,
        height=None,
        kind="deciduous",
        fixed_sun_exposure=False,
        condition=None,
        dieback=None,
    ):
        """The constructor method.

        Args:
            unique_id: (:obj:`int`): a unique agent id
            model: (Mesa.Model): the underlying Mesa model.
            dbh: (:obj:`float`): the DBH measure which is diameter in cm of the trunk
                usually measured at 1.3m from the ground.
            species: (:obj:`str`): species identifier
            height: (:obj:`float`): The tree height in meters.
            kind: (:obj:`str`): identifies the kind of tree can be either "deciduous" or
                "coniferous" (needles)
            fixed_sun_exposure: (:obj:`bool`): If sun exposure is fixed or to be checked at each
                iteration.
            condition: (:obj:`str`): The health condition of the tree.
            dieback: (:obj:`float`): The percent (0.1 for 10%, etc) of the tree that is dead.

        Returns:
            None
        """

        # Initializing parent Class.
        super().__init__(unique_id, model)

        # Initializing variables for a Tree.
        self.model = model
        self.kind = kind
        self.species = model.species.fuzzymatching(species)
        self.dbh = dbh
        self.fixed_sun_exposure = fixed_sun_exposure
        self.overlap_ratio = 0

        # Initialize canonical growth functions
        self.f_tree_height = self.model.species.get_eqn(self.species, "height")
        self.f_biomass = self.model.species.get_eqn_biomass(self.species)
        self.f_crown_width = self.model.species.get_eqn(self.species, "crown_width")
        self.f_crown_height = self.model.species.get_eqn(self.species, "crown_height")

        # Record initial allometries
        if height:
            self.tree_height = height
        else:
            self.tree_height = self.f_tree_height(self.dbh)
        self.crown_width = self.f_crown_width(self.dbh)
        self.crown_height = self.f_crown_height(self.dbh)

        # dieback related initializations:
        # Note: this needs to be handled at the initialization module
        self.dieback = 0
        self.condition = "excellent"
        if dieback and condition:
            self.condition = condition
            self.dieback = dieback
        elif dieback:
            self.condition = self._get_condition_class(dieback)
            self.dieback = dieback
        elif condition:
            self.condition = condition
            self.dieback = self._estimate_dieback(condition)
        else:
            self.dieback = np.random.uniform(0, 0.1)
            self.condition = self._get_condition_class(self.dieback)

        self.diameter_growth = self.model.species.get_diameter_growth(species)
        # Slow, moderate and fast growing species respectively.
        # c(0.23, 0.33, 0.43) in inch/yr Source: https://database.itreetools.org/#/splash
        # Converted into cm.

        # Default crown light exposure based on site types.
        self.cle = Tree.sun_exposure_rates[self.model.project_site_type]
        # Crown light exposure to sunlight (CLE).
        # CLE <- c(0.44, 0.56, 1)
        # (1) Forest conditions with a closed, or nearly closed canopy,
        # (2) Park conditions
        # (3) Open-grown conditions.

        self.average_height_at_maturity = self.model.species.get_height_at_maturity(
            self.species
        )
        # Avg height at maturity for the given species.

        self.biomass = self.compute_biomass()  # In Kg
        self.carbon_storage = Tree.carbon_coeff * self.biomass  # In Kg

        # Amount of carbon release due to dead portion.
        self.decomposition = 0  # In Kg

        # Annual carbon sequestration in Kg.
        self.annual_gross_carbon_sequestration = 0

        self.mulched = 0
        self.decomposing_root = 0
        self.decomposing_trunk = 0
        self.immediate_release = 0
        self.death_acc = False

        if self.model.maintenance_scope == 0:
            self.expected_care = 0.1
        elif self.model.maintenance_scope == 1:
            self.expected_care = 0.5
        else:
            self.expected_care = 0.9

    def step(self):
        """State transitions of a given Tree agent.

        Args:
            None

        Returns:
            None

        Note:
            None

        Todo:
            None
        """
        # Once the replaced tree agents are removed from the model, this line will be idle and s
        # should be removed too.
        if self.condition == "replaced":
            return

        # The tree is dead and its carbon release schedule is accounted.
        if self.death_acc:
            self._reset_release_track()
            if np.random.uniform(0, 1) < self.expected_care:
                self.replace()
            return

        if self.condition == "dead":
            self.compute_decomposition()
            return

        # check frost free days for the past year.
        frost_free_days = self.model.WeatherAPI.check_frost_free_days()
        # print('Tree: {} checks ffdays = {} ...'.format(self.unique_id, frost_free_days))

        # compute the light exposure
        self.compute_light_exposure()

        # check state of the health of the tree
        # print('Tree: {} checks dieback ...'.format(self.unique_id))
        self.check_dieback()

        # compute the growth
        # print('Tree: {} grows ...'.format(self.unique_id))
        self.grow(frost_free_days)

        # compute the total biomass
        self.compute_biomass()
        # print('Tree: {} biomass ...'.format(self.unique_id))

        # compute the amount of new carbon sequestration
        self.compute_sequestration()
        # print('Tree: {} sequestration ...'.format(self.unique_id))

        # compute the amount of carbon release due to decomposition
        # print('Tree: {} decomposition ...'.format(self.unique_id))
        self.compute_decomposition()

    def grow(self, frost_free_days):
        """The method updates DBH of the tree. Currently it is an annual growth in cm.

        Args:
            frost_free_days: (:obj:`int`): the number of observed frost free days

        Returns:
            None

        Note:
            Current version uses the growth model, where
            the number of frost free days and species specific diameter growth factor is
            sufficient for the growth mode. Site specific and species specific diameter growth needs
            to be considered.

            As a tree approaches “maximum” height, growth rate decreases. Thus,the species growth rates
            are adjusted based on the ratio between the current height of the tree and the average height
            at maturity for the species. The estimated tree height at maturity is derived from the literature.
            When a tree’s height is more than 80 percent of its average height at maturity,
            the annual diameter growth is proportionally reduced from full growth at 80 percent of maximum height
            to 2.22 percent of full growth at 125 percent of height at maturity.

        Todo:
            The constants in the formulas below needs to be parameterized.
            References for the current constants:
            David J. Nowak, 2020. Understanding i-Tree: Summary of Programs and Methods
        """

        # This equation need to be corrected. Check i-tree paper.
        height_ratio = self.tree_height / self.average_height_at_maturity
        if height_ratio >= 1.25:
            delta_dbh = 0.0222
        elif height_ratio >= 0.8:
            # delta_dbh = 0.0222 + ((1 - 0.0222) / (1.25 - 0.8)) * (1.25 - height_ratio)
            delta_dbh = 0.0222 + 2.1729 * (1.25 - height_ratio)
        else:
            delta_dbh = self.diameter_growth * self.cle * (1 - self.dieback)

        # Adjust the growth rate according to health condition.
        delta_dbh *= Tree.condition_multiplier[self.condition]
        self.dbh += delta_dbh * (frost_free_days / 153)

        # Update the tree height based on the updated dbh.
        self.update_tree_height(generic=False)
        # Update the change at canopy.
        self.update_crown_width()
        self.update_crown_height()

    def estimate_tree_height(self):
        """Computes the height of tree based on the species and current dbh.

        Args:
            None

        Returns:
            None
        Todo:
            In case of absence of a height function a generic function is used.
            Current generic function is based on family of height functions yet still
            arbitrary. It needs to be updated by the higher level phenotypes (needle, leaf, etc.)

        """
        self.tree_height = self.f_tree_height(self.dbh)
        return self.tree_height

    def update_tree_height(self, generic=False):
        """Computes the height of tree based on the species and current dbh.

        Args:
            None
        Returns:
            None

        """
        if generic:
            self.fleming_height()
            return
        self.tree_height = self.f_tree_height(self.dbh)

    def fleming_height(self):
        """Updates the tree height based on the model by Fleming (1988)

        Args:
            None
        Returns:
            None
        """
        self.tree_height += Tree.condition_multiplier[self.condition] * 0.15

    def update_crown_height(self):
        """Computes the vertical length of the tree crown based on
        the species and its current dbh.

        Args:
            None
        Returns:
            (:obj:`float`): Current crown width in meters
        """
        self.crown_height = self.f_crown_height(self.dbh)
        return self.crown_height

    def update_crown_width(self):
        """Computes the horizontal length of the tree crown based on
        the species and its current dbh.

        Args:
            None
        Returns:
            (:obj:`float`): Current crown width in meters
        """
        self.crown_width = self.f_crown_width(self.dbh)
        return self.crown_width

    def compute_light_exposure(self):
        """The method computes light exposure of a tree on a dynamic manner.
        The tree checks the state of its neighbouring trees and determines its current CLE.

        Args:
            None

        Returns:
            None

        Note:
            Current version checks out other trees within the same grid.
            It checks height and width of each neighboring tree.
            Canopy overlap ratio is used as proxy to determine available CLE. The model assumes that
            a taller neighbourung tree reduces sun light exposure.

        Todo:
            * The assumption needs to be revisited and validated.
            * The model needs to be revised in case the same location is shared by other species.
        """
        if self.fixed_sun_exposure:
            return

        # cellmates = self.model.grid.get_cell_list_contents([self.pos])
        posns_list = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False, radius=1
        )
        neighboors = self.model.grid.get_cell_list_contents(posns_list)
        self.overlap_ratio = 0
        combined_overlap = 0
        for t in neighboors:
            t_w = t.crown_width
            t_h = t.tree_height
            # 0.5 multiplier is a mean multiplier to correct square shaped assumption on tree crown shape.
            overlap = max(0, 0.5 * (self.crown_width + t_w) - self.model.dt_resolution)
            # 0.25 multiplier is to account one of the four sides of the grid cell.
            overlap_ratio = 0.25 * min(1, (overlap / self.crown_width))
            # a taller tree creates more shading
            combined_overlap += overlap_ratio * (t_h / (t_h + self.tree_height))
            self.overlap_ratio += overlap_ratio
        # Cases needs to be inspected
        self.overlap_ratio = min(1, self.overlap_ratio)
        light_loss_multiplier = 0.75  # arbitrary to be fixed with empirical data.
        self.cle = max(0, 1 - light_loss_multiplier * combined_overlap)

        # try:
        #     total_dbh = sum([t.dbh for t in cellmates])
        #     cle = self.dbh / total_dbh
        # except ZeroDivisionError as err:
        #     print(self.unique_id, err)
        #     cle = 0.56
        # self.cle = cle

    def compute_contagion_risk(self):
        """A very simplified version of contagion.

        Args:
            None
        Returns:
            (:obj:`float`):  contagion risk a value within the inclusive range [0,0.9]
        """
        posns_list = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False, radius=1
        )
        neighboors = self.model.grid.get_cell_list_contents(posns_list)
        count_neighboors = len(neighboors)
        if count_neighboors == 0:
            return 0

        contagion_risk = 0
        for atree in neighboors:
            contagion_risk += atree.dieback
        if count_neighboors > 4:
            contagion_risk /= count_neighboors
        else:
            contagion_risk /= 4
        # TODO: 0.9 is an adjustment parameter that needs calibration.
        return 0.9 * contagion_risk

    def check_dieback(self, stochastic=True, risk_rate=0.005, healing_rate=0.005):
        """The dieback calculation for the tree. It is a path dependent.
        A 'random walk' from latest state is considered. The new rate is drawn from
        a uniform distribution between -1 * healing_rate and risk_rate. A tree with
        a dieback ratio 1 can not recover.

        Args:
            risk_rate: (:obj:`float`): A mean risk rate used to draw a random dieback ratio.
            healing_rate: (:obj:`float`): A mean healing rate that is used to model recovery rate
            of a tree from a sickness.

        Returns:
            (:obj:`float`): dieback ratio.

        Todo:
            Update the data either using measured dieback for species or
            a site/species specific distribution function.
        """

        def register_death():
            self.dieback = 1.0
            self.condition = "dead"

        # The dieback model below is based on Nowak et al (1986, 2002b:p13)
        # Mortality rate:
        #  100% for dead trees
        #  1.96% for good-excellent and DBH < 3inches
        #  1.46% for good-excellent and DBH > 3inches
        #  3.32% for fair condition
        #  8.86% for poor condition
        #  13.08% for critical condition
        #  50% for dying condition
        risk = np.random.uniform(0, 1)
        if self.model.maintenance_scope == 0:
            dr = 5
        elif self.model.maintenance_scope == 1:
            dr = 4
        else:
            dr = 1

        if self.dieback >= 1:
            register_death()
        elif self.condition == "dead":
            register_death()
        elif (
            self.condition in ("good", "excellent")
            and self.dbh < 7.62
            and risk <= 0.0196 * dr
        ):
            register_death()
        elif (
            self.condition in ("good", "excellent")
            and self.dbh >= 7.62
            and risk <= 0.0146 * dr
        ):
            register_death()
        elif self.condition == "fair" and risk <= 0.0332 * dr:
            register_death()
        elif self.condition == "poor" and risk <= 0.0886 * dr:
            register_death()
        elif self.condition == "critical" and risk <= 0.1308 * dr:
            register_death()
        elif self.condition == "dying" and risk <= min(0.9, 0.5 * dr):
            register_death()
        else:
            # The dieback model below is path dependent and depends on
            # (i) the latest condition of the tree,
            # (ii) the age via DBH and
            # (iii) the health of neighboring trees.
            # the new rate is drawn from a
            # uniform distribution between -1 * healing_rate and risk_rate.
            if stochastic:
                contagion_risk = self.compute_contagion_risk()
                # multiplier = Tree.condition_multiplier[self.condition] * np.sqrt(self.dbh)
                if self.condition == "excellent":
                    if np.random.uniform(0, 1) < 0.5:
                        self.dieback -= 0.001
                    else:
                        self.dieback += 0.001
                elif self.condition == "good":
                    if np.random.uniform(0, 1) < 0.5:
                        self.dieback -= 0.005
                    else:
                        self.dieback += 0.005
                else:
                    multiplier = (1 - contagion_risk) * np.sqrt(self.dbh) / dr
                    if multiplier > 0.01:
                        heal_range = -1 * multiplier * healing_rate
                        die_range = risk_rate / multiplier
                        # die_range = risk_rate
                        self.dieback += np.random.uniform(heal_range, die_range)
                self.dieback = min(1, self.dieback)
                self.dieback = max(0, self.dieback)
                self.condition = self._get_condition_class(self.dieback)
        return self.dieback

    def _get_condition_class(self, dieback):
        """Determines the condition class based percent crown lost.

        Args:
            dieback: (:obj:`float`): percent crown lost

        Returns:
            (:obj:`str`): condition class.

        Note:
             The class brackets are based on Nowak 2002b.
        """

        if dieback < 0.01:
            condition = "excellent"
        elif dieback <= 0.10:
            condition = "good"
        elif dieback <= 0.25:
            condition = "fair"
        elif dieback <= 0.50:
            condition = "poor"
        elif dieback <= 0.75:
            condition = "critical"
        elif dieback <= 0.99:
            condition = "dying"
        else:
            condition = "dead"
        self.condition = condition
        return condition

    def _estimate_dieback(self, condition):
        """Draws a crown dieback ratio based on the condition class.

        Args:
            dieback: (:obj:`str`): condition class.

        Returns:
            (:obj:`str`): (:obj:`float`): percent crown lost.

        Note:
             This is used when percent crown data is missing but condition of
             of a tree is given a qualitatively. Condition class brackets are based on Nowak 2002b.
        """
        if condition == "excellent":
            self.dieback = np.random.uniform(0, 0.01)
        elif condition == "good":
            self.dieback = np.random.uniform(0.01, 0.11)
        elif condition == "fair":
            dieback = np.random.uniform(0.11, 0.26)
        elif condition == "poor":
            self.dieback = np.random.uniform(0.26, 0.51)
        elif condition == "critical":
            self.dieback = np.random.uniform(0.51, 0.76)
        elif condition == "dying":
            self.dieback = np.random.uniform(0.76, 0.99)
        else:
            self.dieback = 1.0
        return self.dieback

    def compute_biomass(self, ignore_height=True) -> float:
        """The biomass calculation for the tree, in KG.

        Args:
            species: (:obj:`string`): name of the species in 'genusName_speciesName' format.
            Ex: 'picea_abies'. Use the iTree naming scheme: https://database.itreetools.org/#/speciesSearch

        Returns:
            (:obj:`float`): Biomass in Kg.

        Todo:
            Update the generic biomass.

        """
        self.biomass = self.f_biomass(self.dbh)
        carbon_estimate = self.biomass * Tree.carbon_coeff
        if carbon_estimate > Tree.carbon_storage_cap:
            # self.biomass = (1 / Tree.carbon_coeff) * Tree.carbon_storage_cap
            self.biomass = Tree.carbon_storage_cap
        return self.biomass

    def _reset_release_track(self):
        """The resetting state variables that is being observed by data collectors.

        Args:
            None

        Returns:
            None

        """
        self.mulched = 0
        self.decomposing_root = 0
        self.decomposing_trunk = 0
        self.immediate_release = 0

    def compute_decomposition(self):
        """The amount of carbon release in KG due to partial diebacks.

        Args:
            None

        Returns:
            None

        Todo:
            This module on carbon release from alive trees due to partial diebacks or
            crown loss needs to be revised.
        """

        # self.release = self.decomposition_rate * self.dieback * self.carbon_storage
        self._reset_release_track()
        if self.condition == "dead":
            self._compute_decomposition_dead()
            return
        # the tree is alive
        self._compute_decomposition_alive()

    def _compute_decomposition_alive(self):
        """The amount of carbon release in KG due to partial diebacks.

        Args:
            None

        Returns:
            None

        Todo:
            This module on carbon release from alive trees due to partial diebacks or
            crown loss needs to be revised.
        """
        to_decompose = Tree.crown_to_trunk_ratio * self.dieback * self.carbon_storage
        if np.random.uniform(0, 1) < 0.9:
            self.mulched = to_decompose
        else:
            # 100% of the removed dead brunches are burned etc.
            self.immediate_release = 1.0 * to_decompose

    def _compute_decomposition_dead(self):
        """The amount of carbon release in KG due to diebacks.

        Args:
            None

        Returns:
            (:obj:`float`): Amount of carbon release in KG.

        Todo:
            This implementation is based on Nowak et al. (2002b, 2008)
        """

        # self.release = self.decomposition_rate *  self.carbon_storage

        # accounting carbon release process
        self.death_acc = True
        self.decomposing_root = Tree.root_to_shoot_ratio * self.carbon_storage
        decomposable_above_ground = self.carbon_storage - self.decomposing_root

        # the probability of being removed from the site
        if np.random.uniform(0, 1) < 0.5:
            # 70% chance burnt, 30% converted into sustainable products
            self.immediate_release = 0.7 * decomposable_above_ground
            return

        # Not removed from the site:
        # The probability of standing
        if np.random.uniform(0, 1) < 0.4:
            self.decomposing_trunk = decomposable_above_ground
        else:
            self.mulched = decomposable_above_ground
        return

    def compute_sequestration(self):
        """The method estimates annual amount of sequestration in Kg. The model is based on
        Chow and Rolfe (1989) and used by iTree.

        Args:
            None

        Returns:
            (:obj:`float`): annual sequestration in carbon.

        Note:
            The growth of the tree needs to be adjusted for the health condition of a tree
            "For trees in fair to excellent condition, growth rates were multiplied by 1 (no adjustment),
            poor trees’ growth rates were multiplied by 0.76, critical trees by 0.42,
            and dying trees by 0.15 (dead trees’ growth rates = 0).
            Adjustment factors were based on percent crown dieback and the assumption that
            less than 25-percent crown dieback had a limited effect on d.b.h. growth rates."(Nowak et al, 2002b)
            The difference in estimates of C storage between year x and year x+1 is the gross amount of
            C sequestered annually.

        Todo:
            - Raise Warning/Error for unexpected behaviors
        """
        if self.dieback >= 1.0:
            self.annual_gross_carbon_sequestration = 0
            return self.annual_gross_carbon_sequestration

        carbon_storage_lag = self.carbon_storage
        carbon_estimate = self.biomass * Tree.carbon_coeff
        # min() function prevents carbon storage from over estimation for very
        # large trees
        self.carbon_storage = min(carbon_estimate, Tree.carbon_storage_cap)

        if carbon_estimate >= Tree.carbon_storage_cap:
            self.annual_gross_carbon_sequestration = Tree.sequestration_at_maturity
            return self.annual_gross_carbon_sequestration

        sequestration = self.carbon_storage - carbon_storage_lag
        # Double check and raise a warning message here.
        if sequestration < 0:
            sequestration = Tree.sequestration_at_maturity
            self.carbon_storage = carbon_storage_lag

        self.annual_gross_carbon_sequestration = sequestration
        return self.annual_gross_carbon_sequestration

    def replace(self) -> int:
        """In case of a maintenance project in place a new young tree or sapling is planted
        at the location where the tree is dead.

        Args:
            None
        Returns:
            (:obj:`int`): new agent id.

        """
        self.condition = "replaced"

        # Replacing with the site specific minimum sapling.
        dbh = np.random.uniform(self.model.sapling_dbh, self.model.sapling_dbh + 1)
        id = self.model.next_id()
        new_tree = Tree(
            id, self.model, dbh, self.species, condition="excellent", dieback=0
        )
        self.model.grid.place_agent(new_tree, self.pos)
        self.model.schedule.add(new_tree)
        return id
