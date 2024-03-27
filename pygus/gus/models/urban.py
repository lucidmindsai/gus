""" The module holds the main objects that manage and handle simulation runtime and data collection. """

# Importing Python Libraries
import json
import time
from typing import Dict, Union
import pandas as pd
import numpy as np
from functools import reduce
import logging

# Importing necessary Mesa packages
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Importing needed GUS objects
from .config import ScenarioConfig, SiteConfig
from ..utilities import latlng_array_to_xy
from ..enums import HealthCondition
from ..agents import Tree
from ..allometrics import Species
from ..weather import WeatherSim

# logging.basicConfig(level=logging.DEBUG)

class Urban(Model):
    """A generic urban green space model. To be tailored according to specific sites."""

    # Used to hold the scaling of actual physical space within the digital space.
    # It shows the size of each cell (square) in meters.
    # FIXME: this needs to be brought up as a parameter and placed within the groups
    #      of parameters that handle physical to digital twin mapping.
    # dt_resolution = 2  # in meters

    def __init__(
        self,
        population: Union[str, pd.DataFrame],
        species_allometrics: Union[str, Species],
        site_config: Union[str, SiteConfig],
        scenario: Union[str, ScenarioConfig],
    ):
        """The constructor method.

        Args:
            population: (:obj:`pd.DataFrame`): A dataframe tree properties are read from a site.
            species_composition (:obj:`str`): The name of the file that keeps allometrics of the tree species for the site.
            site_config: (:obj:`string`): name of the json file.
            scenario: (:obj:`dict`): Python dictionary that holds experiment parameters.
        Returns:
            None

        Note:
            First release model.

        Todo:
            Check for hard coded constants and parameterize further.
        """

        ############################################################
        # INPUTS
        ############################################################

        super().__init__()
        self.df = self._handle_population_input(population)
        self.allometrics = Species(species_allometrics) if isinstance(species_allometrics, str) else species_allometrics
        self._handle_site_configuration(site_config, len(self.df))
        self._handle_scenario_configuration(scenario)

        ############################################################
        ############################################################

        # If the dataset has no xpos and ypos columns, use the utility method to generate them
        if "xpos" not in self.df.columns or "ypos" not in self.df.columns:
            latlng_array_to_xy(self.df)

        # Setting MESA specific parameters
        width = int(max(self.df.xpos)) + 1
        length = int(max(self.df.ypos)) + 1

        logging.debug("Building grid with size: {} x {} ...".format(width, length))
        self.grid = MultiGrid(width, length, torus=False)
        # to be parameterized and set during initialization.

        # Load species composition and their allometrics
        required_cols = ["id", "dbh", "height", "species", "condition", "xpos", "ypos"]

        # Test that the df is complete or raise keyerror
        for attribute in required_cols:
            self.df[attribute]

        # list any extra columns that are present in the input data
        extra_columns = list(set(self.df.columns) - set(required_cols))

        self.num_agents = len(self.df)
        self.schedule = RandomActivation(self)

        self.sapling_dbh = min(self.df.dbh)
        # Each entry index i, represents number of years since the biomass is decay period.
        self.release_bins = {
            "slow": np.zeros(10),  # for dead root and standing tree.
            "fast": np.zeros(10),  # for mulched biomass
        }

        # Create agents.
        count = 0
        for index, row in self.df.iterrows():
            a = Tree(
                row.id,
                self,
                dbh=row.dbh,
                species=row.species,
                condition=str(row.condition).lower(),
                xpos=row.xpos,
                ypos=row.ypos,
                **{col: row[col] for col in extra_columns}
            )
            self.schedule.add(a)
            count += 1

            # Place trees on the plot based on actual physical positioning
            x = row.xpos
            y = row.ypos
            logging.debug("Placing agent {} at ({},{})".format(index, x, y))
            self.grid.place_agent(a, (x, y))
        logging.info(f"Placed {count} tree agents")

        # This variable below works as an indexer while adding new trees to the population during the run time.
        self.current_id = max(self.df.id)

        ALIVE_STATE = ["excellent", "good", "fair", "poor", "critical", "dying"]
        # Collecting model and agent level data
        self.datacollector = DataCollector(
            model_reporters={
                "Storage": lambda m: self.aggregate(m, "carbon_storage"),
                "Seq": lambda m: self.aggregate(m, "annual_gross_carbon_sequestration"),
                # Avg sequestered carbon per tree is annual carbon divided by number of living trees
                "Avg_Seq": (lambda m: self.aggregate(
                    m, "annual_gross_carbon_sequestration"
                )
                / self.count(
                        m,
                        "condition",
                        lambda x: x in ALIVE_STATE,
                    )
                ),
                "Released": self.compute_current_carbon_release,
                # Avg released carbon per year is annual carbon release divided by number of living trees
                "Avg_Rel": lambda m: self.compute_current_carbon_release(m)
                    / self.count(
                        m,
                        "condition",
                        lambda x: x in ALIVE_STATE,
                    ),
                "Alive": lambda m: self.count(
                    m, "condition", lambda x: x in ALIVE_STATE
                ),
                "Dead": lambda m: self.count(m, "condition", lambda x: x == "dead"),
                "Critical": lambda m: self.count(
                    m, "condition", lambda x: x == "critical"
                ),
                "Dying": lambda m: self.count(m, "condition", lambda x: x == "dying"),
                "Poor": lambda m: self.count(m, "condition", lambda x: x == "poor"),
                "Replaced": lambda m: self.count(
                    m, "condition", lambda x: x == "replaced"
                ),
                "Seq_std": self.agg_std_sequestration,
            },
            agent_reporters={
                "species": "species",
                "dbh": "dbh",
                "height": "tree_height",
                "crownH": "crown_height",
                "crownW": "crown_width",
                "canopy_overlap": "overlap_ratio",
                "cle": "cle",
                "condition": "condition",
                "dieback": "dieback",
                "biomass": "biomass",
                "seq": "annual_gross_carbon_sequestration",
                "carbon": "carbon_storage",
                "deroot": "decomposing_root",
                "detrunk": "decomposing_trunk",
                "mulched": "mulched",
                "burnt": "immediate_release",
                "xpos": "xpos",
                "ypos": "ypos",
                "est_age": "age",
                "lat": "lat",
                "lng": "lng",
                **{col: col for col in extra_columns},
            },
        )

        logging.info(
            "Initialisation of the Digital Twins of {} trees on a {} by {} digital space is complete!".format(
                self.num_agents, width, length
            )
        )

    def run(self, steps=None):
        """Customized MESA method that sets the major components of scenario analyses process."""
        pop = str(self.df.shape[0])
        if not steps:
            steps = self.time_horizon
        logging.debug("Running simulation ({} years)...".format(steps))
        start = time.time()
        for _ in range(steps):
            logging.info("Year:{}".format(self.schedule.time + 1))
            self.step()
        end = time.time()
        logging.debug(
            "{} steps completed (pop. {}): {}".format(steps, pop, end - start)
        )
        logging.info("Simulation is complete!")

        return self.impact_analysis()

    def step(self):
        """Customized MESA method that sets the major components of scenario analyses process."""
        logging.info("Year:{}".format(self.schedule.time + 1))
        self.get_weather_projection()

        logging.info("Agents are working ...")
        self.schedule.step()

        logging.info("Yearly data is being collected ...")
        self.datacollector.collect(self)

        # print('Step:{} ({}s)'.format(self.schedule.time, end-start))
        # print(self.release_bins['slow'])
        # print(self.release_bins['fast'])

    def impact_analysis(self) -> pd.DataFrame:
        """
        Provides impact analysis of the simulation
        """
        df_out_site = self.datacollector.get_model_vars_dataframe()
        return Urban.format_impact_analysis(df_out_site)

    def get_agent_data(self) -> pd.DataFrame:
        """
        Provides agent data of the simulation
        """
        df_out_agents = self.datacollector.get_agent_vars_dataframe()
        return df_out_agents
    
    def _handle_population_input(
            self, population: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load population dataframe if necessary"""
        if isinstance(population, str):
            population = pd.read_csv(population)

        return population

    def _handle_site_configuration(
        self, site_config: Union[str, SiteConfig], population_size: int
    ):
        """Loads site configuration information."""
        if isinstance(site_config, str):
            site_config = SiteConfig.from_file(site_config)

        self.growth_season_mean = site_config.weather.growth_season_mean
        self.growth_season_var = site_config.weather.growth_season_var
        self.project_site_type = site_config.project_site_type
        self.dt_resolution = round(
            np.sqrt(1 / (population_size / site_config.total_m2)),
            2,  # round to < decimal places
        )

    def _handle_scenario_configuration(self, scenario: Union[str, ScenarioConfig]):
        """Loads site configuration information."""
        if isinstance(scenario, str):
            scenario = ScenarioConfig.from_file(scenario)

        self.maintenance_scope = scenario.maintenance_scope
        self.time_horizon = scenario.time_horizon_years

    def get_weather_projection(self):
        """The method retrieves wetaher projection for the current iteration,
        eg. year, at the moment it uses a simulated projections on the go,
        instead previously computed estimates or forecasts can be used as well.

        Args:
            None

        Returns:
            None
        Note:
            The avg season length is based on iTree Glasgow estimates (see iTree 2015 report)
            These variables need to be parameterrized and read-in through site (model) specific
            initialization module.
        Todo:
            * Implement a site initialization method that sets the initial parameters
            and tree specific variables.
        """
        # The avg season length is based on iTree Glasgow estimates (see iTree 2015 report)
        # These variables need to be parameterrized and read-in through site (model) specific
        # initialization module.
        # season_mean = 200

        self.WeatherAPI = WeatherSim(
            season_length=self.growth_season_mean, season_var=self.growth_season_var
        )

    @staticmethod
    def aggregate(model, var, func=lambda x, y: x + y, init=0):
        """A higher order function that aggregates a given variable.

        The aggregation function and the initial conditions can be specified.
        """
        agent_values = [eval("a.{}".format(var)) for a in model.schedule.agents]
        
        # Filter out nan values from the list
        filtered_values = [value for value in agent_values if not np.isnan(value)]
        
        # Use reduce on the filtered list
        return reduce(func, filtered_values, init)

    @staticmethod
    def count(model, memory, predicate):
        """A higher order function for counting agents that satisfy a specific attribute."""
        return len(
            list(
                filter(
                    predicate,
                    [eval("a.{}".format(memory)) for a in model.schedule.agents],
                )
            )
        )

    @staticmethod
    def aggregate_sequestration(model):
        """The function that accumulates yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        Note:
            None
        Todo:
            None
        """
        captured = [a.annual_gross_carbon_sequestration for a in model.schedule.agents]
        return sum(captured)

    @staticmethod
    def agg_std_sequestration(model):
        """The function estimates the standard deviation for yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        """
        captured = [a.annual_gross_carbon_sequestration for a in model.schedule.agents]
        return np.std(captured)

    @staticmethod
    def compute_current_carbon_release(model):
        """The function that accumulates yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        Note:
            None
        Todo:
            None
        """
        # roll current bins stored as np.array
        model.release_bins["slow"] = np.roll(model.release_bins["slow"], 1)
        model.release_bins["fast"] = np.roll(model.release_bins["fast"], 1)

        # aggregate required type of release (mulched, etc)
        mulched = Urban.aggregate(model, "mulched")
        standing = Urban.aggregate(model, "decomposing_trunk")
        root = Urban.aggregate(model, "decomposing_root")
        immediate = Urban.aggregate(model, "immediate_release")

        # add new potential releases into release bins
        model.release_bins["slow"][0] = standing + root
        model.release_bins["fast"][0] = mulched

        # compute and aggregate release
        def update_release(type):
            carbon_release = 0
            for i in range(len(model.release_bins[type])):
                mass = model.release_bins[type][i]
                rate = Urban.compute_carbon_release_rate(i + 1, type)
                released = rate * mass
                carbon_release += released
                model.release_bins[type][i] -= released
            return carbon_release

        carbon_release_slow = update_release("slow")
        carbon_release_fast = update_release("fast")

        return carbon_release_fast + carbon_release_slow + immediate

    @staticmethod
    def compute_carbon_release_rate(year, state="fast"):
        """Determines a carbon release rate.

        Args:

            year: (:obj:`int`) or (:obj:`float`): time passed since the biomass is dead.
            state: (:obj:`str`): Represents exposure type can be one of ('standing', 'mulched', 'root', 'fast', 'slow',)

        Returns:
            (:obj:`float`): decay rate.

        Note:

            The function computes the release rate based on the time spent since
            the total carbon within biomass of a dead tree or its parts. The shape of function
            assures a unit integral as time goes to infinity.

            When k = 2, in the first 3 years around 60% is released, which corresponds to
            empirical findings for mulched biomass above ground.

            When k = 5, in the first 3 years around 40% is released, which corresponds to
            empirical reports for carbon release for standing dead trees as well as underground roots.

            check:
                import scipy.integrate as integrate
                integrate.quad(f, 0, 20) > 0.96 for k in (2:5).
        """
        k = 2 if state in ("mulched", "fast") else 5
        return 1 / k * np.exp(-1 * year / k)

    @staticmethod
    def format_impact_analysis(model_vars: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the output of the simulation
        """
        # Process and clean data after the simulation
        # IMPACT ANALYSIS
        model_vars["Cum_Seq"] = model_vars.Seq.cumsum()

        # Processing to avoid out of range float values
        inf_count = float(np.isinf(model_vars).values.sum())
        if inf_count > 0:
            logging.debug("Cleaning {} INF values...".format(str(inf_count)))
            model_vars.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Replace Nan values with 0. This can happen when the number of alive trees is 0 and the above divisions in 'IMPACT ANALYSIS' are performed.
        has_na_values = model_vars.isnull().values.any()
        if has_na_values:
            logging.debug("Removing N/A values...")
            model_vars.fillna(0, inplace=True)

        return model_vars

class Miyawaki(Model):
    """A generic miyawaki model. To be tailored according to specific sites."""

    # Used to hold the scaling of actual physical space within the digital space.
    # It shows the size of each cell (square) in meters.
    # FIXME: this needs to be brought up as a parameter and placed within the groups
    #      of parameters that handle physical to digital twin mapping.
    # dt_resolution = 2  # in meters

    def __init__(
        self,
        population: Union[str, pd.DataFrame],
        species_allometrics: Union[str, Species],
        site_config: Union[str, SiteConfig],
        scenario: Union[str, ScenarioConfig],
    ):
        """The constructor method.

        Args:
            population: (:obj:`pd.DataFrame`): A dataframe tree properties are read from a site.
            species_composition (:obj:`str`): The name of the file that keeps allometrics of the tree species for the site.
            site_config: (:obj:`string`): name of the json file.
            scenario: (:obj:`dict`): Python dictionary that holds experiment parameters.
        Returns:
            None

        Note:
            First release model.

        Todo:
            Check for hard coded constants and parameterize further.
        """

        ############################################################
        # INPUTS
        ############################################################

        super().__init__()
        self.df = self._handle_population_input(population)
        self.allometrics = Species(species_allometrics) if isinstance(species_allometrics, str) else species_allometrics
        self._handle_site_configuration(site_config, len(self.df))
        self._handle_scenario_configuration(scenario)

        ############################################################
        ############################################################

        # If the dataset has no xpos and ypos columns, use the utility method to generate them
        if "xpos" not in self.df.columns or "ypos" not in self.df.columns:
            latlng_array_to_xy(self.df)

        # Setting MESA specific parameters
        width = int(max(self.df.xpos)) + 1
        length = int(max(self.df.ypos)) + 1

        logging.debug("Building grid with size: {} x {} ...".format(width, length))
        self.grid = MultiGrid(width, length, torus=False)
        # to be parameterized and set during initialization.

        # Load species composition and their allometrics
        required_cols = ["id", "dbh", "height", "species", "condition", "xpos", "ypos"]

        # Test that the df is complete or raise keyerror
        for attribute in required_cols:
            self.df[attribute]

        # list any extra columns that are present in the input data
        extra_columns = list(set(self.df.columns) - set(required_cols))

        self.num_agents = len(self.df)
        self.schedule = RandomActivation(self)

        self.sapling_dbh = min(self.df.dbh)
        # Each entry index i, represents number of years since the biomass is decay period.
        self.release_bins = {
            "slow": np.zeros(10),  # for dead root and standing tree.
            "fast": np.zeros(10),  # for mulched biomass
        }

        # Create agents.
        count = 0
        for index, row in self.df.iterrows():
            a = Tree(
                row.id,
                self,
                dbh=row.dbh,
                species=row.species,
                condition=str(row.condition).lower(),
                xpos=row.xpos,
                ypos=row.ypos,
                **{col: row[col] for col in extra_columns}
            )
            self.schedule.add(a)
            count += 1

            # Place trees on the plot based on actual physical positioning
            x = row.xpos
            y = row.ypos
            logging.debug("Placing agent {} at ({},{})".format(index, x, y))
            self.grid.place_agent(a, (x, y))
        logging.info(f"Placed {count} tree agents")

        # This variable below works as an indexer while adding new trees to the population during the run time.
        self.current_id = max(self.df.id)

        ALIVE_STATE = ["excellent", "good", "fair", "poor", "critical", "dying"]
        # Collecting model and agent level data
        self.datacollector = DataCollector(
            model_reporters={
                "Storage": lambda m: self.aggregate(m, "carbon_storage"),
                "Seq": lambda m: self.aggregate(m, "annual_gross_carbon_sequestration"),
                # Avg sequestered carbon per tree is annual carbon divided by number of living trees
                "Avg_Seq": (lambda m: self.aggregate(
                    m, "annual_gross_carbon_sequestration"
                )
                / self.count(
                        m,
                        "condition",
                        lambda x: x in ALIVE_STATE,
                    )
                ),
                "Released": self.compute_current_carbon_release,
                # Avg released carbon per year is annual carbon release divided by number of living trees
                "Avg_Rel": lambda m: self.compute_current_carbon_release(m)
                    / self.count(
                        m,
                        "condition",
                        lambda x: x in ALIVE_STATE,
                    ),
                "Alive": lambda m: self.count(
                    m, "condition", lambda x: x in ALIVE_STATE
                ),
                "Dead": lambda m: self.count(m, "condition", lambda x: x == "dead"),
                "Critical": lambda m: self.count(
                    m, "condition", lambda x: x == "critical"
                ),
                "Dying": lambda m: self.count(m, "condition", lambda x: x == "dying"),
                "Poor": lambda m: self.count(m, "condition", lambda x: x == "poor"),
                "Replaced": lambda m: self.count(
                    m, "condition", lambda x: x == "replaced"
                ),
                "Seq_std": self.agg_std_sequestration,
            },
            agent_reporters={
                "species": "species",
                "dbh": "dbh",
                "height": "tree_height",
                "crownH": "crown_height",
                "crownW": "crown_width",
                "canopy_overlap": "overlap_ratio",
                "cle": "cle",
                "condition": "condition",
                "dieback": "dieback",
                "biomass": "biomass",
                "seq": "annual_gross_carbon_sequestration",
                "carbon": "carbon_storage",
                "deroot": "decomposing_root",
                "detrunk": "decomposing_trunk",
                "mulched": "mulched",
                "burnt": "immediate_release",
                "xpos": "xpos",
                "ypos": "ypos",
                "est_age": "age",
                "lat": "lat",
                "lng": "lng",
                **{col: col for col in extra_columns},
            },
        )

        logging.info(
            "Initialisation of the Digital Twins of {} trees on a {} by {} digital space is complete!".format(
                self.num_agents, width, length
            )
        )

    def run(self, steps=None):
        """Customized MESA method that sets the major components of scenario analyses process."""
        pop = str(self.df.shape[0])
        if not steps:
            steps = self.time_horizon
        logging.debug("Running simulation ({} years)...".format(steps))
        start = time.time()
        for _ in range(steps):
            logging.info("Year:{}".format(self.schedule.time + 1))
            self.step()
        end = time.time()
        logging.debug(
            "{} steps completed (pop. {}): {}".format(steps, pop, end - start)
        )
        logging.info("Simulation is complete!")

        return self.impact_analysis()

    def step(self):
        """Customized MESA method that sets the major components of scenario analyses process."""
        logging.info("Year:{}".format(self.schedule.time + 1))
        self.get_weather_projection()

        logging.info("Agents are working ...")
        self.schedule.step()

        logging.info("Yearly data is being collected ...")
        self.datacollector.collect(self)

        # print('Step:{} ({}s)'.format(self.schedule.time, end-start))
        # print(self.release_bins['slow'])
        # print(self.release_bins['fast'])

    def impact_analysis(self) -> pd.DataFrame:
        """
        Provides impact analysis of the simulation
        """
        df_out_site = self.datacollector.get_model_vars_dataframe()
        return Urban.format_impact_analysis(df_out_site)

    def get_agent_data(self) -> pd.DataFrame:
        """
        Provides agent data of the simulation
        """
        df_out_agents = self.datacollector.get_agent_vars_dataframe()
        return df_out_agents
    
    def _handle_population_input(
            self, population: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load population dataframe if necessary"""
        if isinstance(population, str):
            population = pd.read_csv(population)

        return population

    def _handle_site_configuration(
        self, site_config: Union[str, SiteConfig], population_size: int
    ):
        """Loads site configuration information."""
        if isinstance(site_config, str):
            site_config = SiteConfig.from_file(site_config)

        self.growth_season_mean = site_config.weather.growth_season_mean
        self.growth_season_var = site_config.weather.growth_season_var
        self.project_site_type = site_config.project_site_type
        self.dt_resolution = round(
            np.sqrt(1 / (population_size / site_config.total_m2)),
            2,  # round to < decimal places
        )
        self.light_loss_multiplier = site_config.light_loss_multiplier

    def _handle_scenario_configuration(self, scenario: Union[str, ScenarioConfig]):
        """Loads site configuration information."""
        if isinstance(scenario, str):
            scenario = ScenarioConfig.from_file(scenario)

        self.maintenance_scope = scenario.maintenance_scope
        self.time_horizon = scenario.time_horizon_years

    def get_weather_projection(self):
        """The method retrieves wetaher projection for the current iteration,
        eg. year, at the moment it uses a simulated projections on the go,
        instead previously computed estimates or forecasts can be used as well.

        Args:
            None

        Returns:
            None
        Note:
            The avg season length is based on iTree Glasgow estimates (see iTree 2015 report)
            These variables need to be parameterrized and read-in through site (model) specific
            initialization module.
        Todo:
            * Implement a site initialization method that sets the initial parameters
            and tree specific variables.
        """
        # The avg season length is based on iTree Glasgow estimates (see iTree 2015 report)
        # These variables need to be parameterrized and read-in through site (model) specific
        # initialization module.
        # season_mean = 200

        self.WeatherAPI = WeatherSim(
            season_length=self.growth_season_mean, season_var=self.growth_season_var
        )

    @staticmethod
    def aggregate(model, var, func=lambda x, y: x + y, init=0):
        """A higher order function that aggregates a given variable.

        The aggregation function and the initial conditions can be specified.
        """
        agent_values = [eval("a.{}".format(var)) for a in model.schedule.agents]
        
        # Filter out nan values from the list
        filtered_values = [value for value in agent_values if not np.isnan(value)]
        
        # Use reduce on the filtered list
        return reduce(func, filtered_values, init)

    @staticmethod
    def count(model, memory, predicate):
        """A higher order function for counting agents that satisfy a specific attribute."""
        return len(
            list(
                filter(
                    predicate,
                    [eval("a.{}".format(memory)) for a in model.schedule.agents],
                )
            )
        )

    @staticmethod
    def aggregate_sequestration(model):
        """The function that accumulates yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        Note:
            None
        Todo:
            None
        """
        captured = [a.annual_gross_carbon_sequestration for a in model.schedule.agents]
        return sum(captured)

    @staticmethod
    def agg_std_sequestration(model):
        """The function estimates the standard deviation for yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        """
        captured = [a.annual_gross_carbon_sequestration for a in model.schedule.agents]
        return np.std(captured)

    @staticmethod
    def compute_current_carbon_release(model):
        """The function that accumulates yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        Note:
            None
        Todo:
            None
        """
        # roll current bins stored as np.array
        model.release_bins["slow"] = np.roll(model.release_bins["slow"], 1)
        model.release_bins["fast"] = np.roll(model.release_bins["fast"], 1)

        # aggregate required type of release (mulched, etc)
        mulched = Urban.aggregate(model, "mulched")
        standing = Urban.aggregate(model, "decomposing_trunk")
        root = Urban.aggregate(model, "decomposing_root")
        immediate = Urban.aggregate(model, "immediate_release")

        # add new potential releases into release bins
        model.release_bins["slow"][0] = standing + root
        model.release_bins["fast"][0] = mulched

        # compute and aggregate release
        def update_release(type):
            carbon_release = 0
            for i in range(len(model.release_bins[type])):
                mass = model.release_bins[type][i]
                rate = Urban.compute_carbon_release_rate(i + 1, type)
                released = rate * mass
                carbon_release += released
                model.release_bins[type][i] -= released
            return carbon_release

        carbon_release_slow = update_release("slow")
        carbon_release_fast = update_release("fast")

        return carbon_release_fast + carbon_release_slow + immediate

    @staticmethod
    def compute_carbon_release_rate(year, state="fast"):
        """Determines a carbon release rate.

        Args:

            year: (:obj:`int`) or (:obj:`float`): time passed since the biomass is dead.
            state: (:obj:`str`): Represents exposure type can be one of ('standing', 'mulched', 'root', 'fast', 'slow',)

        Returns:
            (:obj:`float`): decay rate.

        Note:

            The function computes the release rate based on the time spent since
            the total carbon within biomass of a dead tree or its parts. The shape of function
            assures a unit integral as time goes to infinity.

            When k = 2, in the first 3 years around 60% is released, which corresponds to
            empirical findings for mulched biomass above ground.

            When k = 5, in the first 3 years around 40% is released, which corresponds to
            empirical reports for carbon release for standing dead trees as well as underground roots.

            check:
                import scipy.integrate as integrate
                integrate.quad(f, 0, 20) > 0.96 for k in (2:5).
        """
        k = 2 if state in ("mulched", "fast") else 5
        return 1 / k * np.exp(-1 * year / k)

    @staticmethod
    def format_impact_analysis(model_vars: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the output of the simulation
        """
        # Process and clean data after the simulation
        # IMPACT ANALYSIS
        model_vars["Cum_Seq"] = model_vars.Seq.cumsum()

        # Processing to avoid out of range float values
        inf_count = float(np.isinf(model_vars).values.sum())
        if inf_count > 0:
            logging.debug("Cleaning {} INF values...".format(str(inf_count)))
            model_vars.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Replace Nan values with 0. This can happen when the number of alive trees is 0 and the above divisions in 'IMPACT ANALYSIS' are performed.
        has_na_values = model_vars.isnull().values.any()
        if has_na_values:
            logging.debug("Removing N/A values...")
            model_vars.fillna(0, inplace=True)

        return model_vars
