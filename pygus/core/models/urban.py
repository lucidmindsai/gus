""" The module holds the main objects that manage and handle simulation runtime and data collection. """

# Importing Python Libraries
import time
import numpy as np
import pandas as pd
from typing import Union
from functools import reduce

# Importing necessary Mesa packages
from mesa import Model
from mesa.datacollection import DataCollector

# Importing needed GUS objects
from ..agents import Tree
from ..weather import WeatherSim
from ..allometrics import Species
from .config import ScenarioConfig, SiteConfig
from ..utilities import latlng_array_to_xy, calculate_dataframe_area

from .sparse_multigrid import SparseMultiGrid

import logging
logger = logging.getLogger(__name__)

ALIVE_STATE = ["EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL", "DYING"]

# TODO: Mesa batch-runner instead of our current version. Has built in support for parameter sweepign
#            https://mesa.readthedocs.io/stable/tutorials/intro_tutorial.html#batch-run


# logging.basicConfig(level=logging.DEBUG)
class Urban(Model):
    """An urban green space model. To be tailored according to specific sites."""

    # Contagion multiplier for the contagion level parameters received from the scenario builder inputs.
    contagion_multiplier_low_rate = 0.5
    contagion_multiplier_high_rate = 2

    # Disease risks component that is not spread through neighbouring trees.
    # TODO: To be made and instance attribute and  set via site configurations
    noncontagion_disease_rate = 0.05
   
    def __init__(
        self,
        population: Union[str, pd.DataFrame],
        scenario: Union[str, ScenarioConfig] = None,
        site_config: Union[str, SiteConfig] = None,
        species_allometrics: Union[str, Species] = None,
        SHOULD_USE_NEW_ALLOMETRICS: bool = True,
        grid_shape: tuple = None,
        seed=None,
    ):
        """The constructor method.

        Args:
            population: (:obj:`pd.DataFrame`): A dataframe tree properties are read from a site.
            species_composition (:obj:`str`): The name of the file that keeps allometrics of the tree species for the site.
            site_config: (:obj:`string`): name of the json file.
            scenario: (:obj:`dict`): Python dictionary that holds experiment parameters.

        Todo:
            Check for hard coded constants and parameterize further.
        """

        super().__init__(seed=seed)
        self.df = self._handle_population_input(population)
        self.USING_NEW_ALLOMETRICS = SHOULD_USE_NEW_ALLOMETRICS
        self._handle_allometrics(species_allometrics)
        self._handle_site_configuration(site_config, len(self.df))
        self._handle_scenario_configuration(scenario)

        ############################################################
        ############################################################

        # If the dataset has no xpos and ypos columns, use the utility method to generate them
        if "xpos" not in self.df.columns or "ypos" not in self.df.columns:
            latlng_array_to_xy(self.df)

        # Setting MESA specific parameters
        if grid_shape is None:
            width = int(max(self.df.xpos)) + 1
            length = int(max(self.df.ypos)) + 1
        else:
            width, length = grid_shape

        logger.debug(
            "Building grid with size: {} x {} ({}) ...".format(
                width, length, width * length
            )
        )
        s = time.time()
        self.grid = SparseMultiGrid(width, length, torus=False)
        e = time.time()
        logger.debug("Grid built in {}s".format(e - s))

        # Load species composition and their allometrics
        required_cols = ["id", "dbh", "height", "species", "condition", "xpos", "ypos"]

        # Test that the df is complete or raise keyerror
        for attribute in required_cols:
            self.df[attribute]

        # list any extra columns that are present in the input data
        extra_columns = list(set(self.df.columns) - set(required_cols))

        self.num_agents = len(self.df)

        self.sapling_dbh = min(self.df.dbh)
        # Each entry index i, represents number of years since the biomass is decay period.
        self.release_bins = {
            "slow": np.zeros(10),  # for dead root and standing tree.
            "fast": np.zeros(10),  # for mulched biomass
        }

        # Function to check if a tree is alive
        def is_alive(condition):
            return condition.upper() in ALIVE_STATE

        # Create agents.
        count = 0
        logger.debug(f"Placing {len(self.df)} tree agents")
        logger.debug(f"{self.df['condition'].value_counts()}")
        for index, row in self.df.iterrows():
            a = Tree(
                self,
                dbh=row.dbh,
                species=row.species,
                # height (calculated from dbh)
                # kind
                condition=str(row.condition).lower(),
                xpos=row.xpos,
                ypos=row.ypos,
                **{col: row[col] for col in extra_columns},
            )
            count += 1

            # Place trees on the plot based on actual physical positioning
            x = row.xpos
            y = row.ypos
            logging.debug("Placing agent {} at ({},{})".format(index, x, y))
            self.grid.place_agent(a, (x, y))
            
        # This variable below works as an indexer while adding new trees to the population during the run time.
        self.current_id = max(self.df.id)

        logger.debug("########## GRID #################################")
        logger.debug(f"Grid takes up: {self.grid._grid.__sizeof__()/ 1024 / 1024} MB")

        logger.debug(f"Placed {count} tree agents")
        logger.debug(self.df["condition"].describe())
        logger.debug("Alive trees: ", len(self.df[self.df["condition"].apply(is_alive)]))
        logger.debug("#################################################")

        # Collecting model and agent level data
        self.datacollector = DataCollector(
            model_reporters={
                "AllTrees": all_trees,
                "Storage": storage,
                "Seq": seq,
                "Avg_Seq": avg_seq,
                "Released": self.compute_current_carbon_release,
                "Avg_Rel": avg_rel,
                "Alive": alive_count,
                "Dead": dead_count,
                "Critical": critical_count,
                "Dying": dying_count,
                "Poor": poor_count,
                "Replaced": replaced_count,
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

        logging.debug(
            "Initialisation of the Digital Twin is complete. {} trees have been placed on a space of {} by {} ({}) cells!".format(
                self.num_agents, width, length, width * length
            )
        )

    def run_model(self):
        """Customized MESA method that sets the major components of scenario analyses process."""
        pop = str(self.df.shape[0])
        if not self.steps:
            steps = self.time_horizon
        logging.debug("Running simulation ({} years)...".format(steps))
        start = time.time()
        for _ in range(steps):
            logging.debug("Year:{}".format(self.steps + 1))
            self.step()
        end = time.time()
        logging.debug(
            "{} steps completed (pop. {}): {}".format(steps, pop, end - start)
        )
        logging.debug("Simulation is complete!")

        return self.impact_analysis()

    # Custom methods to support pickling
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes
        state = self.__dict__.copy()
        # Remove unpickleable attributes if any
        # e.g., state.pop('unpickleable_attr', None)
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)

    def step(self):
        """Customized MESA method that sets the major components of scenario analyses process."""
        logging.debug("Year:{}".format(self.steps + 1))
        self.get_weather_projection()

        logging.debug("Agents are working ...")
        self.agents.shuffle_do("step")

        logging.debug("Yearly data is being collected ...")
        self.datacollector.collect(self)
        # print survival rate
        alive = 0
        replaced = 0
        for a in self.agents:
            if a.condition.upper() in ALIVE_STATE:
                alive += 1
            if a.condition.upper() == "REPLACED":
                replaced += 1

        logger.debug(
            "Survival rate after {}yrs: {}%".format(
                self.steps + 1,
                (alive / (len(self.agents) - replaced)) * 100,
            )
        )

        # logger.debug(self.release_bins['slow'])
        # logger.debug(self.release_bins['fast'])

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

    def _handle_population_input(
        self, population: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load population dataframe if necessary"""
        if isinstance(population, str):
            population = pd.read_csv(population)

        if isinstance(population, pd.DataFrame):
            return population

        raise ValueError("Population input must be a pandas DataFrame or a file path.")

    def _handle_allometrics(self, species_allometrics: Union[str, Species]):
        """Loads allometric data for the species in the population."""
        if species_allometrics is None:
            species_allometrics = Species()

        if isinstance(species_allometrics, str):
            species_allometrics = Species(species_allometrics)

        self.allometrics = species_allometrics

    def _handle_site_configuration(
        self, site_config: Union[str, SiteConfig], population_size: int
    ):
        """Loads site configuration information."""
        if site_config is None:
            site_config = SiteConfig()

        if isinstance(site_config, str):
            site_config = SiteConfig.from_file(site_config)

        if site_config.total_m2 is None or site_config.total_m2 == 0:
            total_m2 = calculate_dataframe_area(self.df)
            site_config.total_m2 = total_m2
            site_config.impervious_m2 = 0
            site_config.pervious_m2 = total_m2

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
        if scenario is None:
            scenario = ScenarioConfig()

        if isinstance(scenario, str):
            scenario = ScenarioConfig.from_file(scenario)

        self.maintenance_scope = scenario.maintenance_scope
        self.time_horizon = scenario.time_horizon_years

        contagion_level = scenario.contagion_level
        if contagion_level == 'low':
            self.contagion_multiplier = self.contagion_multiplier_low_rate
        elif contagion_level == 'high':
            self.contagion_multiplier = self.contagion_multiplier_high_rate
        else:
            self.contagion_multiplier = 1.0

    @staticmethod
    def aggregate(model, var, func=lambda x, y: x + y, init=0):
        """A higher order function that aggregates a given variable.

        The aggregation function and the initial conditions can be specified.
        """
        agent_values = [eval("a.{}".format(var)) for a in model.agents]

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
                    [eval("a.{}".format(memory)) for a in model.agents],
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
        captured = [a.annual_gross_carbon_sequestration for a in model.agents]
        return sum(captured)

    @staticmethod
    def agg_std_sequestration(model):
        """The function estimates the standard deviation for yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        """
        captured = [a.annual_gross_carbon_sequestration for a in model.agents]
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


# Predicate functions
def always_true(x):
    return True

def is_alive(condition):
    return condition.upper() in ALIVE_STATE

def is_dead(condition):
    return condition.lower() == "DEAD"

def is_critical(condition):
    return condition.lower() == "CRITICAL"

def is_dying(condition):
    return condition.lower() == "DYING"

def is_poor(condition):
    return condition.lower() == "POOR"

def is_replaced(condition):
    return condition.lower() == "REPLACED"

# Model reporter functions
def all_trees(model):
    return Urban.count(model, "condition", always_true)  # Counting all trees

def storage(model):
    return Urban.aggregate(model, "carbon_storage")

def seq(model):
    return Urban.aggregate(model, "annual_gross_carbon_sequestration")

def avg_seq(model):
    total_seq = Urban.aggregate(model, "annual_gross_carbon_sequestration")
    alive_count = Urban.count(model, "condition", is_alive)
    return total_seq / alive_count if alive_count != 0 else 0

def avg_rel(model):
    total_rel = Urban.compute_current_carbon_release(model)
    alive_count = Urban.count(model, "condition", is_alive)
    return total_rel / alive_count if alive_count != 0 else 0

def alive_count(model):
    return Urban.count(model, "condition", is_alive)

def dead_count(model):
    return Urban.count(model, "condition", is_dead)

def critical_count(model):
    return Urban.count(model, "condition", is_critical)

def dying_count(model):
    return Urban.count(model, "condition", is_dying)

def poor_count(model):
    return Urban.count(model, "condition", is_poor)

def replaced_count(model):
    return Urban.count(model, "condition", is_replaced)