""" The module holds the main objects that manage and handle simulation runtime and data collection. """

# Importing Python Libraries
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Union, List

from joblib import Parallel, delayed

# Importing necessary Mesa packages
from mesa import Model

# Importing needed GUS objects
from .urban import Urban
from .config import ScenarioConfig, SiteConfig
from sklearn.cluster import DBSCAN
from ..allometrics import Species
from ..utilities import calculate_dataframe_area

ALIVE_STATE = ["EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL", "DYING"]


def _min_cluster_size(clusters):
    return min([len(cluster) for cluster in clusters])
CPUs = min(4, os.cpu_count())
# logging.basicConfig(level=logging.DEBUG)
class ClusteredUrban(Model):
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
    ):
        start = time.time()
        # build an urban model for each cluster
        # parallelise the whole simulation for each one, and aggregation after?
        self.df = self._handle_population_input(population)
        self.site_config = self._handle_site_configuration(site_config)
        self.scenario = self._handle_scenario_configuration(scenario)

        clusters = self._cluster_coordinates()
        print(
            f"Found {len(clusters)} clusters with min. size: {_min_cluster_size(clusters)} trees"
        )

        if species_allometrics is None:
            # use defaults
            species_allometrics = Species()

        # We require global grid coordinates ...
        width = int(max(self.df.xpos)) + 1
        length = int(max(self.df.ypos)) + 1

        # For each one of the clusters, build a model
        self.models: List[Urban] = Parallel(n_jobs=CPUs)(
            delayed(self._get_urban)(
                cluster, self.scenario, self.site_config, species_allometrics, SHOULD_USE_NEW_ALLOMETRICS, (width, length)
            ) for cluster in clusters
        )
        end = time.time()
        print(
            "Model creation complete in {} seconds.".format(round(end - start, 2))
        )

    def _get_urban(self, cluster_pop, scenario, site_config, species_allometrics, SHOULD_USE_NEW_ALLOMETRICS, grid_size):
        return Urban(
            cluster_pop,
            scenario,
            site_config,
            species_allometrics,
            SHOULD_USE_NEW_ALLOMETRICS,
            grid_size
        )

    def _cluster_coordinates(self, gap_threshold=100.0):
        """
        Clusters scattered points based on a distance threshold using DBSCAN.

        Parameters
        ----------
        df : pandas.DataFrame (with xpos, ypos)
        gap_threshold : float
            The distance threshold below which points are considered neighbours.
        """
        start = time.time()
        coords = self.df[["xpos", "ypos"]].to_numpy()

        MIN_UNCLUSTERED_XY_RANGE = 1000 # < 1,000,000 m^2 (1 km^2)
        if (
            max(self.df["xpos"]) - min(self.df["xpos"]) < MIN_UNCLUSTERED_XY_RANGE
            and max(self.df["ypos"]) - min(self.df["ypos"]) < MIN_UNCLUSTERED_XY_RANGE
        ):
            if len(self.df) < 2000:
                # breakout for small populations
                return [self.df]
            gap_threshold = 10.0
        else:
            gap_threshold = 100.0
        
        runs = 0
        clusters = ["one"]
        # Arbitrarily, there should be more than one cluster and fewer clusters than 1/5 of the total population
        # i.e. at least 5 trees per cluster on average
        while not len(clusters) > 1:
            db = DBSCAN(eps=gap_threshold, min_samples=1)
            labels = db.fit_predict(coords)

            # Create clusters
            clusters = []
            for label in np.unique(labels):
                cluster_df = self.df[labels == label].copy()
                cluster_df.reset_index(drop=True, inplace=True)
                clusters.append(cluster_df)
            
            if len(clusters) > len(self.df)/5:
                gap_threshold += gap_threshold*(1/10)
            else:
                gap_threshold -= gap_threshold*(1/10)

            runs += 1

            # If the gap threshold is too small, break
            if gap_threshold <= 1.5:
                break        

        print(f"Threshold for cluster gaps: {gap_threshold}m")
        end = time.time()
        print(
            "Clustering complete in {} seconds (with {} clustering attempts).".format(round(end - start, 2), runs)
        )
        return clusters

    def run(self, steps=None, with_agents=False):
        """Customized MESA method that sets the major components of scenario analyses process."""
        print("Running clustered simulation...")
        # Run the models in parallel
        start = time.time()
        # Aggregate the results

        IMPACTS = pd.DataFrame()
        if not with_agents:
            for result in Parallel(n_jobs=CPUs)(delayed(model.run)() for model in self.models):
                assert isinstance(result[0], pd.DataFrame)
                IMPACTS += result[0]

            end = time.time()
            logging.debug(
                "Simulation is complete in {} seconds.".format(round(end - start, 2))
            )
            self.impacts = IMPACTS
            return self.impacts, None
        
        # Initialize offest for new AgentIDs
        global_agent_id = 0
        AGENTS = pd.DataFrame()
        # with_agents = Parallel(n_jobs=CPUs)(delayed(self._run_with_agents)(model) for model in self.models)
        for result in Parallel(n_jobs=CPUs)(delayed(self._run_with_agents)(model) for model in self.models):

            # list of tuples (impact, analysis)
            IMPACTS += result[0]

            misnumbered_agents = result[1]            
            misnumbered_agents['AgentID'].apply(lambda x: x + global_agent_id)
            # Update the global ID offset for the next dataframe
            global_agent_id += len(misnumbered_agents)

            # Append the updated dataframe to the list
            AGENTS = AGENTS.append(misnumbered_agents, ignore_index=True)
            
            # Concatenate all dataframes with updated IDs

        end = time.time()
        logging.debug(
            "Simulation is complete in {} seconds.".format(round(end - start, 2))
        )
        self.impacts = IMPACTS
        self.agents = AGENTS

        return self.impacts, self.agents
    
    def _run_with_agents(self, model, steps=None):
        """Run the model with agents."""
        impacts = model.run(steps)
        return impacts, model.get_agent_data().reset_index(drop=False)

    def impact_analysis(self):
        if hasattr(self, "impacts"):
            return self.impacts
        
        raise ValueError("No impacts have been calculated yet. Please run the simulation first.")   

    def get_agent_data(self) -> pd.DataFrame:
        if hasattr(self, "agents"):
            return self.agents
        
        raise ValueError("No agent data has been collected yet. Please run the simulation `with_agents=True` first.")

    def _handle_population_input(
        self, population: Union[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load population dataframe if necessary"""
        if isinstance(population, str):
            population = pd.read_csv(population)

        if isinstance(population, pd.DataFrame):
            return population

        raise ValueError("Population input must be a pandas DataFrame or a file path.")

    def _handle_site_configuration(
        self, site_config: Union[str, SiteConfig]
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

        self.site_config = site_config

    def _handle_scenario_configuration(self, scenario: Union[str, ScenarioConfig]):
        """Loads site configuration information."""
        if scenario is None:
            scenario = ScenarioConfig()

        if isinstance(scenario, str):
            scenario = ScenarioConfig.from_file(scenario)

        self.scenario = scenario