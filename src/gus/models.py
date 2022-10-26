#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing Python Libraries
import site
import numpy as np
from functools import reduce
import json
import logging

# Importing necessary Mesa packages
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Importing needed GUS objects
from .agents import Tree
from .allometrics import Species
from .weather import WeatherSim

class Urban(Model):
    """A generic urban green space model. To be tailored according to specific sites.
    """
    # Used to hold the scaling of actual physical space within the digital space. 
    # It shows the size of each cell (square) in meters.
    # TODO: this needs to be brought up as a parameter and placed within the groups
    #      of parameters that handle physical to digital twin mapping.  
    dt_resolution = 2 #in meters

    site_types = ['park','street','forest','pocket']

    def __init__(self,
        population,
        species_composition,
        site_config,
        scenario,
        batch=False):
        """The constructor method.

        Args:
            population: (:obj:`pd.DataFrame`): A dataframe tree properties are read from a site.
            species_composition (:obj:`str`): The name of the file that keeps allometrics of the tree species for the site. 
            site_config: (:obj:`string`): name of the json file.
            scenario: (:obj:`dict`): Python dictionary that holds experiment parameters.
            batch: (:obj:`bool`): Mesa parameter to control single vs batch runs.
        Returns:
            None

        Note:
            First release model.

        Todo:
            Check for hard coded constants and parameterize further.  
        """
        super().__init__()
        # Setting MESA specific parameters
        width = int(max(population.xpos)) + 1
        height = int(max(population.ypos)) + 1
        self.grid = MultiGrid(width, height, torus=False)
                # to be parameterized and set during initialization.
        self.schedule = RandomActivation(self)
        
        self._load_site_parameters(site_config)
        self._load_experiment_parameters(scenario)

        # Load species composition and their allometrics
        self.species = Species(species_composition)  # will be used by agents.
    
        # Test that the df is complete or raise keyerror
        for attribute in ['dbh', 'species', 'condition', 'xpos', 'ypos']:
            population[attribute]

        # copy and import df
        self.df = population
        self.num_agents = len(population)
        self.sapling_dbh = min(population.dbh)
        # Each entry index i, represents number of years since the biomass is decay period.
        self.release_bins = {'slow': np.zeros(10),  # for dead root and standing tree.
                             'fast': np.zeros(10)  # for mulched biomass
                             }

        # Create agents.
        for index, row in self.df.iterrows():
            # Tree init
            a = Tree(row.id, self,
                     dbh=row.dbh,
                     species=row.species,
                     condition=row.condition)
            self.schedule.add(a)

            # Place trees on the plot sequentially
            # based on their id/index.
            # TODO: This snippet may need to be converted into a function as part of
            # initilisartion module and x,y points need to be part of input DB.
            #x = int(index % self.grid.width)
            #y = int(index / self.grid.height)

            # # Add the agent to a random grid cell
            #x = self.random.randrange(self.grid.width)
            #y = self.random.randrange(self.grid.height)

            # # Locate the trees based on actual physical positioning
            x = row.xpos
            y = row.ypos
            self.grid.place_agent(a, (x, y))

        # This variable below works as an indexer while adding new trees to the population during the run time.
        self.current_id = max(population.id)

        # Collecting model and agent level data
        self.datacollector = DataCollector(
            model_reporters={
                "Storage": lambda m: self.aggregate(
                    m,
                    'carbon_storage'),
                "Seq": lambda m: self.aggregate(
                    m,
                    'annual_gross_carbon_sequestration'),
                # "Sequestrated": self.aggregate_sequestration,
                "Released": self.compute_current_carbon_release,
                "Alive": lambda m: self.count(
                    m,
                    'condition',
                    lambda x: x in ['excellent', 'good', 'fair', 'poor', 'critical', 'dying']),
                "Dead": lambda m: self.count(
                    m,
                    'condition',
                    lambda x: x == 'dead'),
                "Critical": lambda m: self.count(
                    m,
                    'condition',
                    lambda x: x == 'critical'),
                "Dying": lambda m: self.count(
                    m,
                    'condition',
                    lambda x: x == 'dying'),
                "Poor": lambda m: self.count(
                    m,
                    'condition',
                    lambda x: x == 'poor'),
                "Replaced": lambda m: self.count(
                    m,
                    'condition',
                    lambda x: x == 'replaced'),
                "Seq_std": self.agg_std_sequestration,
            },
            agent_reporters={
                "species": 'species',
                "dbh": 'dbh',
                "height": 'tree_height',
                "crownH": 'crown_height',
                "crownW": 'crown_width',
                "canopy_overlap": "overlap_ratio",
                "cle": "cle",
                "condition": 'condition',
                "dieback": 'dieback',
                "biomass": 'biomass',
                "seq": 'annual_gross_carbon_sequestration',
                "carbon": "carbon_storage",
                "deroot": 'decomposing_root',
                "detrunk": 'decomposing_trunk',
                "mulched": 'mulched',
                "burnt": 'immediate_release',
                "coordinates": 'pos',})
        logging.info("Initialisation of the Digital Twins of {} trees on a {} by {} digital space is complete!".format(self.num_agents,width,height))

    def step(self):
        """Customized MESA method that sets the major components of scenario analyses process.

        Args:
            None

        Returns:
            None

        Note:

        Todo:

        """
        logging.info("Year:{}".format(self.schedule.time + 1))
        self.get_weather_projection()

        logging.info("Agents are working ...")
        self.schedule.step()
        
        logging.info("Yearly data is being collected ...")
        self.datacollector.collect(self)
        # print('Step:{}'.format(self.schedule.time))
        # print(self.release_bins['slow'])
        # print(self.release_bins['fast'])

    def _load_experiment_parameters(self, experiment):
        """Loads site configuration information.

        Args:
            experiment: (:obj:`dict`): Python dictionary that holds experiment parameters.

        Returns:
            None
        Todo:

        """

        # Read denisty to set the digital twin resolution which is defined as
        # the cell size in terms of actual distance.
        if 'maintenance_scope' in experiment.keys():
            #maintenance_scope: (:obj:`int`): It can be 0:None,1:base, 2:cared)
            self.maintenance_scope = experiment['maintenance_scope']
        else:
            logging.warning("Maintenance scope is not given. A high maintenance site is assumed.")
            self.maintenance_scope = 2
        

    def _load_site_parameters(self, config_file):
        """Loads site configuration information.

        Args:
            config_file: (:obj:`string`): name of the json file.

        Returns:
            None
        Todo:

        """
        try:
            f = open(config_file)
        except IOError as e:
            print(str(e))
        params = json.loads(f.read())
        
        #read site type
        stype = 'park' #default type
        if 'project_site_type' in params.keys():
            if params['project_site_type'] in Urban.site_types:
                stype = params['project_site_type']
            else:
                logging.warning("Undefined site type recognized. Park type will be used.")
        else:
            logging.warning("Site type is not provided. Park type will be used.")   
        self.site_type = stype
        
        # Read in growth season mean and variance to be used by weather forecasting module.
        try:
            self.season_mean = params['weather']['growth_season_mean']
            self.season_var = params['weather']['growth_season_var']
        except KeyError:
            self.season_mean = 153
            self.season_var = 7
            logging.warning("Tree growth season mean and variance is not provided as expected. Global average is used.")

        # Read denisty to set the digital twin resolution which is defined as
        # the cell size in terms of actual distance.
        if 'area_tree_density_per_hectare' in params.keys():
            self.dt_resolution = np.sqrt(10000 / params['area_tree_density_per_hectare'])
            # The distance between the center of two tree trunks in meters. Even spatial distribution is assumed.
        else:
            logging.warning("area_tree_density_per_hectare is not given the default {} meters is used the distance from the clossest tree trunks.".format(self.dt_resolution)) 
    

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
            season_length=self.season_mean, season_var=self.season_var)

    @staticmethod
    def aggregate(model, var, func=lambda x, y: x + y, init=0):
        """A higher order function that aggregates a given variable.

        The aggregation function and the initial conditions can be specified.
        """
        return reduce(func, [eval('a.{}'.format(var))
                      for a in model.schedule.agents], init)

    @staticmethod
    def count(model, memory, predicate):
        """A higher order function for counting agents that satisfy a specific attribute."""
        return len(
            list(
                filter(
                    predicate, [
                        eval(
                            'a.{}'.format(memory)) for a in model.schedule.agents])))

    @staticmethod
    def aggregate_sequestration(model):
        """The function that accumulates yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        Note:
        Todo:
        """
        captured = [
            a.annual_gross_carbon_sequestration for a in model.schedule.agents]
        return sum(captured)

    @staticmethod
    def agg_std_sequestration(model):
        """The function estimates the standard deviation for yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        """
        captured = [
            a.annual_gross_carbon_sequestration for a in model.schedule.agents]
        return np.std(captured)

    @staticmethod
    def compute_current_carbon_release(model):
        """The function that accumulates yearly sequestration data from all trees within the site.

        Args:
            model: (:obj:`Urban`): an urban forest model.


        Returns:
            (:obj:float`): total sequestration in Kg.
        Note:
        Todo:
        """
        # roll current bins stored as np.array
        model.release_bins['slow'] = np.roll(model.release_bins['slow'], 1)
        model.release_bins['fast'] = np.roll(model.release_bins['fast'], 1)

        # aggregate required type of release (mulched, etc)
        mulched = Urban.aggregate(model, 'mulched')
        standing = Urban.aggregate(model, 'decomposing_trunk')
        root = Urban.aggregate(model, 'decomposing_root')
        immediate = Urban.aggregate(model, 'immediate_release')

        # add new potential releases into release bins
        model.release_bins['slow'][0] = standing + root
        model.release_bins['fast'][0] = mulched

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

        carbon_release_slow = update_release('slow')
        carbon_release_fast = update_release('fast')

        return carbon_release_fast + carbon_release_slow + immediate

    @staticmethod
    def compute_carbon_release_rate(year, state='fast'):
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
        k = 2 if state in ('mulched', 'fast') else 5
        return 1 / k * np.exp(-1 * year / k)
