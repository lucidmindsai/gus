import json
import logging
from typing import Union, Dict

from ..enums import ProjectSiteType

class ScenarioConfig:
    def __init__(self, maintenance_scope: int = 2, is_crownsunexposure_fixed: int = 1, time_horizon_years: int = 10):
        self.maintenance_scope = maintenance_scope
        self.is_crownsunexposure_fixed = is_crownsunexposure_fixed
        self.time_horizon_years = time_horizon_years

    @staticmethod
    def from_file(config_file: str):
        """Loads scenario configuration information from a json file in the form:

        {
            "maintenance_scope": 2,
            "is_crownsunexposure_fixed": 1,
            "time_horizon_years": 10
        }

        Args:
            config_file: (:obj:`string`): name of the json file.
        """
        params = _read_file_to_dict(config_file)
        return ScenarioConfig(
            maintenance_scope=params.get("maintenance_scope", 2),
            is_crownsunexposure_fixed=params.get("is_crownsunexposure_fixed", 1),
            time_horizon_years=params.get("time_horizon_years", 10)
        )

class WeatherConfig:
    def __init__(self, growth_season_mean: int = 153, growth_season_var: int = 7):
        self.growth_season_mean = growth_season_mean
        self.growth_season_var = growth_season_var

class SiteConfig:
    """A class to hold site configuration parameters."""

    def __init__(
        self,
        total_m2: int,
        impervious_m2: int,
        pervious_m2: int,
        weather: Union[Dict, WeatherConfig],
        tree_density_per_ha: int = None,
        project_site_type: str = "park",
    ):
        self.total_m2 = total_m2
        self.impervious_m2 = impervious_m2
        self.pervious_m2 = pervious_m2
        self.tree_density_per_ha = tree_density_per_ha
        # if weather is a dict, create a weatherConfig, else use the object
        if isinstance(weather, dict):
            self.weather = WeatherConfig(**weather)
        else:
            self.weather = weather

        self.project_site_type = project_site_type 

    @staticmethod
    def from_file(config_file: str):
        """Loads site configuration information from a json file in the form:

        {
            "total_m2":1000,
            "impervious_m2":500,
            "pervious_m2": 500,
            "tree_density_per_ha": 400,
            "weather": {
                "growth_season_mean": 200,
                "growth_season_var": 7
                },
            "project_site_type":"park"
        }

        Args:
            config_file: (:obj:`string`): name of the json file.
        """
        params = _read_file_to_dict(config_file)

        # Read in growth season mean and variance to be used by weather forecasting module.
        try:
            season_mean = params["weather"]["growth_season_mean"]
            season_var = params["weather"]["growth_season_var"]
        except KeyError:
            logging.warning(
                "Tree growth season mean and variance is not provided as expected. Global average is used."
            )
        weather = WeatherConfig(season_mean, season_var)

        # read site type
        stype = "park"  # default type
        if "project_site_type" in params.keys():
            if params["project_site_type"] in ProjectSiteType:
                stype = params["project_site_type"]
            else:
                logging.warning("Undefined site type recognized. Park type will be used.")
        else:
            logging.warning("Site type is not provided. Park type will be used.")

        return SiteConfig(
            total_m2=params.get("total_m2", 1000),
            impervious_m2=params.get("impervious_m2", 500),
            pervious_m2=params.get("pervious_m2", 500),
            tree_density_per_ha=params.get("tree_density_per_ha", 400),
            weather=weather,
            project_site_type=stype,
        )
    
def _read_file_to_dict(file):
    try:
        f = open(file)
    except IOError as e:
        logging.error(str(e))
    return json.loads(f.read())