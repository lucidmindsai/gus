"""The module holds general purpose python functions on data preperations."""
from functools import reduce
import utm
import json
import math
import logging
import pandas as pd
import numpy as np

from .models import Urban, SiteConfig, WeatherConfig


def get_raster_data(
    dtwin,
    var="unique_id",
    probe=lambda x, y: x + y,
    predicate=lambda x: x,
    init=0,
    counts=False,
):
    """A general purpose function that collects celular data on the grid.

    Args:
            dtwin: (:obj:`Urban`): an urban forest model.
            var: (:obj:`string`): a valid memory variable of a tree.

    Returns:
            (:obj:numpy.ndarray`): raster like data on a 2D grid.
        Note:
            None
        Todo:
            None
    """
    if counts:
        probe = lambda x, y: x + 1
    raster = np.zeros((dtwin.grid.width, dtwin.grid.height))
    for cell in dtwin.grid.coord_iter():
        cell_content, x, y = cell
        cell_content = filter(
            lambda x: x.condition not in ["dead", "replaced"], cell_content
        )
        filtered = filter(predicate, [eval("a.{}".format(var)) for a in cell_content])
        raster[x][y] = reduce(probe, filtered, init)
    return raster


def latlng_array_to_xy(population_df, lat_column="lat", lng_column="lng"):
    """A general purpose function that translates lat, lng data to x,y pos.

    Args:
            lat_array: (:obj:`numpy.ndarray`): array of latitudes.
            lng_array: (:obj:`numpy.ndarray`): array of longitudes.

    Returns:
            (:obj:numpy.ndarray`): array of x,y positions.
        Note:
            None
    """
    lat = population_df[lat_column].to_numpy()
    lng = population_df[lng_column].to_numpy()
    easting, northing, _, _ = utm.from_latlon(
        lat, lng
    )  # also returns zone and zone letter
    population_df["xpos"] = easting.astype(int)
    population_df["ypos"] = northing.astype(int)

    # Normalise the xpos, ypos values, and round to integers
    population_df["xpos"] = population_df["xpos"] - population_df["xpos"].min()
    population_df["ypos"] = population_df["ypos"] - population_df["ypos"].min()

    return population_df


def latlng_to_xy(row):
    """DEPRECATED: Much slower than latlng_array_to_xy, but works on a single row.

    A general purpose function that translates lat, lng data to x,y pos.

    Args:
            row: (pandas.DataFrame.row): a Pandas DataFrame row.

    Returns:
            row: (pandas.DataFrame.row): converted xpos and ypos added to the DataFrame row.
        Note:
            None
        Todo:
            None
    """
    coordinates = utm.from_latlon(row["lat"], row["lng"])
    row["xpos"], row["ypos"] = coordinates[0], coordinates[1]
    return row


def raster_grid(df, minx, miny, grid_width):
    """A general purpose function is to place the data on the grid with given sizes.

    Args:
            row: (pandas.DataFrame.row): a Pandas DataFrame row.
            minx: (:obj:`int`): minimum x pos value.
            miny: (:obj:`int`): minimum y pos value.
            grid_width: (:obj:`string`): width of the grid to be mapped at.
    Returns:
            row: (pandas.DataFrame.row): converted xpos and ypos added to the DataFrame row.
        Note:
            None
        Todo:
            None
    """
    df["gus_x"] = ((df["xpos"] - minx) // grid_width).astype(int)
    df["gus_y"] = ((df["ypos"] - miny) // grid_width).astype(int)
    return df


def load_site_config_file(config_file) -> SiteConfig:
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
    try:
        f = open(config_file)
    except IOError as e:
        print(str(e))
    params = json.loads(f.read())

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
        if params["project_site_type"] in Urban.site_types:
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


def calculate_dataframe_area(tree_df: pd.DataFrame):
    # If xpos and ypos columns exist, calculate area with them
    if "xpos" in tree_df.columns and "ypos" in tree_df.columns:
        # Get area
        min_x = tree_df["xpos"].min()
        max_x = tree_df["xpos"].max()
        min_y = tree_df["ypos"].min()
        max_y = tree_df["ypos"].max()
        return (max_x - min_x) * (max_y - min_y)
    # otherwise, check for lat and lon columns
    elif "lat" in tree_df.columns and "lng" in tree_df.columns:
        R = 6371000  # Radius of the Earth in meters
        min_lat = tree_df["lat"].min()
        max_lat = tree_df["lat"].max()
        min_lon = tree_df["lng"].min()
        max_lon = tree_df["lng"].max()

        # Convert degrees to radians
        lat1_rad = math.radians(min_lat)
        lat2_rad = math.radians(max_lat)
        lon1_rad = math.radians(min_lon)
        lon2_rad = math.radians(max_lon)

        # Calculate average latitude
        lat_avg = (lat1_rad + lat2_rad) / 2

        # Calculate the differences in latitude and longitude
        d_lat = abs(math.sin(lat2_rad) - math.sin(lat1_rad))
        d_lon = abs(lon2_rad - lon1_rad)

        # Calculate the area
        return d_lat * d_lon * R * R * math.cos(lat_avg)

    return None
