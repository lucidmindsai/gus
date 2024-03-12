"""The module holds general purpose python functions on data preperations."""
import utm
import math
import logging
import numpy as np
import pandas as pd
from functools import reduce


def get_raster_data(
    dtwin, # Urban
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
        Note:()
            None
        Todo:
            None
    """
    logging.debug(dtwin.df.value_counts("condition"))

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
            population_df (:obj:`pandas.DataFrame`): DataFrame containing latitude and longitude columns.
            lat_column (str): Name of the latitude column.
            lng_column (str): Name of the longitude column.

    Returns:
            (:obj:`pandas.DataFrame`): DataFrame with added xpos and ypos columns representing x,y positions.
    """

    lat = population_df[lat_column].to_numpy()
    lng = population_df[lng_column].to_numpy()
    easting, northing, _, _ = utm.from_latlon(lat, lng)  # also returns zone and zone letter

    # Use .loc to ensure we modify the original dataframe
    population_df['xpos'] = easting.astype(int)
    population_df['ypos'] = northing.astype(int)

    # Normalise the xpos, ypos values, and round to integers
    population_df["xpos"] = population_df["xpos"] - population_df["xpos"].min()
    population_df["ypos"] = population_df["ypos"] - population_df["ypos"].min()


def calculate_dataframe_area(tree_df: pd.DataFrame):
    """Calculates the area in UTM grid units (or square metres) of a dataframe of trees.
    
    Args:
        tree_df (pd.DataFrame): The dataframe of trees to calculate the area of

    Returns:
        _type_: Returns the area of the dataframe in UTM grid units (or square metres)
    """
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

    # FIXME - this should be an exception
    return None

def filter_dataframe_to_bounding_box(df: pd.DataFrame, bbox, lat_col = "lat", lng_col = "lng") -> pd.DataFrame:
    """Filters a pandas dataframe with `lat` and `lng` columns to a bbox (list of floats), of the form:
    
    [min_lng, min_lat, max_lng, max_lat]

    Args:
        df (pd.DataFrame): The dataframe to filter
        bbox (list): A list of floats representing the bounding box
        lat_col (str, optional): The name for the latitude column. Defaults to "lat".
        lng_col (str, optional): The name for the longitude column. Defaults to "lng".

    Returns:
        pd.DataFrame: Returns the filtered dataframe
    """
    # This could be a type check, but it's not worth it
    assert len(bbox) == 4
    assert type(bbox) == list
    assert type(bbox[0]) == float
    assert type(bbox[1]) == float
    assert type(bbox[2]) == float
    assert type(bbox[3]) == float
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]
    
    # Filter the trees
    df = df[df[lng_col] > bbox[0]]
    df = df[df[lng_col] < bbox[2]]
    df = df[df[lat_col] > bbox[1]]
    df = df[df[lat_col] < bbox[3]]
    return df