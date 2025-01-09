"""The module holds general purpose python functions on data preparations."""
import utm
import math
import logging
import numpy as np
import pandas as pd
from functools import reduce
from shapely.geometry import shape, Point

def get_raster_data(
    dtwin,  # Urban
    var="unique_id",
    probe=None,
    predicate=None,
    init=0,
    counts=False,
):
    """A general purpose function that collects cellular data on the grid.

    Args:
            dtwin: (:obj:`Urban`): an urban forest model.
            var: (:obj:`string`): a valid memory variable of a tree.
            probe: (callable): a function to combine elements. Defaults to addition.
            predicate: (callable): a function to filter elements. Defaults to returning True for all elements.
            init: (any): initial value for the reduction.
            counts: (bool): if True, count the number of items.

    Returns:
            (:obj:`numpy.ndarray`): raster-like data on a 2D grid.
    """
    logging.debug(dtwin.df.value_counts("condition"))

    def increment(x, y):
        return x + 1

    def default_probe(x, y):
        return x + y

    def always_true_predicate(x):
        return True

    if counts:
        probe = increment

    # Ensure probe is defined
    probe = default_probe if 'probe' not in locals() else probe

    # Ensure predicate is defined
    predicate = always_true_predicate if 'predicate' not in locals() else predicate

    raster = np.zeros((dtwin.grid.width, dtwin.grid.height))

    for cell in dtwin.grid.coord_iter():
        cell_content, x, y = cell
        cell_content = [a for a in cell_content if a.condition.upper() not in ["DEAD", "REPLACED"]]
        filtered = filter(predicate, [getattr(a, var) for a in cell_content])
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

    population_df.loc[:, 'xpos'] = easting.astype(int)
    population_df.loc[:, 'ypos'] = northing.astype(int)

    population_df.loc[:, 'xpos'] = population_df['xpos'] - population_df['xpos'].min()
    population_df.loc[:, 'ypos'] = population_df['ypos'] - population_df['ypos'].min()

def calculate_dataframe_area(tree_df: pd.DataFrame):
    """Calculates the area in UTM grid units (or square metres) of a dataframe of trees.
    
    Args:
        tree_df (pd.DataFrame): The dataframe of trees to calculate the area of

    Returns:
        _type_: Returns the area of the dataframe in UTM grid units (or square metres)
    """
    MIN_AREA = 1
    if "lat" in tree_df.columns and "lng" in tree_df.columns:
        R = 6371000  # Radius of the Earth in meters
        min_lat = tree_df["lat"].min()
        max_lat = tree_df["lat"].max()
        min_lon = tree_df["lng"].min()
        max_lon = tree_df["lng"].max()

        lat1_rad = math.radians(min_lat)
        lat2_rad = math.radians(max_lat)
        lon1_rad = math.radians(min_lon)
        lon2_rad = math.radians(max_lon)

        lat_avg = (lat1_rad + lat2_rad) / 2

        d_lat = lat2_rad - lat1_rad
        d_lon = lon2_rad - lon1_rad

        return max(d_lat * d_lon * R * R * math.cos(lat_avg), MIN_AREA)
    elif "xpos" in tree_df.columns and "ypos" in tree_df.columns:
        min_x = tree_df["xpos"].min()
        max_x = tree_df["xpos"].max()
        min_y = tree_df["ypos"].min()
        max_y = tree_df["ypos"].max()
        return max((max_x - min_x) * (max_y - min_y), MIN_AREA)

    raise ValueError("Unable to calculate area: dataframe does not contain latitude/longitude or x/y positions")

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

def filter_dataframe_to_geojson_polygon_or_multipolygon(df: pd.DataFrame, geojson, lat_col = "lat", lng_col = "lng") -> pd.DataFrame:
    """Filters a pandas dataframe with `lat` and `lng` columns to a GeoJSON polygon or multipolygon.
    This is done using the shapely library

    Args:
        df (pd.DataFrame): The dataframe to filter
        geojson (dict): A GeoJSON dictionary representing a polygon or multipolygon
        lat_col (str, optional): The name for the latitude column. Defaults to "lat".
        lng_col (str, optional): The name for the longitude column. Defaults to "lng".

    Returns:
        pd.DataFrame: Returns the filtered dataframe
    """

    assert type(geojson) == dict
    assert geojson["type"] in ["Polygon", "MultiPolygon"]
    
    polygon = shape(geojson)
    def filter_fn(row):
        point = Point(row[lng_col], row[lat_col])
        return polygon.contains(point)
    
    return df[df.apply(filter_fn, axis=1)]