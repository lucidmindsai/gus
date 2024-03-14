import os
import math
import random
import logging
import pandas as pd
import numpy as np
from typing import Union
from pkg_resources import resource_filename

from shapely.geometry import Polygon, MultiPolygon, Point

from .allometrics import Species
from .utilities import latlng_array_to_xy


def tree_population_from_geojson(
    geojson: Union[Polygon, MultiPolygon],
    num_trees,
    allometrics_file=None,
    dbh_range=[10, 15], # cm
    height_range=[2, 5], # m
    crownW_range=[1, 4],  # m (computed from dbh in the model)
    species={"Deciduous": 0.8, "Conifers": 0.2}, # % conifers
    condition_weights=[0.6, 0.3, 0.1, 0.0, 0.0], # excellent, good, fair, critical, dying
) -> pd.DataFrame:
    """This function generates a population of trees, with possibility for modifying the ranges of
    the different parameters. The trees are randomly distributed in a square of side length equal
    to the square root of the area in m2. The trees are assigned a species, a condition, a dbh, a
    height and a crown width.

    Args:
        geojson (dict): A geojson dictionary containing the coordinates of the area where the trees will be generated.
        num_trees (int): The number of trees to generate.

    Returns:
        pd.DataFrame: Returns a tree population which can be used as input for the model.
    """
    df = pd.DataFrame()
    _generate_locations_in_geojson(df, geojson, num_trees)

    #TODO: What do we do with this now?
    if allometrics_file:
        evergreen_pc = _get_evergreen_percentage(species, allometrics_file)
    else:
        evergreen_pc = _get_evergreen_percentage(species)

    return generate_population_features(
        df, dbh_range, height_range, crownW_range, species, condition_weights
    )


# Ideally, this function could take a dataframe, and add any missing elements needed for the simulation
# id, dbh, height, crownW, species, condition, xpos, ypos, lat, lng
def generate_population_features(
    df: pd.DataFrame,
    dbh_range=[10, 15],
    height_range=[2, 5],
    crownW_range=[1, 4],  # computed from dbh
    species={"Deciduous": 0.8, "Conifers": 0.2}, # % conifers
    condition_weights=[0.6, 0.3, 0.1, 0.0, 0.0],
) -> pd.DataFrame:
    """This function generates a population of trees, with possibility for modifying the ranges of
    the different parameters. The trees are randomly distributed in a square of side length equal
    to the square root of the area in m2. The trees are assigned a species, a condition, a dbh, a
    height and a crown width.

    Args:
        df (pd.DataFrame): _description_
        location (_type_): _description_
        area_m2 (_type_): _description_
        dbh_range (list, optional): _description_. Defaults to [10, 15].
        height_range (list, optional): _description_. Defaults to [2, 5].
        crownW_range (list, optional): _description_. Defaults to [1, 4].
        conifer_percentage (float, optional): _description_. Defaults to 0.2.
        condition_weights (list, optional): _description_. Defaults to [0.6, 0.3, 0.1, 0.0, 0.0].

    Returns:
        pd.DataFrame: Returns a tree population which can be used as input for the model.
    """
    ### id,lat,lng

    ## id,lat,lng,xpos,ypos
    if "xpos" not in df.columns or "ypos" not in df.columns:
        latlng_array_to_xy(df, "lat", "lng")

    ## id,lat,lng,xpos,ypos,species
    if "species" not in df.columns:
        # species is a dict of str to float
        df["species"] = random.choices(
            list(species.keys()), weights=list(species.values()),
            k=len(df),
        )

    ## id,lat,lng,xpos,ypos,species,condition
    if "condition" not in df.columns:
        df["condition"] = random.choices(
            ["excellent", "good", "fair", "critical", "dying"],
            weights=condition_weights,
            k=len(df),
        )

    ## id,lat,lng,xpos,ypos,species,condition,dbh
    if "dbh" not in df.columns:
        df["dbh"] = np.around(np.random.uniform(*dbh_range, len(df)), 3)

    ## id,lat,lng,xpos,ypos,species,condition,dbh,height
    if "height" not in df.columns:
        df["height"] = np.around(np.random.uniform(*height_range, len(df)), 3)

    ## id,lat,lng,xpos,ypos,species,condition,dbh,height,crownW
    if "crownW" not in df.columns:
        df["crownW"] = np.around(
            np.random.uniform(
                df["dbh"] * crownW_range[0], df["dbh"] * crownW_range[1], len(df)
            ),
            3,
        )
    if "id" not in df.columns:
        df.index.name = 'id'
        df.reset_index(inplace=True)

    column_out_order = ["id", "species", "dbh", "height", "lat", "lng", "condition", "crownW", "xpos", "ypos"]
    df.reindex(columns=column_out_order)
    return df
        
def _generate_locations_in_geojson(df, polygon: Union[Polygon, MultiPolygon], num_points):
    if polygon.geom_type == "Polygon":
        min_x, min_y, max_x, max_y = polygon.bounds
    elif polygon.geom_type == "MultiPolygon":
        min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    while len(points) < num_points:
        random_point = Point(
            [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        )
        if polygon.contains(random_point):
            points.append(random_point)

    df["lat"] = [point.y for point in points]
    df["lng"] = [point.x for point in points]

def _get_evergreen_percentage(species_dict, path = resource_filename("pygus", "gus/inputs/allometrics.json")):
    allometrics = Species(path)

    # check species_dict values total 1
    if not sum(species_dict.values()) - 1.0 < 1e-6:
        raise ValueError(f"Species dictionary values must sum to 1, got {sum(species_dict.values())}")
    
    evergreens = 0
    for species in species_dict:
        fuzzy = allometrics.fuzzymatch(species)
        if fuzzy is not None:
            if allometrics.is_evergreen(fuzzy):
                print(f"{species} is evergreen ({species_dict[species] * 100:.2f}%)")
                evergreens += species_dict[species]
        else:
            print(f"Species {species} not found in allometrics.json")

    logging.debug(f"Evergreen percentage: {evergreens * 100:.2f}%")
    return evergreens


def _generate_locations_in_square_area(
    df: pd.DataFrame, initial_coordinates, area_m2: float
) -> pd.DataFrame:
    lat = initial_coordinates[0]
    lng = initial_coordinates[1]
    half_side_length = np.sqrt(area_m2) / 2
    R = 6371e3  # Earth radius in metres

    bottom_left = [
        lat - (half_side_length / R) * (180 / math.pi),
        lng - (half_side_length / R) * (180 / math.pi) / math.cos(lat * math.pi / 180),
    ]
    top_right = [
        lat + (half_side_length / R) * (180 / math.pi),
        lng + (half_side_length / R) * (180 / math.pi) / math.cos(lat * math.pi / 180),
    ]
    bbox = [*bottom_left, *top_right]

    # for each element of the dataframe, assign a random lat and lng
    for i in range(len(df)):
        lat = np.random.uniform(bbox[1], bbox[3])
        lng = np.random.uniform(bbox[0], bbox[2])
        df.loc[i, "lat"] = lat
        df.loc[i, "lng"] = lng

def _bounding_box_to_geojson(bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]
        ],
    }