import math
import random
import pandas as pd
import numpy as np
from .utilities import latlng_array_to_xy


def generate_population_allometrics(
    df: pd.DataFrame,
    location,
    area_m2,
    dbh_range=[10, 15],
    height_range=[2, 5],
    crownW_range=[1, 4],
    conifer_percentage=0.2,
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
    if "lat" not in df.columns or "lng" not in df.columns:
        df = generate_tree_locations(df, location, area_m2)

    ## id,lat,lng,xpos,ypos
    if "xpos" not in df.columns or "ypos" not in df.columns:
        df = latlng_array_to_xy(df, "lat", "lng")

    ## id,lat,lng,xpos,ypos,species
    if "species" not in df.columns:
        df["species"] = random.choices(
            ["Deciduous", "Conifers"],
            weights=[
                1 / (1 + conifer_percentage),
                conifer_percentage / (1 + conifer_percentage),
            ],
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

    df.reset_index(inplace=True)
    return df


def generate_tree_locations(
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

    return df


def generate_scenario_population(actual, scenario):
    pass