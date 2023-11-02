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

    ## These could be determined by the model, so I understood we shouldn't bother creating these..
    if "height" not in df.columns:
        df["height"] = np.around(np.random.uniform(*height_range, len(df)), 3)

    if "CrownW" not in df.columns:
        df["CrownW"] = np.around(np.random.uniform(*crownW_range, len(df)), 3)

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


# FRom GPT:

estimated_allometrics_at_maturity = {
    "Lime - small": {"Height (m)": (15, 20), "DBH (cm)": (40, 60)},
    "Oak Sessile": {"Height (m)": (20, 40), "DBH (cm)": (70, 120)},
    "Oak English": {"Height (m)": (20, 40), "DBH (cm)": (70, 120)},
    "Hornbeam": {"Height (m)": (15, 25), "DBH (cm)": (40, 80)},
    "Willow goat": {"Height (m)": (10, 15), "DBH (cm)": (30, 50)},
    "Willow bay": {"Height (m)": (6, 12), "DBH (cm)": (20, 40)},
    "Rowan": {"Height (m)": (6, 15), "DBH (cm)": (20, 40)},
    "Maple": {"Height (m)": (10, 25), "DBH (cm)": (30, 60)},
    "Crab apple": {"Height (m)": (4, 10), "DBH (cm)": (15, 30)},
    "Willow - grey": {"Height (m)": (4, 12), "DBH (cm)": (20, 40)},
    "Yew": {"Height (m)": (10, 20), "DBH (cm)": (30, 60)},
    "Holly": {"Height (m)": (2, 15), "DBH (cm)": (10, 30)},
    "Hazel": {"Height (m)": (4, 8), "DBH (cm)": (15, 25)},
    "Hawthorn": {"Height (m)": (5, 15), "DBH (cm)": (20, 40)},
    "Blackthorn": {"Height (m)": (2, 4), "DBH (cm)": (10, 20)},
    "Whitebeam": {"Height (m)": (8, 15), "DBH (cm)": (20, 40)},
    "Willow wooly": {"Height (m)": (3, 8), "DBH (cm)": (10, 25)},
}
