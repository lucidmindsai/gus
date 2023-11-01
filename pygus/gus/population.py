import random
import pandas as pd
import numpy as np
from .utilities import latlng_array_to_xy

def generate_tree_population_dataframe(size=1000) -> pd.DataFrame:
        # id,dbh,height,CrownW,condition,species,GRID_ID_200,lat,lng,GRID_ID_100
    out_df = pd.DataFrame()
    # WARNING
        # Here we create the tree population assuming:
        # 1. All the newly planted trees will be roughly the same size
        # 2. The newly planted trees will be in pretty good condition
        
    species_ratio = 0.2 # TODO: make this a parameter
    
    # for each row of the scenario DF, we need to pick a condition and a species
    out_df["species"] = random.choices(
        ["Deciduous", "Conifers"],
        weights=[1 / (1 + species_ratio), species_ratio / (1 + species_ratio)],
        k=size,
    )
    out_df["condition"] = random.choices(
        ["excellent", "good", "fair"],
        weights=[0.6, 0.3, 0.1],
        k=size,
    )
    
    # remaining: height, crownW
    out_df["dbh"] = np.around(np.random.uniform(10, 15, size), 3)
    out_df["height"] = np.around(np.random.uniform(3, 6, size), 3)
    out_df["CrownW"] = np.around(np.random.uniform(1, 3, size), 3)
    
    # add utm xy
    out_df = latlng_array_to_xy(out_df, "lat", "lng")
    
    return out_df

def generate_scenario_population(actual, scenario):
    pass