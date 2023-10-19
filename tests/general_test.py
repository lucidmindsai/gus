# -*- coding: utf-8 -*-
import pytest
import time
import json
import pandas as pd

from pygus.gus import Urban
from pygus.gus.utilities import latlng_array_to_xy, load_site_config_file

# input_trees = "pygus/gus/inputs/trees.csv"
input_trees = "pygus/gus/inputs/amsterdam_all_trees_1000.csv"


def test_parallelise():
    scenario_file = "pygus/gus/inputs/scenario.json"
    site_file = "pygus/gus/inputs/site.json"
    try:
        scen = json.loads(open(scenario_file).read())
    except IOError as e:
        print(str(e))

    tree_population = latlng_array_to_xy(pd.read_csv(input_trees))
    model = Urban(
        population=pd.read_csv(input_trees),
        species_allometrics_file="pygus/gus/inputs/allometrics.json",
        site_config=load_site_config_file(site_file),
        scenario=scen,
    )

    start = time.time()
    # model.run(scenario.get("time_horizon_years"))
    impacts: pd.DataFrame = model.run()
    end = time.time()
    print("Time elapsed: {}".format(end - start))

    assert not impacts.empty
    assert not impacts["Replaced"].empty
    assert not impacts["Avg_Seq"].empty
    assert not impacts["Alive"].empty
    assert not impacts["Dead"].empty
    assert not impacts["Cum_Seq"].empty

    assert len(impacts["Replaced"]) == len(impacts["Alive"])
    assert len(impacts["Replaced"]) == scenario.get("time_horizon_years")
