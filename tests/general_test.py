# -*- coding: utf-8 -*-
import logging
import pytest
import time
import json
import pandas as pd

from pygus import Urban, SiteConfig, latlng_array_to_xy


# input_trees = "pygus/gus/inputs/trees.csv"
input_trees = "pygus/gus/inputs/amsterdam_all_trees_1000.csv"


def test_parallelise():
    site_file = "pygus/gus/inputs/site.json"
    tree_population = pd.read_csv(input_trees)
    scenario_file = "pygus/gus/inputs/scenario.json"
    try:
        f = open(scenario_file)
    except IOError as e:
        logging.error(str(e))
    scenario = json.loads(f.read())

    model = Urban(
        population=tree_population,
        species_allometrics_file="pygus/gus/inputs/allometrics.json",
        site_config=SiteConfig.from_file(site_file),
        scenario=scenario,
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
