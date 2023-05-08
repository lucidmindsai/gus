# -*- coding: utf-8 -*-
import pytest
import json
import pandas as pd
from pygus.gus import Urban

def test_urban():
    scenario_file = "pygus/gus/inputs/scenario.json"
    try:
        f = open(scenario_file)
    except IOError as e:
        print(str(e))
    scenario = json.loads(f.read())
    model = Urban(
        population=pd.read_csv("pygus/gus/inputs/trees.csv"),
        species_composition="pygus/gus/inputs/allometrics.json",
        site_config="pygus/gus/inputs/site.json",
        scenario=scenario
    )
    
    _dict = model.run_model()
    assert _dict
    assert _dict["Replaced"]
    assert _dict["Alive"]
    assert _dict["Dead"]
    assert _dict["Cum_Seq"]
    
    assert len(_dict["Replaced"]) == len(_dict["Alive"])
    assert len(_dict["Replaced"]) == scenario.get("time_horizon_years")