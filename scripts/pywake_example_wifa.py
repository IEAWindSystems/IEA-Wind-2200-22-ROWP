# -*- coding: utf-8 -*-
"""
Description: Simple exemplary script to run a PyWake wind farm flow analysis using the WIFA pipeline.
This demonstrates the machine-actionable feature of the reference plant dataset provided according to the windIO ontology.
Author: Samuel Kainz
Date: 24/04/2026
"""

from wifa.main_api import run_api

# Run simulation. With WIFA multi-farm support (EUFLOW/WIFA#62), run_api
# returns one AEP value in GWh per wind farm, in the order the farms are
# listed in wind_farm.yaml (North, Mid, South).
per_farm_aep = run_api("../data/wind_energy_system.yaml")

for name, aep in zip(["North", "Mid", "South"], per_farm_aep):
    print(f"{name} farm AEP: {aep:.1f} GWh")
print(f"Total plant AEP: {sum(per_farm_aep):.1f} GWh")
