# -*- coding: utf-8 -*-
"""
Description: Simple exemplary script to run a PyWake wind farm flow analysis using the WIFA pipeline.
This demonstrates the machine-actionable feature of the reference plant dataset provided according to the windIO ontology.
Author: Samuel Kainz
Date: 24/04/2026
"""

from wifa.main_api import run_api
# Run simulation
results = run_api("../data/wind_energy_system.yaml")