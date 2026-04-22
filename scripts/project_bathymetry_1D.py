# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:34:01 2026

@author: Samuel
"""

import os
from pathlib import Path
import utm
import windIO
import numpy as np
import netCDF4 as nc

# --- Open original NetCDF file ---
infile = r"..\inputs\Bathymetry_EMODnet_org.nc"
ds = nc.Dataset(infile, "r")

# --- Load 2D arrays ---
X = np.array(ds['latitude'][:])    # 1D or 2D
Y = np.array(ds['longitude'][:])   # 1D or 2D
elevation = np.array(ds['elevation'][:])  # 2D grid
ds.close()

# --- Create 2D grids ---
lon_grid, lat_grid = np.meshgrid(Y, X)

# --- Convert to UTM ---
easting, northing, zone_number, zone_letter = utm.from_latlon(lat_grid, lon_grid)

# --- Flatten to 1D ---
x = easting.ravel()
y = northing.ravel()
depth = elevation.ravel()

# --- Create new NetCDF file ---
outfile = r"..\inputs\Bathymetry_UTM_1D.nc"
ncfile = nc.Dataset(outfile, "w", format="NETCDF4")

# --- Create dimension ---
ncfile.createDimension("points", x.size)

# --- Create variables ---
x_var = ncfile.createVariable("x", "f8", ("points",))
y_var = ncfile.createVariable("y", "f8", ("points",))
depth_var = ncfile.createVariable("depth", "f4", ("points",), fill_value=np.nan)

# --- Write data ---
x_var[:] = x
y_var[:] = y
depth_var[:] = depth

# --- Metadata ---
x_var.units = "m"
x_var.long_name = "Northing - WGS84 / UTM zone 31N / EPSG:32631"
x_var.standard_name = "projection_x_coordinate"

y_var.units = "m"
y_var.long_name = "Easting - WGS84 / UTM zone 31N / EPSG:32631"
y_var.standard_name = "projection_y_coordinate"

depth_var.units = "m"
depth_var.long_name = "Depth linked with coordinates"
depth_var.standard_name = "depth"
depth_var.coordinates = "x y"

# --- Close file ---
ncfile.close()
