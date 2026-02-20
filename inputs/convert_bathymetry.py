"""Convert EMODnet bathymetry (lat/lon) to a regular UTM grid for windIO validation."""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "Bathymetry_EMODnet.nc"
DST = HERE / "Bathymetry_UTM.nc"

# UTM zone 31N (EPSG:32631) covers the North Sea area of this dataset
CRS_LATLON = "EPSG:4326"
CRS_UTM = "EPSG:32631"


def main():
    ds = xr.open_dataset(SRC)

    lat = ds.latitude.values  # 1-D, length 455
    lon = ds.longitude.values  # 1-D, length 520
    depth = ds.depth.values  # (lat, lon), float32

    # Build interpolator on the regular lat/lon grid
    interp = RegularGridInterpolator(
        (lat, lon), depth, method="linear", bounds_error=False, fill_value=np.nan
    )

    # Determine UTM extent from the 2D x/y already stored in the file
    x2d = ds.x.values  # (lat, lon)
    y2d = ds.y.values  # (lat, lon)
    x_min, x_max = float(np.nanmin(x2d)), float(np.nanmax(x2d))
    y_min, y_max = float(np.nanmin(y2d)), float(np.nanmax(y2d))

    # Build regular 1-D UTM axes at ~same resolution as lat/lon grid
    # Original lat spacing ≈ 1/960 deg ≈ ~115 m, lon spacing ≈ ~70 m
    dx = 100.0  # metres
    dy = 100.0
    x_utm = np.arange(x_min, x_max + dx, dx)
    y_utm = np.arange(y_min, y_max + dy, dy)

    # Map every UTM grid point back to lat/lon for interpolation
    transformer = Transformer.from_crs(CRS_UTM, CRS_LATLON, always_xy=True)
    xx, yy = np.meshgrid(x_utm, y_utm)
    lon_grid, lat_grid = transformer.transform(xx, yy)

    # Interpolate depth onto the regular UTM grid
    pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    depth_utm = interp(pts).reshape(xx.shape).astype(np.float32)

    # Write output
    ds_out = xr.Dataset(
        {
            "depth": (["y", "x"], depth_utm),
        },
        coords={
            "x": ("x", x_utm),
            "y": ("y", y_utm),
        },
    )
    ds_out.x.attrs = {"units": "m", "long_name": "Easting (UTM 31N)"}
    ds_out.y.attrs = {"units": "m", "long_name": "Northing (UTM 31N)"}
    ds_out.depth.attrs = {"units": "m", "long_name": "Water depth"}

    ds_out.to_netcdf(DST)
    print(f"Wrote {DST}  ({len(x_utm)} x {len(y_utm)} grid)")


if __name__ == "__main__":
    main()
