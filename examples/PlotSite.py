import numpy as np
import utm
import matplotlib.pyplot as plt
from windIO.utils.yml_utils import load_yaml

# Load data
system = load_yaml('..\inputs\wind_energy_system.yaml')

# Extract site and wind farm
b = system['site']
farm = system['wind_farm']

# Extract bathymetry data
X = np.array(system['site']['Bathymetry']['latitude'])
Y = np.array(system['site']['Bathymetry']['longitude'])
Z = np.array(system['site']['Bathymetry']['elevation']['data'])

# Transfer from LongLat to UTM (km)
X_utm = utm.from_latlon(np.ones(len(Y))*X[0],Y)
Y_utm = utm.from_latlon(X,np.ones(len(X))*Y[0])

# # Extract turbine coordinates
Tx = farm['layouts']['initial_layout']['coordinates']['x']
Ty = farm['layouts']['initial_layout']['coordinates']['y']

# # Cable and substation data
Subs_x = system['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['x']
Subs_y = system['wind_farm']['electrical_substations']['electrical_substation']['coordinates']['y']

# Create plot
fig = plt.figure()
ax = fig.gca()
ax.set_aspect(1)

# # Plot bathymetry: filled contour plots
CS = ax.contourf(X_utm[0]/1000,Y_utm[1]/1000, -Z, 100, cmap=plt.colormaps.get_cmap('Blues'))
cb = fig.colorbar(CS)
cb.set_label('Depth [m]',fontsize=9)
cb.ax.invert_yaxis()
cb.set_ticks(np.arange(21,35,2))
cb.ax.tick_params(labelsize=8)

# Plot boundaries
for i in range(len(b['boundaries']['polygons'])):
    bx = b['boundaries']['polygons'][i]['x']
    by = b['boundaries']['polygons'][i]['y']
    bx.append(bx[0])
    by.append(by[0])
    if i == 1:
        ax.plot([x/1000 for x in bx],[y/1000 for y in by],color='k',linewidth=1,label='Boundary')
    else:
        ax.plot([x/1000 for x in bx],[y/1000 for y in by],color='k',linewidth=1)
        
# Plot turbines
# ax.scatter([x/1000 for x in Tx], [y/1000 for y in Ty], c='darkorange', marker='2', zorder=3, linewidth=1.5, label='Turbine')

# Plot Substation
ax.scatter([x/1000 for x in Subs_x],[y/1000 for y in Subs_y],marker='s',s=7,color='k', zorder=3,label='Substation')

# Labels, legend, grid
ax.set_ylabel('Northing [km]',fontsize=9)
ax.set_xlabel('Easting [km]',fontsize=9)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_title('HKW', pad=10, fontsize=10)
ax.legend(prop={'size': 8},loc = 'upper left')
ax.grid(alpha=0.6)