# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:31:32 2024

@author: Samuel Kainz
"""

import matplotlib
import os
from openmdao.api import ExplicitComponent
import matplotlib.pyplot as plt
import numpy as np
import topfarm
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle, Polygon

def mypause(interval):
    # pause without show
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

class XYPlotCompBathym(ExplicitComponent):
    """Plotting component for turbine locations"""
    # colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 100
    colors = [c['color'] for c in iter(matplotlib.rcParams['axes.prop_cycle'])] * 100

    def __init__(self, memory=10, delay=0.001, plot_initial=True, plot_improvements_only=False, ax=None, legendloc=1, save_plot_per_iteration=False, X=None, Y=None, Z=None, Sx=None, Sy=None, cables=None, metrics_recorder=None, Xn=[], Yn=[], b=[], opt_nr=None, folder='Figures', sampling=False, obj=None, optimize=True, ploteach=1, iter_nr=1, paper=False):
        """Initialize component for plotting turbine locations

        Parameters
        ----------
        memory : int, optional
            Number of previous iterations to remember
        delay : float, optional
            Time delay in seconds between plotting updates
        plot_initial : bool, optional
            Flag to plot the initial turbine locations
        plot_improvements_only : bool, optional
            Flag to plot only improvements in cost
        ax : matplotlib axes, optional
            Axes into which to make the plot
        legendloc : int
            Location of the legend in the plot
        """
        ExplicitComponent.__init__(self)
        self.delay = delay
        self.plot_improvements_only = plot_improvements_only
        self._ax = ax
        self.memory = memory
        self.delay = max([delay, 1e-6])
        self.plot_initial = plot_initial
        self.history = []
        self.counter = 0
        self.by_pass = False
        self.legendloc = legendloc
        self.save_plot_per_iteration = save_plot_per_iteration
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Sx = Sx
        self.Sy = Sy
        self.cables = cables
        self.metrics_recorder = metrics_recorder
        self.Xn = Xn
        self.Yn = Yn
        self.b = b
        self.opt_nr = opt_nr
        self.folder = folder
        self.sampling = sampling
        self.obj = obj
        self.optimize = optimize
        self.ploteach = ploteach
        self.iter_nr = iter_nr
        self.paper = paper
        self.FS = 9
    @property
    def ax(self):
        return self._ax or plt.gca()

    def show(self):
        plt.show()

    def setup(self):
        if topfarm.x_key in self.problem.design_vars:
            units_x = self.problem.design_vars[topfarm.x_key][-1]
        else:
            units_x = None
        if topfarm.y_key in self.problem.design_vars:
            units_y = self.problem.design_vars[topfarm.y_key][-1]
        else:
            units_y = None
        self.add_input(topfarm.x_key, np.zeros(self.n_wt), units=units_x)
        self.add_input(topfarm.y_key, np.zeros(self.n_wt), units=units_y)
        if hasattr(self.problem, 'xy_boundary'):
            self.xy_boundary = self.problem.xy_boundary
        if hasattr(self.problem.cost_comp, 'output_key'):
            self.cost_key = self.problem.cost_comp.output_key
            self.cost_unit = self.problem.cost_comp.output_unit
        else:
            self.cost_key = "Cost"
            self.cost_unit = ""
        self.add_input(self.cost_key, 0.)
        self.add_output('plot_counter')

    def init_plot(self, limits):
        self.ax.cla()
        fig = plt.gcf()
        fig.set_size_inches(4.72, 5)
        fig.tight_layout()

        # self.ax.axis('equal')

        mi = limits.min(0)
        ma = limits.max(0)
        ra = ma - mi + 1
        ext = .1
        xlim, ylim = np.array([mi - ext * ra, ma + ext * ra]).T
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
    def plot_bathymetry(self):
        CS = self.ax.contourf(self.X, self.Y, np.ma.masked_invalid(self.Z), 100, cmap=plt.colormaps.get_cmap('Blues'))
        CS.set_linewidth(0)  # Or use a list of widths if you need different values per level
        CS.set_edgecolor('face')  # Alternatively, you could use this to turn off edges
        if self.paper:
            CS.set_rasterized(True)
        self.ax.set_aspect(1)
        fig = plt.gcf()
        # Check if a colorbar already exists in the figure
        colorbar_exists = any(isinstance(ax, plt.Axes) and ax.get_label() == '<colorbar>' for ax in fig.axes)
        if not colorbar_exists:
            cb = fig.colorbar(CS)
            cb.set_label('Depth [m]',fontsize=self.FS)
            cb.ax.invert_yaxis()
            cb.set_ticks(np.arange(21,36,2))
            cb.ax.tick_params(labelsize=self.FS-1)

#     def plot_boundary(self):
#         b = np.r_[self.xy_boundary[:], self.xy_boundary[:1]]
#         plt.plot(b[:, 0], b[:, 1], 'k')

    def plot_constraints(self):
        for constr in self.problem.model.constraint_components:
            constr.plot(self.ax)
            for line in self.ax.get_lines():
                line.set_label('_nolegend_')
    def plot_history(self, x, y):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            def get(xy, xy_key, pw):
                rec_xy = pw[xy_key][-self.memory:]
                if len(rec_xy.shape) == 1:
                    rec_xy = rec_xy[:, np.newaxis]
                return np.r_[rec_xy, [xy]]
            pw = self.problem.get_vars_from_recorder()
            x = get(x, topfarm.x_key, pw)
            y = get(y, topfarm.y_key, pw)
            for c, x_, y_ in zip(self.colors, x.T, y.T):
                self.ax.plot(x_, y_, '--', color=c)

    def plot_initial2current(self, x0, y0, x, y):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            pw = self.problem.get_vars_from_recorder()
            x0 = np.atleast_1d(pw['x0'])
            y0 = np.atleast_1d(pw['y0'])
            for c, x0_, y0_, x_, y_ in zip(self.colors, x0, y0, x, y):
                self.ax.plot(x0_, y0_, '>', markerfacecolor=c, markeredgecolor='k')
                self.ax.plot((x0_, x_), (y0_, y_), '-', color=c)
            self.ax.plot([], [], '>k', markerfacecolor="#00000000", markeredgecolor='k', label='Initial position')

    def plot_current_position(self, x, y):
        # if self.paper:
        self.ax.scatter(x, y, facecolors='darkorange', edgecolors='black', marker='^', s=25, zorder=3, linewidth=0.5, label='Turbine')
        # else:
            # for c, x_, y_ in zip(self.colors, x, y):
            #     self.ax.plot(x_, y_, 'o', color=c, ms=5)
            #     self.ax.plot(x_, y_, 'xk', ms=4)
            # self.ax.plot([], [], 'xk', label='Current position')
            
    def plot_tur_spacing(self, x, y):
        for i, (x, y) in enumerate(zip(x, y)):
            circle = Circle((x, y), self.metrics_recorder['general_settings'][0]['d_RD']*283.2181/2,
                            facecolor=(255/255, 140/255, 0/255, 0.3), # darkorange with alpha = 0.3
                            edgecolor='darkorange',
                            linewidth=0.7,
                            label='Spacing constraint' if i == 0 else None)
            self.ax.add_patch(circle)
            
    def plot_nb_spacing(self, x, y):
        for x, y in zip(x,y):
            circle = Circle((x, y), self.metrics_recorder['general_settings'][0]['d_RD']*283.2181/2,
                            facecolor=(255/255, 140/255, 0/255, 0.3), # darkorange with alpha = 0.3
                            edgecolor='darkorange',
                            linewidth=0.7)
            self.ax.add_patch(circle)
        
    def plot_nb_position(self, x, y):
        # if self.paper:
        self.ax.scatter(x, y, facecolors='darkorange', edgecolors='black', marker='^', s=25, zorder=3, linewidth=0.5)
        # else:
        #     for c, x_, y_ in zip(self.colors, x, y):
        #         self.ax.plot(x_, y_, 'o', color=c, ms=5)
        #         self.ax.plot(x_, y_, 'xk', ms=4)
        
    def plot_boundaries(self):
        wf = {
            "north": 0,
            "mid": 1,
            "south": 2,
            "all": 5
        }
        i_opt = wf[self.metrics_recorder['cur_zone'][-1][0]]
        
        b = self.b
        for i in range(len(b)):
            bx = b[i][:,0]
            by = b[i][:,1]
            bx = np.append(bx,bx[0])
            by = np.append(by,by[0])
            if self.optimize and i == i_opt:
                polygon = Polygon(np.c_[bx, by], closed=True, facecolor='blueviolet', edgecolor='none', alpha=0.17)
                self.ax.add_patch(polygon)
            if i == 0:
                self.ax.plot(bx,by,color='k',linewidth=0.5,label='Boundary')
            else:
                self.ax.plot(bx,by,color='k',linewidth=0.5)
        
    def plot_cables(self,x,y):
        # CabName = ['Cable A=' + str(round(self.cables[0][0])) + 'mm²','Cable A=' + str(round(self.cables[1][0])) + 'mm²','Cable A=' + str(round(self.cables[2][0])) + 'mm²']
        CabName = [f'Cable A={round(c[0])}mm$^2$' for c in self.cables]
        # go through the different zones
        plot2 = 0           # helper to plot legend of cable types only once
        cabplot = [0,0,0]   #            - " -
        for idx, zone in enumerate(self.metrics_recorder['sequence']):
            if self.metrics_recorder['cable_u_' + zone]:
                u = self.metrics_recorder['cable_u_' + zone][-1]
                v = self.metrics_recorder['cable_v_' + zone][-1]
                con = list(zip([x + 1 for x in v], [x + 1 for x in u], self.metrics_recorder['cable_type_' + zone][-1]))
                # Plot cabling
                # a) Combine turbine + subsation coordinates
                AllX = [self.Sx[zone]] + self.metrics_recorder['x_' + zone][-1]
                AllY = [self.Sy[zone]] + self.metrics_recorder['y_' + zone][-1]
                # b) helper for plot
                lw = [0.5,1,1.7]    # line width of different cable types
                # c) go through all turbines and plot connection
                for i in range(len(con)):
                    if cabplot[con[i][2]] == 0 and con[i][2] == plot2:
                        self.ax.plot([AllX[con[i][0]],AllX[con[i][1]]],[AllY[con[i][0]],AllY[con[i][1]]],color='firebrick',linewidth=lw[con[i][2]],label=CabName[con[i][2]])
                        cabplot[con[i][2]] = 1
                        plot2 = plot2 + 1
                    else:
                        self.ax.plot([AllX[con[i][0]],AllX[con[i][1]]],[AllY[con[i][0]],AllY[con[i][1]]],color='firebrick',linewidth=lw[con[i][2]])
        
    def plot_substations(self):
        self.ax.scatter(list(self.Sx.values()),list(self.Sy.values()),marker='s',s=7,color='k', zorder=3,label='Substation')
    
    def set_title(self):
        # Overall optimization
        if self.optimize:
            title = "\nIteration: %d"  % (self.metrics_recorder['iteration'][-1]-1)
        else:
            title = ''
        # Plot lcoe if no sampling
        if not self.sampling and self.obj:
            if self.obj == 'lcoe':
                unit = '$/MWh'
                divider = 1
                obj = 'LCOE'
            else:
                unit = 'GWh'
                divider = 1000
                obj = 'AEP'
            # For sequential layout, add overall LCOE
            title += "Overall " + obj + " = %.2f %s (%+.2f%%)" % (self.metrics_recorder[self.obj + '_all'][-1] / divider, unit, (self.metrics_recorder[self.obj + '_all'][-1] - self.metrics_recorder[self.obj + '_all'][0]) / self.metrics_recorder[self.obj + '_all'][0] * 100)
            #
            items = ['north','mid','south']
            items = [item for item in items if item in self.metrics_recorder['sequence']]
            for idx, zone in enumerate(items):
                if self.metrics_recorder[self.obj + '_' + zone][-1] != 0:
                    title += " \n" + obj + ' ' + zone + " = %.2f %s (%+.2f%%)" % (self.metrics_recorder[self.obj + '_' + zone][-1] / divider, unit, (self.metrics_recorder[self.obj + '_' + zone][-1] - self.metrics_recorder[self.obj + '_' + zone][next(i for i, value in enumerate(self.metrics_recorder[self.obj + '_' + zone]) if value != 0)]) / (self.metrics_recorder[self.obj + '_' + zone][next(i for i, value in enumerate(self.metrics_recorder[self.obj + '_' + zone]) if value != 0)]) * 100)
                else:
                    title += " \n" + obj + " " + zone + " = - " + unit + " (-%)"
        self.ax.set_title(title,fontsize=self.FS)
        
    def get_initial(self):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            pw = self.problem.get_vars_from_recorder()
            cost0 = self.problem.recorder[self.cost_key][0]
            # cost0 = pw['cost0']
            return pw['x0'], pw['y0'], cost0

    def compute(self, inputs, outputs):
        if (self.metrics_recorder['iteration'][-1]-1) % self.ploteach == 0:
            # find limits
            def get_lim(key):
                if (key in self.problem.design_vars and
                        isinstance(self.problem.design_vars[key], tuple) and
                        len(self.problem.design_vars[key]) == 4):
                    return np.min(self.problem.design_vars[key][1]), np.max(np.min(self.problem.design_vars[key][2]))
                else:
                    return min(inputs[key]), max(inputs[key])
            if self.optimize:
                min_x, max_x = get_lim('x')
                min_y, max_y = get_lim('y')
            else:
                min_x = min(inputs['x'])
                max_x = max(inputs['x'])
                min_y = min(inputs['y'])
                max_y = max(inputs['y'])
            self.init_plot(np.array([[min_x, min_y], [max_x, max_y]]))
            
            self.plot_bathymetry()
            
            if len(self.b) > 0:
                self.plot_boundaries()
                
            x = inputs['x']
            y = inputs['y']
                
            if self.optimize:
                self.plot_constraints()
            else:
                self.plot_tur_spacing(x, y)

            self.plot_cables(x,y)
            self.plot_substations()
            self.plot_current_position(x, y)
            if len(self.Xn) > 0:
                self.plot_nb_position(self.Xn,self.Yn)
                self.plot_nb_spacing(self.Xn,self.Yn)
            
            self.set_title()
            self.ax.legend(loc='upper left',fontsize=self.FS-1)
            
            self.ax.grid(alpha=0.6)
            self.ax.tick_params(axis='both', which='major', labelsize=self.FS-1)
            
            # Format axes to display in kilometers
            def meters_to_kilometers(x, _):
                return f'{x / 1000:.0f}'
            plt.gca().xaxis.set_major_formatter(FuncFormatter(meters_to_kilometers))
            plt.gca().yaxis.set_major_formatter(FuncFormatter(meters_to_kilometers))
            self.ax.set_ylabel('Northing [km]',fontsize=self.FS)
            self.ax.set_xlabel('Easting [km]',fontsize=self.FS)
            self.ax.set_xlim([533200,562500])
            self.ax.set_ylim([5813300,5852700])
            # self.ax.tick_params(labelsize=self.FS-1)
            # self.ax.set_xlim([525700,567000])
            # self.ax.set_ylim([5804500,5862500])
            
            plt.gcf().tight_layout()
            
            if not self.optimize:
                plt.gcf().subplots_adjust(
                    top=0.864,
                    bottom=0.083,
                    left=0.022,
                    right=0.978,
                    hspace=0.2,
                    wspace=0.2
                )
            else:
                plt.gcf().subplots_adjust(
                    top=0.952,
                    bottom=0.093,
                    left=0.119,
                    right=0.974,
                    hspace=0.2,
                    wspace=0.2
                )
                     
            # if self.counter == 0:
            #     plt.pause(1)
            # mypause(self.delay)
            
            if self.optimize:
                self.counter += 1
                outputs['plot_counter'] = self.counter
            else:
                self.counter = self.iter_nr
                
            if self.save_plot_per_iteration:
                if not os.path.exists(self.folder):
                    os.makedirs(self.folder)
                if self.opt_nr:
                    plt.savefig(self.folder + '/iteration_z' + str(self.opt_nr) + '_%s.png' % self.counter, dpi=200, pad_inches=0)
                else:
                    plt.savefig(self.folder + '/iteration_%s.png' % self.counter)