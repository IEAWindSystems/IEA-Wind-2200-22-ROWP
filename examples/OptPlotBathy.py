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

    def __init__(self, memory=10, delay=0.001, plot_initial=True, plot_improvements_only=False, ax=None, legendloc=1, save_plot_per_iteration=False, X=None, Y=None, Z=None, Sx=None, Sy=None, cables=None, metrics_recorder=None, Xn=[], Yn=[], b=[], opt_nr=None):
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
        fig.set_size_inches(10, 5)
        # self.ax.axis('equal')

        mi = limits.min(0)
        ma = limits.max(0)
        ra = ma - mi + 1
        ext = .1
        xlim, ylim = np.array([mi - ext * ra, ma + ext * ra]).T
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
    def plot_bathymetry(self):
        CS = self.ax.contourf(self.X[0], self.Y[1], -self.Z, 100, cmap=plt.colormaps.get_cmap('Blues'))
        self.ax.set_aspect(1)
        fig = plt.gcf()
        # Check if a colorbar already exists in the figure
        colorbar_exists = any(isinstance(ax, plt.Axes) and ax.get_label() == '<colorbar>' for ax in fig.axes)
        if not colorbar_exists:
            cb = fig.colorbar(CS)
            cb.set_label('Depth [m]',fontsize=8)
            cb.ax.invert_yaxis()
            cb.set_ticks(np.arange(21,35,2))
            cb.ax.tick_params(labelsize=7)

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
        for c, x_, y_ in zip(self.colors, x, y):
            self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_, 'xk', ms=4)
        self.ax.plot([], [], 'xk', label='Current position')
        
    def plot_nb_position(self, x, y):
        for c, x_, y_ in zip(self.colors, x, y):
            self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_, 'xk', ms=4)
        
    def plot_boundaries(self):
        b = self.b
        for i in range(len(b)):
            bx = b[i][:,0]
            by = b[i][:,1]
            bx = np.append(bx,bx[0])
            by = np.append(by,by[0])
            if i == 0:
                self.ax.plot(bx,by,color='k',linewidth=0.5,label='Boundary')
            else:
                self.ax.plot(bx,by,color='k',linewidth=0.5)
        
    def plot_cables(self,x,y):
        u = self.metrics_recorder['cable_u'][-1]
        v = self.metrics_recorder['cable_v'][-1]
        con = list(zip([x - 1 for x in v], [x - 1 for x in u], self.metrics_recorder['cable_type'][-1]))
        CabName = ['Cable A=' + str(self.cables[0][0]) + 'mm²','Cable A=' + str(self.cables[1][0]) + 'mm²','Cable A=' + str(self.cables[2][0]) + 'mm²']
        # Plot cabling
        # a) Combine turbine + subsation coordinates
        if len(self.Xn) == 0:
            AllX = self.Sx + x.tolist()
            AllY = self.Sy + y.tolist()
        else:
            AllX = self.Sx + x.tolist() + self.Xn.tolist()
            AllY = self.Sy + y.tolist() + self.Yn.tolist()
        # b) helper for plot
        lw = [0.5,1,1.7]    # line width of different cable types
        plot2 = 0           # helper to plot legend of cable types only once
        cabplot = [0,0,0]   #            - " -
        # c) go through all turbines and plot connection
        for i in range(len(AllX)-len(self.Sx)):
            if cabplot[con[i][2]] == 0 and con[i][2] == plot2:
                self.ax.plot([AllX[con[i][0]],AllX[con[i][1]]],[AllY[con[i][0]],AllY[con[i][1]]],color='firebrick',linewidth=lw[con[i][2]],label=CabName[con[i][2]])
                cabplot[con[i][2]] = 1
                plot2 = plot2 + 1
            else:
                self.ax.plot([AllX[con[i][0]],AllX[con[i][1]]],[AllY[con[i][0]],AllY[con[i][1]]],color='firebrick',linewidth=lw[con[i][2]])
        # d) Plot Substation
        self.ax.scatter(self.Sx,self.Sy,marker='s',s=7,color='k', zorder=3,label='Substation')
    def set_title(self, cost0, cost):
        # Overall optimization
        title = "\nIteration: %d"  % (self.metrics_recorder['iteration'][-1]-1)
        # For sequential layout, add overall LCOE
        title += "\n" + "Overall LCOE" + " = %.2f $/MWh (%+.2f%%)" % (self.metrics_recorder['lcoe_all'][-1], (self.metrics_recorder['lcoe_all'][-1] - self.metrics_recorder['lcoe_all'][0]) / self.metrics_recorder['lcoe_all'][0] * 100)
        #
        items = ['north','mid','south']
        for idx, zone in enumerate(items):
            if self.metrics_recorder['lcoe_' + zone][-1] != 0:
                title += " \n LCOE " + zone + " = %.2f $/MWh (%+.2f%%)" % (self.metrics_recorder['lcoe_' + zone][-1], (self.metrics_recorder['lcoe_' + zone][-1] - self.metrics_recorder['lcoe_' + zone][next(i for i, value in enumerate(self.metrics_recorder['lcoe_' + zone]) if value != 0)]) / (self.metrics_recorder['lcoe_' + zone][next(i for i, value in enumerate(self.metrics_recorder['lcoe_' + zone]) if value != 0)]) * 100)
            else:
                title += " \n LCOE " + zone + " = - $/MWh (-%)"
        self.ax.set_title(title,fontsize=9)
        
    def get_initial(self):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            pw = self.problem.get_vars_from_recorder()
            cost0 = self.problem.recorder[self.cost_key][0]
            # cost0 = pw['cost0']
            return pw['x0'], pw['y0'], cost0

    def compute(self, inputs, outputs):
        if self.by_pass is False:
            cost = inputs[self.cost_key][0]

            if (self.plot_improvements_only and
                'cost' in self.problem.recorder.driver_iteration_dict and
                len(self.problem.recorder['cost']) and
                    cost > self.problem.recorder['cost'].min()):
                return

            # find limits
            def get_lim(key):
                if (key in self.problem.design_vars and
                        isinstance(self.problem.design_vars[key], tuple) and
                        len(self.problem.design_vars[key]) == 4):
                    return np.min(self.problem.design_vars[key][1]), np.max(np.min(self.problem.design_vars[key][2]))
                else:
                    return min(inputs[key]), max(inputs[key])
            min_x, max_x = get_lim(topfarm.x_key)
            min_y, max_y = get_lim(topfarm.y_key)

            self.init_plot(np.array([[min_x, min_y], [max_x, max_y]]))
            
            self.plot_bathymetry()
            
            if len(self.b) > 0:
                self.plot_boundaries()
            
            self.plot_constraints()

            initial = self.get_initial()

            x = inputs[topfarm.x_key]
            y = inputs[topfarm.y_key]
            if initial is not None:
                x0, y0, cost0 = initial
                if self.plot_initial:
                    self.plot_initial2current(x0, y0, x, y)
                if self.memory > 0:
                    self.plot_history(x, y)
            else:
                cost0 = cost
            self.plot_cables(x,y)
            self.plot_current_position(x, y)
            if len(self.Xn) > 0:
                self.plot_nb_position(self.Xn,self.Yn)
            self.set_title(cost0, cost)
            self.ax.legend(loc='upper left',fontsize=7)
            
            self.ax.grid(alpha=0.6)
            self.ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Format axes to display in kilometers
            def meters_to_kilometers(x, _):
                return f'{x / 1000:.0f}'
            plt.gca().xaxis.set_major_formatter(FuncFormatter(meters_to_kilometers))
            plt.gca().yaxis.set_major_formatter(FuncFormatter(meters_to_kilometers))
            self.ax.set_ylabel('Northing [km]',fontsize=9)
            self.ax.set_xlabel('Easting [km]',fontsize=9)
            self.ax.set_xlim([534700,561000])
            self.ax.set_ylim([5813500,5852500])
            
            if self.counter == 0:
                plt.pause(1e-6)
            mypause(self.delay)
            
            self.counter += 1
            outputs['plot_counter'] = self.counter

            if self.save_plot_per_iteration:
                if not os.path.exists('Figures'):
                    os.makedirs('Figures')
                if self.opt_nr:
                    plt.savefig('Figures/iteration_z' + str(self.opt_nr) + '_%s.png' % self.counter)
                else:
                    plt.savefig('Figures/iteration_%s.png' % self.counter)