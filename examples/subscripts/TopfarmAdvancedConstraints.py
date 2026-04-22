# Should be directly imported from topfarm after update

import numpy as np
import topfarm
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components import Constraint, ConstraintComponent
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingComp
from topfarm.constraint_components.boundary import XYBoundaryConstraint, MultiPolygonBoundaryComp
import matplotlib.pyplot as plt

# region Custom Classes for This Example

class CorrectedSpacingComp(SpacingComp):
    def _compute(self, x, y):
        n_wt = len(x)
        if n_wt <= 1: return np.array([])
        dX, dY = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T for xy in [x, y]]
        dXY2 = dX**2 + dY**2
        return dXY2[np.triu_indices(n_wt, 1)]

    def compute(self, inputs, outputs):
        outputs[self.constraint_key] = self._compute(inputs[topfarm.x_key], inputs[topfarm.y_key])

    def plot(self, ax=None, x=None, y=None):
        from matplotlib.pyplot import Circle
        ax = ax or plt.gca()
        if x is None or y is None: return
        for x_i, y_i in zip(x, y):
            circle = Circle((x_i, y_i), self.min_spacing / 2, color='k', ls='--', fill=False)
            ax.add_artist(circle)


class SpacingCompWithAdditionalTurbines(CorrectedSpacingComp):
    def __init__(self, n_wt, min_spacing, add_wt_x, add_wt_y, const_id=None, units=None):
        self.add_wt_x, self.add_wt_y = np.asarray(add_wt_x), np.asarray(add_wt_y)
        self.n_add_wt = len(self.add_wt_x)
        super().__init__(n_wt, min_spacing, const_id, units)
        self.veclen = n_wt * (n_wt - 1) // 2 + n_wt * self.n_add_wt

    def setup(self):
        self.add_input(topfarm.x_key, np.zeros(self.n_wt), units=self.units)
        self.add_input(topfarm.y_key, np.zeros(self.n_wt), units=self.units)
        self.add_output(self.constraint_key, np.zeros(self.veclen))
        self.declare_partials(self.constraint_key, [topfarm.x_key, topfarm.y_key])

    def compute(self, inputs, outputs):
        x_des, y_des = inputs[topfarm.x_key], inputs[topfarm.y_key]
        dist_sq_dd = super()._compute(x_des, y_des)
        dx_da = x_des[:, np.newaxis] - self.add_wt_x
        dy_da = y_des[:, np.newaxis] - self.add_wt_y
        dist_sq_da = (dx_da**2 + dy_da**2).flatten()
        outputs[self.constraint_key] = np.concatenate([dist_sq_dd, dist_sq_da])

    def compute_partials(self, inputs, J):
        x, y = inputs[topfarm.x_key], inputs[topfarm.y_key]
        n_wt, n_dd = self.n_wt, self.n_wt * (self.n_wt - 1) // 2
        J_dd_x, J_dd_y = np.zeros((n_dd, n_wt)), np.zeros((n_dd, n_wt))
        if n_wt > 1:
            dS_dxij_dd, dS_dyij_dd = self._compute_partials(x, y)
            col_pairs = np.array([(i, j) for i in range(n_wt - 1) for j in range(i + 1, n_wt)])
            for i in range(n_dd):
                i_wt, j_wt = col_pairs[i]
                J_dd_x[i, i_wt], J_dd_x[i, j_wt] = dS_dxij_dd[i, 0], dS_dxij_dd[i, 1]
                J_dd_y[i, i_wt], J_dd_y[i, j_wt] = dS_dyij_dd[i, 0], dS_dyij_dd[i, 1]
        
        n_da = n_wt * self.n_add_wt
        J_da_x, J_da_y = np.zeros((n_da, n_wt)), np.zeros((n_da, n_wt))
        if self.n_add_wt > 0:
            dx_da = x[:, np.newaxis] - self.add_wt_x
            dy_da = y[:, np.newaxis] - self.add_wt_y
            grad_x_flat, grad_y_flat = (2 * dx_da).flatten(), (2 * dy_da).flatten()
            row_indices, col_indices = np.arange(n_da), np.repeat(np.arange(n_wt), self.n_add_wt)
            J_da_x[row_indices, col_indices], J_da_y[row_indices, col_indices] = grad_x_flat, grad_y_flat
        
        J[self.constraint_key, topfarm.x_key] = np.vstack([J_dd_x, J_da_x]).flatten()
        J[self.constraint_key, topfarm.y_key] = np.vstack([J_dd_y, J_da_y]).flatten()

class SpacingConstraintWithAdditionalTurbines(SpacingConstraint):
    def __init__(self, min_spacing, additional_turbines, units=None, name='spacing_comp_with_add_wt'):
        self.add_wt_x, self.add_wt_y = additional_turbines
        super().__init__(min_spacing, units=units, name=name)

    def _setup(self, problem):
        self.n_wt = problem.n_wt
        self.spacing_comp = SpacingCompWithAdditionalTurbines(
            self.n_wt, self.min_spacing, self.add_wt_x, self.add_wt_y, self.const_id, self.units
        )
        problem.model.constraint_group.add_subsystem(
            self.const_id, self.spacing_comp,
            promotes=[topfarm.x_key, topfarm.y_key, 'wtSeparationSquared']
        )
        
class CorrectedBoundaryComp(MultiPolygonBoundaryComp):
    def __init__(self, n_wt, zones, const_id=None, units=None, relaxation=False, method='nearest',
                 simplify_geometry=False):
        super().__init__(n_wt, zones, const_id, units, relaxation, method,
                     simplify_geometry)

    def plot(self, ax=None, x=None, y=None):
        super().plot(ax)

    def satisfy(self, state):
        return super().satisfy(state)

class CorrectedXYBoundaryConstraint(XYBoundaryConstraint):
    def get_comp(self, n_wt):
        if not hasattr(self, 'boundary_comp'):
            self.boundary_comp = CorrectedBoundaryComp(
                n_wt, self.zones, const_id=self.const_id, units=self.units, relaxation=self.relaxation
            )
        return self.boundary_comp

class AggregatedConstraintComp(CostModelComponent, ConstraintComponent):
    def __init__(self, problem, constraints, **kwargs):
        self.problem = problem
        self.constraints = constraints
        super().__init__(**kwargs)
    
    def plot(self, ax):
        x, y = self.problem[topfarm.x_key], self.problem[topfarm.y_key]
        for constraint in self.constraints:
            if hasattr(constraint, 'constraintComponent') and hasattr(constraint.constraintComponent, 'plot'):
                constraint.constraintComponent.plot(ax, x=x, y=y)

    def satisfy(self, state):
        pass

class DistanceConstraintAggregation(Constraint):
    def __init__(self, boundary_constraint, spacing_constraint, n_wt):
        self.boundary_constraint, self.spacing_constraint = boundary_constraint, spacing_constraint
        self.n_wt, self.const_id = n_wt, 'constraint_aggregation_comp'
        self.constraints = [self.spacing_constraint, self.boundary_constraint]

    def _constr_aggr_func(self, wtSeparationSquared, boundaryDistances, **kwargs):
        sep_con = wtSeparationSquared - self.spacing_constraint.min_spacing**2
        return np.sum(-sep_con[sep_con < 0]) + np.sum(boundaryDistances[boundaryDistances < 0]**2)

    @property
    def constraintComponent(self):
        return self.constraint_aggregation_comp

    def _setup(self, problem):
        for constraint in self.constraints:
            constraint._setup(problem)
        
        input_keys = [
            ('wtSeparationSquared', np.zeros(self.spacing_constraint.constraintComponent.veclen)),
            # ** FIX IS HERE: Remove .flatten() to match the (n_wt, n_vertices) source shape **
            ('boundaryDistances', self.boundary_constraint.constraintComponent.zeros)
        ]
        
        self.constraint_aggregation_comp = AggregatedConstraintComp(
            problem, self.constraints, input_keys=input_keys, n_wt=self.n_wt,
            cost_function=self._constr_aggr_func,
            objective=False, output_keys=[('sgd_constraint', 0.0)])
        
        problem.model.add_subsystem(self.const_id, self.constraint_aggregation_comp, promotes=['*'])
    
    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint('sgd_constraint', lower=0)

    def setup_as_penalty(self, problem):
        self._setup(problem)