# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:01:35 2025

@author: Samuel Kainz
"""

import numpy as np

def create_recorder(Sequence):
    metrics_recorder = {
        "sequence": Sequence,
        "iteration": [],
        "opt_nr": [],
        "cur_zone": [],
        "neighbours": [],
        "aep": [],
        "mp_cost": [],
        "cable_cost": [],
        "lcoe": [],
        "cable_u": [],
        "cable_v": [],
        "cable_u_north": [],
        "cable_u_mid": [],
        "cable_u_south": [],
        "cable_v_north": [],
        "cable_v_mid": [],
        "cable_v_south": [],
        "cable_type": [],
        "cable_type_north": [],
        "cable_type_mid": [],
        "cable_type_south": [],
        "aep_north": [],
        "aep_mid": [],
        "aep_south": [],
        "aep_all": [],
        "mp_cost_north": [],
        "mp_cost_mid": [],
        "mp_cost_south": [],
        "mp_cost_all": [],
        "cable_cost_north": [],
        "cable_cost_mid": [],
        "cable_cost_south": [],
        "cable_cost_all": [],
        "lcoe_north": [],
        "lcoe_mid": [],
        "lcoe_south": [],
        "lcoe_all": [],
        "x_north": [],
        "y_north": [],
        "x_mid": [],
        "y_mid": [],
        "x_south": [],
        "y_south": [],
        "x_final": [],
        "y_final": [],
        "lcoe_final": [],
        "cable_solver": [],
        "time_limit": [],
        "mip_gap": [],
        "sgd_constraint_violation": [],
        "tur_dist_violation": [],
        "bound_violation": [],
        "general_settings": [],
        "current_settings": []
    }
    return metrics_recorder

def record_cable_metrics(metrics_recorder, wfn, curzone, nnb, nb, cable_solver, time_limit, mip_gap):
    cab_data = wfn.get_network()
    # get connection matrix
    u_fnt = []
    v_fnt = []
    fnT = wfn.G.graph.get('fnT', None)
    fnT_state = fnT is None
    for u, v in wfn.G.edges():
         u_fnt.append(u if fnT_state else fnT[u])
         v_fnt.append(v if fnT_state else fnT[v])

    # record
    metrics_recorder["cable_u_" + curzone].append(u_fnt)
    metrics_recorder["cable_v_" + curzone].append(v_fnt)
    metrics_recorder["cable_type_" + curzone].append(cab_data['cable'].tolist()) #([t[2]['cable'] for t in cab_data])
    metrics_recorder["cable_solver"].append(cable_solver)
    metrics_recorder["time_limit"].append(time_limit)
    metrics_recorder["mip_gap"].append(mip_gap)
    
    for zone in nnb:
        metrics_recorder["cable_u_" + zone].append([])
        metrics_recorder["cable_v_" + zone].append([])
        metrics_recorder["cable_type_" + zone].append([])

    for zone in nb:
        metrics_recorder["cable_u_" + zone].append(metrics_recorder['cable_u_' + zone][-1])
        metrics_recorder["cable_v_" + zone].append(metrics_recorder['cable_v_' + zone][-1])
        metrics_recorder["cable_type_" + zone].append(metrics_recorder['cable_type_' + zone][-1])

def record_main_metrics_multisub(metrics_recorder, opt_nr, aep, x, y, mp_cost, cable_costs, lcoe,
                               curzone, nb, nnb, wf, tur_nr, cable_cost_n, mp_cost_n,
                               npv, capex, LP, CRF, OpexAnnual, Sequence, **kwargs):
    metrics_recorder["iteration"].append(kwargs.get("iteration", len(metrics_recorder["iteration"]) + 1))
    metrics_recorder["opt_nr"].append(opt_nr)
    metrics_recorder["aep"].append(aep.isel(wt=slice(0,len(x))).sum().item())
    metrics_recorder["mp_cost"].append(float(sum(mp_cost)))
    metrics_recorder["cable_cost"].append(sum(cable_costs))
    metrics_recorder["lcoe"].append(float(lcoe))
    metrics_recorder["cur_zone"].append([curzone])
    
    indices = [np.arange(sum(tur_nr[:idx]), sum(tur_nr[:idx + 1])) for idx in range(len(wf))]
    for z, zone in enumerate(Sequence):
        metrics_recorder["neighbours"].append(nb)
        metrics_recorder["x_" + zone].append(x[indices[z]].flatten().tolist())
        metrics_recorder["y_" + zone].append(y[indices[z]].flatten().tolist())
        metrics_recorder["aep_" + zone].append(aep.isel(wt=slice(min(indices[z]),max(indices[z]+1))).sum().item())
        metrics_recorder["cable_cost_" + zone].append(cable_costs[z])
        metrics_recorder["mp_cost_" + zone].append(float(sum(mp_cost[indices[z]])))
        metrics_recorder["lcoe_" + zone].append(float(((capex*tur_nr[z] + sum(mp_cost[indices[z]]) + cable_costs[z] + LP*tur_nr[z]) * CRF + OpexAnnual*tur_nr[z]) / aep.isel(wt=slice(min(indices[z]),max(indices[z]+1))).sum().item()))
    metrics_recorder["aep_all"].append(np.sum(aep).item())
    metrics_recorder["lcoe_all"].append(float(lcoe))
    metrics_recorder["cable_cost_all"].append(sum(cable_costs))
    metrics_recorder["mp_cost_all"].append(float(sum(mp_cost)))

def record_main_metrics_singlesub(metrics_recorder, opt_nr, aep, x, y, mp_cost, cable_cost, lcoe,
                               curzone, nb, nnb, wf, tur_nr, cable_cost_n, mp_cost_n,
                               npv, capex, LP, CRF, OpexAnnual, **kwargs):
    metrics_recorder["iteration"].append(kwargs.get("iteration", len(metrics_recorder["iteration"]) + 1))
    metrics_recorder["opt_nr"].append(opt_nr)
    metrics_recorder["aep"].append(aep.isel(wt=slice(0,len(x))).sum().item())
    metrics_recorder["mp_cost"].append(sum(mp_cost))
    metrics_recorder["cable_cost"].append(cable_cost)
    metrics_recorder["lcoe"].append(lcoe)
    
    # performance of individual zones
    # current zone
    metrics_recorder["cur_zone"].append([curzone])
    metrics_recorder["neighbours"].append(nb)
    metrics_recorder["x_" + curzone].append(x.flatten().tolist())
    metrics_recorder["y_" + curzone].append(y.flatten().tolist())
    metrics_recorder["aep_" + curzone].append(aep.isel(wt=slice(0,tur_nr[wf[curzone]])).sum().item())
    metrics_recorder["cable_cost_" + curzone].append(cable_cost)
    metrics_recorder["mp_cost_" + curzone].append(sum(mp_cost))
    metrics_recorder["lcoe_" + curzone].append(lcoe)
    npv_all = [npv]

    # 0 for non-neighbours
    for n, zone in enumerate(nnb):
        metrics_recorder["x_" + zone].append([])
        metrics_recorder["y_" + zone].append([])
        metrics_recorder["aep_" + zone].append(0)
        metrics_recorder["cable_cost_" + zone].append(0)
        metrics_recorder["mp_cost_" + zone].append(0)
        metrics_recorder["lcoe_" + zone].append(0)

    # neighbours
    for n, zone in enumerate(nb):
        metrics_recorder["x_" + zone].append(metrics_recorder["x_" + zone][-1])
        metrics_recorder["y_" + zone].append(metrics_recorder["y_" + zone][-1])
        if n == 0:
            aep_n = aep.isel(wt=slice(tur_nr[wf[curzone]],tur_nr[wf[curzone]] + tur_nr[wf[zone]])).sum().item()
        elif n == 1:
            aep_n = aep.isel(wt=slice(tur_nr[wf[curzone]] + tur_nr[wf[nb[n-1]]],None)).sum().item()
        metrics_recorder["aep_" + zone].append(aep_n)
        metrics_recorder["cable_cost_" + zone].append(cable_cost_n[n])
        metrics_recorder["mp_cost_" + zone].append(mp_cost_n[n])
        npv_n = (capex*tur_nr[wf[zone]] + mp_cost_n[n] + cable_cost_n[n] + LP*tur_nr[wf[zone]]) * CRF + OpexAnnual*tur_nr[wf[zone]]
        metrics_recorder["lcoe_" + zone].append(npv_n / aep_n)
        npv_all.append(npv_n)
    
    # total
    metrics_recorder["aep_all"].append(np.sum(aep).item())
    metrics_recorder["cable_cost_all"].append(sum(cable_cost_n) + cable_cost)
    metrics_recorder["mp_cost_all"].append(sum(mp_cost_n) + sum(mp_cost))
    metrics_recorder["lcoe_all"].append(sum(npv_all)/np.sum(aep).item())
    
def record_results_constraints(metrics_recorder, recorder, state, cost, min_spacing_m):
    metrics_recorder["lcoe_final"].append([cost])
    metrics_recorder["x_final"].append(state['x'].tolist())
    metrics_recorder["y_final"].append(state['y'].tolist())
    metrics_recorder["sgd_constraint_violation"] += recorder['sgd_constraint'].tolist()
    
    # min spacing constraint
    dv = np.sqrt(recorder['wtSeparationSquared']) - min_spacing_m
    dv[dv > 0] = 0
    metrics_recorder["tur_dist_violation"] += dv.sum(axis=1).tolist()
    
    # boundary constraint
    bv = recorder['boundaryDistances']
    bv[bv > 0] = 0
    metrics_recorder["bound_violation"] += bv.sum(axis=1).tolist()