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
        "lcoe_final": []
    }
    return metrics_recorder

def record_cable_metrics_singlesub(metrics_recorder, wfn, curzone, nnb, nb):
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

    for zone in nnb:
        metrics_recorder["cable_u_" + zone].append([])
        metrics_recorder["cable_v_" + zone].append([])
        metrics_recorder["cable_type_" + zone].append([])

    for zone in nb:
        metrics_recorder["cable_u_" + zone].append(metrics_recorder['cable_u_' + zone][-1])
        metrics_recorder["cable_v_" + zone].append(metrics_recorder['cable_v_' + zone][-1])
        metrics_recorder["cable_type_" + zone].append(metrics_recorder['cable_type_' + zone][-1])
        
def record_cable_metrics_multisub(metrics_recorder, Sequence, cab_data1, cab_data2, cab_data3, Sx, tur_nr):
    metrics_recorder["cable_u_" + Sequence[0]].append(cab_data1['u'].tolist())
    metrics_recorder["cable_v_" + Sequence[0]].append(cab_data1['v'].tolist())
    metrics_recorder["cable_u_" + Sequence[1]].append(cab_data2['u'].tolist())
    metrics_recorder["cable_v_" + Sequence[1]].append(cab_data2['v'].tolist())
    metrics_recorder["cable_u_" + Sequence[2]].append(cab_data3['u'].tolist())
    metrics_recorder["cable_v_" + Sequence[2]].append(cab_data3['v'].tolist())
    metrics_recorder["cable_type_" + Sequence[0]].append(cab_data1['cable'].tolist())
    metrics_recorder["cable_type_" + Sequence[1]].append(cab_data2['cable'].tolist())
    metrics_recorder["cable_type_" + Sequence[2]].append(cab_data3['cable'].tolist())
    #
    metrics_recorder["cable_u"].append([x+len(Sx)-1 if x != 1 else x for x in cab_data1['u'].tolist()] + [x+len(Sx)-1+tur_nr[0] if x != 1 else x+1 for x in cab_data2['u'].tolist()] + [x+len(Sx)-1+sum(tur_nr[0:2]) if x != 1 else x+2 for x in cab_data3['u'].tolist()])
    metrics_recorder["cable_v"].append([y+len(Sx)-1 if y != 1 else y for y in cab_data1['v'].tolist()] + [y+len(Sx)-1+tur_nr[0] if y != 1 else y+1 for y in cab_data2['v'].tolist()] + [y+len(Sx)-1+sum(tur_nr[0:2]) if y != 1 else y+2 for y in cab_data3['v'].tolist()])
    metrics_recorder["cable_type"].append(cab_data1['cable'].tolist() + cab_data2['cable'].tolist() + cab_data3['cable'].tolist())

def record_main_metrics_multisub(metrics_recorder, opt_nr, aep, x, y, mp_cost, cable_cost, lcoe,
                               tur_nr, G1, G2, G3, capex, LP, CRF, OpexAnnual, **kwargs):

    metrics_recorder["iteration"].append(kwargs.get("iteration", len(metrics_recorder["iteration"]) + 1))
    metrics_recorder["opt_nr"].append(opt_nr)
    metrics_recorder["aep"].append(aep.isel(wt=slice(0,len(x))).sum().item())
    metrics_recorder["mp_cost"].append(sum(mp_cost))
    metrics_recorder["cable_cost"].append(cable_cost)
    metrics_recorder["lcoe"].append(lcoe)
    # store the values of the individual zones
    metrics_recorder["x_north"].append(x[:tur_nr[0]].flatten().tolist())
    metrics_recorder["y_north"].append(y[:tur_nr[0]].flatten().tolist())
    metrics_recorder["x_mid"].append(x[tur_nr[0]:sum(tur_nr[0:2])].flatten().tolist())
    metrics_recorder["y_mid"].append(y[tur_nr[0]:sum(tur_nr[0:2])].flatten().tolist())
    metrics_recorder["x_south"].append(x[sum(tur_nr[0:2]):].flatten().tolist())
    metrics_recorder["y_south"].append(y[sum(tur_nr[0:2]):].flatten().tolist())
    mp_cost1 = np.sum(mp_cost[:tur_nr[0]])
    mp_cost2 = np.sum(mp_cost[tur_nr[0]:sum(tur_nr[0:2])])
    mp_cost3 = np.sum(mp_cost[sum(tur_nr[0:2]):])
    cable_cost1 = G1.cost
    cable_cost2 = G2.cost 
    cable_cost3 = G3.cost
    npv1 = (capex*tur_nr[0] + mp_cost1 + cable_cost1 + LP*tur_nr[0]) * CRF + OpexAnnual*tur_nr[0]
    npv2 = (capex*tur_nr[1] + mp_cost2 + cable_cost2 + LP*tur_nr[1]) * CRF + OpexAnnual*tur_nr[1]
    npv3 = (capex*tur_nr[2] + mp_cost3 + cable_cost3 + LP*tur_nr[2]) * CRF + OpexAnnual*tur_nr[2]
    aep1 = aep.isel(wt=slice(0,tur_nr[0])).sum().item()
    aep2 = aep.isel(wt=slice(tur_nr[0],sum(tur_nr[0:2]))).sum().item()
    aep3 = aep.isel(wt=slice(sum(tur_nr[0:2]),None)).sum().item()
    lcoe1 = npv1 / aep1
    lcoe2 = npv2 / aep2
    lcoe3 = npv3 / aep3
    metrics_recorder["aep_north"].append(aep1)
    metrics_recorder["aep_mid"].append(aep2)
    metrics_recorder["aep_south"].append(aep3)
    metrics_recorder["aep_all"].append(np.sum(aep).item())
    metrics_recorder["cable_cost_north"].append(cable_cost1)
    metrics_recorder["cable_cost_mid"].append(cable_cost2)
    metrics_recorder["cable_cost_south"].append(cable_cost3)
    metrics_recorder["cable_cost_all"].append(cable_cost)
    metrics_recorder["mp_cost_north"].append(mp_cost1)
    metrics_recorder["mp_cost_mid"].append(mp_cost2)
    metrics_recorder["mp_cost_south"].append(mp_cost3)
    metrics_recorder["mp_cost_all"].append(sum(mp_cost))
    metrics_recorder["lcoe_north"].append(lcoe1)
    metrics_recorder["lcoe_mid"].append(lcoe2)
    metrics_recorder["lcoe_south"].append(lcoe3)
    metrics_recorder["lcoe_all"].append(lcoe)


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