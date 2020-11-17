#!/usr/bin/env python
"""
First solve the stationary electric conduction problem. Then use its
results to solve the evolutionary heat conduction problem.

Run this example as on a command line::

    $ python <path_to_this_file>/thermal_electric.py
"""
from __future__ import absolute_import
import sys
from math import log

import numpy as np
from sfepy.base.base import output

import params2
from var_TPBR import i_c

sys.path.append('.')
import os
from var_TPBL import i_a
import sfepy.discrete.fem.periodic as per
from params import *

filename_mesh = 'mesh/3dcell.mesh'
# Time stepping for the heat conduction problem.
t0 = 0.0
t1 = 0.5
n_step = 11

# Material parameters.
specific_heat = 1.2

##########

cwd = os.path.split(os.path.join(os.getcwd(), __file__))[0]

options = {
    'absolute_mesh_path': True,
    'output_dir': os.path.join(cwd, 'output')
}

regions = {
    'Omega': 'all',
    'Omega_topchannel': 'cells of group 5',
    'Omega_top': 'cells of group 6',
    'Omega_middle': 'cells of group 7',
    'Omega_bottom': 'cells of group 8',
    'Omega_bottomchannel': 'cells of group 9',
    'Omega_toppart': 'r.Omega_top +c r.Omega_topchannel',
    'Omega_bottompart': 'r.Omega_bottom +c r.Omega_bottomchannel',
    'Omega_cell': 'r.Omega_top +c r.Omega_middle +c r.Omega_bottom',
    'topinlet': ('r.Omega_topchannel *v vertices in (z<0.01e-3)', 'facet'),
    'topoutlet': ('r.Omega_topchannel *v vertices in (z>9.99e-3)', 'facet',),
    'topconnector': ('vertices in (y>0.99e-4)&(y<1.01e-4)&((x<2.51e-4)|(x>7.49e-4))', 'facet'),
    'bottominlet': ('r.Omega_bottomchannel *v vertices in (z>9.99e-3)', 'facet'),
    'bottomoutlet': ('r.Omega_bottomchannel *v vertices in (z<0.01e-3)', 'facet'),
    'bottomconnector': ('vertices in (y>-2.01e-4)&(y<-1.99e-4)&((x<2.51e-4)|(x>7.49e-4))', 'facet'),
    'Omega_Surface_T': ('r.Omega_topchannel *v r.Omega_top', 'facet'),
    'Omega_Surface_T1': ('copy r.Omega_Surface_T', 'facet', 'Omega_topchannel'),
    'Omega_Surface_T2': ('copy r.Omega_Surface_T', 'facet', 'Omega_top'),
    'Omega_Surface_TS': ('r.Omega_top *v r.Omega_middle', 'facet'),
    'Omega_Surface_TS1': ('copy r.Omega_Surface_TS', 'facet', 'Omega_top'),
    'Omega_Surface_TS2': ('copy r.Omega_Surface_TS', 'facet', 'Omega_middle'),
    'Omega_Surface_BS': ('r.Omega_bottom *v r.Omega_middle', 'facet'),
    'Omega_Surface_BS1': ('copy r.Omega_Surface_BS', 'facet', 'Omega_middle'),
    'Omega_Surface_BS2': ('copy r.Omega_Surface_BS', 'facet', 'Omega_bottom'),
    'Omega_Surface_B': ('r.Omega_bottom *v r.Omega_bottomchannel', 'facet'),
    'Omega_Surface_B1': ('copy r.Omega_Surface_B', 'facet', 'Omega_bottom'),
    'Omega_Surface_B2': ('copy r.Omega_Surface_B', 'facet', 'Omega_bottomchannel'),
}


def get_is(ts, coors, mode=None, **kwargs):
    if mode == 'qp':
        nqp, dim = coors.shape
        phis1 = 0.5
        phil = 2e-3
        phis3 = 0.2
        eta1 = 0
        eta2 = 1
        a = i_a(eta1)
        c = i_c(eta2)
        out = {'i_a': a * np.ones((nqp, 1, 1), dtype=np.float64),
               'i_c': c * np.ones((nqp, 1, 1), dtype=np.float64),
               }
        return out


def get_tpb_an(ts, coors, problem, equations=None, mode='qp',bc=None, **kwargs):
    if mode == 'qp':
        # T-field values in quadrature points coordinates given by integral i
        # - they are the same as in `coors` argument.
        phis1_values = problem.evaluate('ev_surface_integrate.i.Omega_Surface_TS1(phis1)',
                                        mode='qp', verbose=False)
        phil_values = problem.evaluate('ev_surface_integrate.i.Omega_Surface_TS2(phil)',
                                       mode='qp', verbose=False)
        # val = phis1_values - phil_values
        val = phis1_values
        output('phi: min:', val.min(), 'max:', val.max())

        val.shape = (val.shape[0]*val.shape[1], 1, 1)
        val2=val[0:441,0,0]
        # val2.shape=(441,1)
        # return {'val': val2}
        return val2

def get_tpb_ca(ts, coors, problem, equations=None, mode='qp',bc=None, **kwargs):
    if mode == 'qp':
        # T-field values in quadrature points coordinates given by integral i
        # - they are the same as in `coors` argument.
        # phis3_values = problem.evaluate('ev_surface_integrate.i.Omega_Surface_BS2(phis3)',
        #                                 mode='qp', verbose=False)
        # phil_values = problem.evaluate('ev_surface_integrate.i.Omega_Surface_BS1(phil)',
        #                                mode='qp', verbose=False)
        phis3_values = problem.evaluate('ev_volume_integrate.i.Omega_bottom(phis3)',
                                        mode='qp', verbose=False)
        phil_values = problem.evaluate('ev_volume_integrate.i.Omega_middle(phil)',
                                       mode='qp', verbose=False)
        # val = phis3_values - phil_values
        val = phis3_values
        output('phi: min:', val.min(), 'max:', val.max())

        val.shape = (val.shape[0]*val.shape[1], 1, 1)
        val2=val[0:441,0,0]
        # val2.shape=(441,1)
        # return {'val': val2}
        return np.zeros(441)-0.3

def set_electric_bc(coor):
    y = coor[:, 1]
    ymin, ymax = y.min(), y.max()
    val = 2.0 * (((y - ymin) / (ymax - ymin)) - 0.5)
    return val


# def get_circle(coors, domain=None):
#     r = np.sqrt(coors[:,0]**2.0 + coors[:,1]**2.0)
#     return np.where(r < 0.2)[0]
def get_shift1(ts, coors, region):
    # val = 0.1 * coors[:, 0]
    val = np.empty_like(coors[:, 1])
    # val.fill(-params.E_eq_c-R_const*T/4.0/F_const*log(cH2_ref**2*cO2_ref/cH2O_ref**2))
    val.fill(-params2.Eeq_a)
    return val


def get_shift2(ts, coors, region):
    val = np.empty_like(coors[:, 1])
    val.fill(-params2.Eeq_c)

    return val


# def mat_fun(ts, coors, mode=None, **kwargs):
#     if mode == 'qp':
#         nqp, dim = coors.shape
#         alpha = np.zeros((nqp,1,1), dtype=np.float64)
#         alpha[0:nqp // 2,...] = alpha1
#         alpha[nqp // 2:,...] = alpha2
#         K = np.eye(dim, dtype=np.float64)
#         K2 = np.tile(K, (nqp,1,1))
#         out = {
#             'K' : K2,
#             'f_1': 2800.0 * np.ones((nqp,1,1), dtype=np.float64),
#             'f_2': -20.0 * np.ones((nqp,1,1), dtype=np.float64),
#             'G_alfa': G_bar * alpha,
#
#             }
#
#         return out
functions = {
    'get_tpb_an': (get_tpb_an,),
    'get_tpb_ca': (get_tpb_ca,),
    'get_is': (get_is,),
    'get_shift1': (get_shift1,),
    'get_shift2': (get_shift2,),
    'match_y_line': (per.match_y_line,),
    'match_x_line': (per.match_x_line,),
    # 'set_electric_bc' : (lambda ts, coor, bc, problem, **kwargs:
    #                      set_electric_bc(coor),),
}

materials = {
    'is': 'get_is',
    'm': ({
              'thermal_conductivity': 25.0,
              'electric_conductivity': params2.kseff_a,  # electric_conductivity000
              'electroyle_conductivity': params2.kl,  # electroyle_conductivity000
              # 'D' : D_agg,
              'D': 0.001,
              'DH': 0.1,
              'k': 1,
              'v': [[0.0], [0.0], [0.0]],
              'nFA_h2': 1.0 / F_const / 2 * 1e3,
              'nFA_h2o': -1.0 / F_const / 2 * 1e3,
              'nFA_o2': 1.0 / F_const / 4 * 1e2,
              'h': 50,
              'T0': 1073.15,
              'p1_k': 0.089 * eps_mic / kappa_p,
              'p2_k': 1.293 * eps_mac / kappa_p,
              'i0': 2.0,
          },),
}

# The fields use the same approximation, so a single field could be used
# instead.
fields = {
    'temperature': ('real', 1, 'Omega', 1),
    'potential_l': ('real', 1, 'Omega_cell', 1),
    'potential_s1': ('real', 1, 'Omega_top', 1),
    'potential_s2': ('real', 1, 'Omega_bottom', 1),
    'concentration1': ('real', 1, 'Omega_top', 1),
    'concentration2': ('real', 1, 'Omega_bottom', 1),
    'pressure1': ('real', 1, 'Omega_topchannel', 1),
    'pressure2': ('real', 1, 'Omega_bottomchannel', 1),
    'currents': ('real', 3, 'Omega_cell', 1),
    'currentl': ('real', 3, 'Omega_cell', 1)
}

variables = {
    'T': ('unknown field', 'temperature', 0, 1),
    's': ('test field', 'temperature', 'T'),
    'phil': ('unknown field', 'potential_l', 1),
    'psil': ('test field', 'potential_l', 'phil'),
    'phis1': ('unknown field', 'potential_s1', 2),
    'psis1': ('test field', 'potential_s1', 'phis1'),
    'phis3': ('unknown field', 'potential_s2', 3),
    'psis3': ('test field', 'potential_s2', 'phis3'),
    'is': ('unknown field', 'currents', 4),
    'iss': ('test field', 'currents', 'is'),
    'il': ('unknown field', 'currentl', 5),
    'ils': ('test field', 'currentl', 'il'),
    'c_h2a': ('unknown field', 'concentration1', 6, 1),
    'c_h2as': ('test field', 'concentration1', 'c_h2a'),
    'c_h2oa': ('unknown field', 'concentration1', 7, 1),
    'c_h2oas': ('test field', 'concentration1', 'c_h2oa'),
    'c_o2c': ('unknown field', 'concentration2', 8, 1),
    'c_o2cs': ('test field', 'concentration2', 'c_o2c'),
    'c_n2c': ('unknown field', 'concentration2', 9, 1),
    'c_n2cs': ('test field', 'concentration2', 'c_n2c'),
    'p1': ('unknown field', 'pressure1', 10),
    'q1': ('test field', 'pressure1', 'p1'),
    'p2': ('unknown field', 'pressure2', 11),
    'q2': ('test field', 'pressure2', 'p2'),
}

# ics = {
#     'ic' : ('Omega', {'T.0' : 0.0}),
# }

ebcs = {
    'Topconnector': ('topconnector', {'phis1.0': 0.0}),  # V_cell000
    'Bottomconnector': ('bottomconnector', {'phis3.0': 0.7}),  # V_cell000
    'OOmega_Surface_T2': ('Omega_Surface_T2', {'c_h2a.0': 0.95, 'c_h2oa.0': 0.05}),
    'OOmega_Surface_B1': ('Omega_Surface_B1', {'c_o2c.0': 0.21, 'c_n2c.0': 0.78}),
    'Topinlet': ('topinlet', {'p1.0': 1.1}),  # V_cell000
    'Topoutlet': ('topoutlet', {'p1.0': 1.0}),
    'Bottomoutlet': ('bottomoutlet', {'p2.0': 1.0}),  # V_cell000
    'Bottominlet': ('bottominlet', {'p2.0': 1.1}),
    'Topmiddle': ('Omega_Surface_TS2', {'phil.0': 'get_tpb_an'}),
    'Bottommiddle': ('Omega_Surface_BS1', {'phil.0': 'get_tpb_ca'}),
}

lcbcs = {
    'shifted1' : (('Omega_Surface_TS2', 'Omega_Surface_TS1'),
                  {'phil.all' : 'phis1.all'},
                  'match_y_line', 'shifted_periodic',
                  'get_shift1'),
    'shifted2' : (('Omega_Surface_BS1', 'Omega_Surface_BS2'),
                  {'phil.all' : 'phis3.all'},
                  'match_y_line', 'shifted_periodic',
                  'get_shift2'),
}
integrals = {
    'i': 3
}

equations = {
    '2': """%.12e *dw_volume_dot.i.Omega( s, dT/dt )
            + dw_laplace.i.Omega( m.thermal_conductivity, s, T )
            =-dw_electric_source.i.Omega_top( m.DH,s, phis1 )-dw_electric_source.i.Omega_middle( m.DH,
                                           s, phil )-dw_electric_source.i.Omega_bottom( m.DH,
                                           s, phis3 )+dw_bc_newton.2.Omega_Surface_T2(m.h,m.T0,s,T)+dw_bc_newton.2.Omega_Surface_B1(m.h,m.T0,s,T)""" % specific_heat,
    'eq1': """dw_laplace.i.Omega_cell( m.electroyle_conductivity, psil, phil ) = 0""",
    'eq2': """dw_laplace.i.Omega_top( m.electric_conductivity, psis1, phis1 ) = 0""",
    'eq3': """dw_laplace.i.Omega_bottom( m.electric_conductivity, psis3, phis3 ) = 0""",
    # 'butler-volmer1':"""dw_volume_dot.3.Omega_cell(is1,i1)=dw_volume_integrate.3.Omega_cell(m.i0,is1)""",
    # 'butler-volmer1': """dw_volume_dot.3.Omega_cell(is1,i1)+dw_surface_dot.2.Omega_Surface_TS1(is1,i1)
    #                 =dw_surface_integrate.2.Omega_Surface_TS1(m.i0,is1)""",
    # 'butler-volmer1':"""dw_volume_dot.3.Omega_cell(ils,il)
    #         =dw_stokes.3.Omega_cell(m.electroyle_conductivity,ils,phil)""",
    # 'butler-volmer2':"""dw_surface_dot.2.Omega_Surface_BS2(is2,i2)=0""",
    'advection-diffusion-anode-h2': """dw_volume_dot.i.Omega_top(c_h2as, dc_h2a/dt)
     + dw_advect_div_free.i.Omega_top(m.v, c_h2as, c_h2a)
     + dw_laplace.i.Omega_top(m.D, c_h2as, c_h2a)
     = dw_laplace.i.Omega_top(m.nFA_h2,c_h2as,phis1)""",
    'advection-diffusion-anode-h2o': """dw_volume_dot.i.Omega_top(c_h2oas, dc_h2oa/dt)
     + dw_advect_div_free.i.Omega_top(m.v, c_h2oas, c_h2oa)
     + dw_laplace.i.Omega_top(m.D, c_h2oas, c_h2oa)
     = dw_laplace.i.Omega_top(m.nFA_h2o,c_h2oas,phis1)""",
    'advection-diffusion-cathode-o2': """dw_volume_dot.i.Omega_bottom(c_o2cs, dc_o2c/dt)
     + dw_advect_div_free.i.Omega_bottom(m.v, c_o2cs, c_o2c)
     + dw_laplace.i.Omega_bottom(m.D, c_o2cs, c_o2c)
     = dw_laplace.i.Omega_bottom(m.nFA_o2,c_o2cs,phis3)""",
    'advection-diffusion-cathode-n2': """dw_volume_dot.i.Omega_bottom(c_n2cs, dc_n2c/dt)
     + dw_advect_div_free.i.Omega_bottom(m.v, c_n2cs, c_n2c)
     + dw_laplace.i.Omega_bottom(m.D, c_n2cs, c_n2c)
     = 0""",
    'darcy1': """dw_laplace.i.Omega_topchannel(m.p1_k, q1, p1)
     = 0""",
    'darcy2': """dw_laplace.i.Omega_bottomchannel(m.p2_k, q2, p2)
     = 0""",
}

solvers = {
    'ls': ('ls.scipy_direct', {}),
    'newton': ('nls.newton', {
        'i_max': 1,
        'eps_a': 1e-10,
        'problem': 'nonlinear',
    }),
    # 'ts' : ('ts.simple', {
    #     't0'     : t0,
    #     't1'     : t1,
    #     'dt'     : None,
    #     'n_step' : n_step, # has precedence over dt!
    #     'verbose' : 1,
    # }),
}
