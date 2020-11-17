from __future__ import absolute_import
import sys
from numpy import exp

from sfepy.base.base import output

sys.path.append('.')
import os
import numpy as np

filename_mesh = 'mesh/1dtest.mesh'
cwd = os.path.split(os.path.join(os.getcwd(), __file__))[0]
t0 = 0.0
t1 = 0.1
n_step = 25
options = {
    'absolute_mesh_path': True,
    'output_dir': os.path.join(cwd, 'output'),
    'nls': 'newton',
    'ls': 'ls',
    'ts': 'ts',
    # 'save_times': 'all',
}
regions = {
    'Omega': 'all',
    'Omega1': 'cells of group 1',
    'Omega2': 'cells of group 2',
    'Gamma': ('r.Omega1 *v r.Omega2', 'facet'),
    'Gamma1': ('copy r.Gamma', 'facet', 'Omega1'),
    'Gamma2': ('copy r.Gamma', 'facet', 'Omega2'),
    'Gamma_left': ('r.Omega1 *v vertices in (x<0.01)', 'facet'),
    'Gamma_right': ('r.Omega2 *v vertices in (x>1.98)', 'facet',),
}

F_const = 96485.33289  # "法拉第常数[C/mol]"
R_const = 8.31446261815324  # "气体常数[J/(mol·K)]"
T = 1073.13
Eeq = 1
ics = {
    'ic1': ('Omega1', {'phil.0': 0.0}),
    'ic2': ('Omega2', {'phis.0': 0.7}),
}

ilk_1 = np.empty([100,2, 1, 1])
ilk_1.fill(-0.3)
flk_1 = np.empty([100,2, 1, 1])
flk_1.fill(0.0)
def get_current_l(ts, coors, problem, equations=None, mode=None, **kwargs):
    if mode == 'qp':
        global ilk_1
        global flk_1
        ev = problem.evaluate
        currentl = ev('ev_diffusion_velocity.i.Omega2(m.electroyle_conductivity,phil)', mode='qp')
        # currents = ev('ev_diffusion_velocity.i.Omega2(m.electric_conductivity,phis)', mode='qp')
        phil_values = ev('ev_volume_integrate.i.Omega2(phil)',
                         mode='qp', verbose=False)
        phis_values = ev('ev_volume_integrate.i.Omega2(phis)',
                         mode='qp', verbose=False)
        eta = phis_values - phil_values - Eeq
        current = 1e2 * (exp(0.5 * F_const * eta / R_const / T) - exp(-0.5 * F_const * eta / R_const / T))
        # output('conductivity: min:', val.min(), 'max:', val.max())
        # current = -eta * 100
        # currentl[:,0,0,0] = currentl[:,0,0,0][::-1]
        # currentl[:, 1, 0, 0] = currentl[:, 1, 0, 0][::-1]
        # temp = np.array([currentl[:, 0, 0, 0][::-1], currentl[:, 1, 0, 0][::-1]])
        # temp.shape = (100, 2, 1, 1)
        # currentl = temp
        ik = -currentl
        fk = ik - current
        ikp1 = ik - (fk * (ik - ilk_1) / (fk - flk_1))
        # if np.sum((fk - flk_1)**2)>1e-6:
        #     ikp1 = ik - (fk * (ik - ilk_1) / (fk - flk_1))
        # else:
        #     ikp1 = ilk_1
        print(ikp1)
        # r = np.ones([100,2,1,1])*(-0.3)
        r = ikp1
        ilk_1 = ik
        flk_1 = fk
        r.shape = (r.shape[0] * r.shape[1], 1, 1)
        return {'val': r}

isk_1 = np.empty([100,2, 1, 1])
isk_1.fill(-0.3)
fsk_1 = np.empty([100,2, 1, 1])
fsk_1.fill(0.0)
def get_current_s(ts, coors, problem, equations=None, mode=None, **kwargs):
    if mode == 'qp':
        global isk_1
        global fsk_1
        ev = problem.evaluate
        # currentl = ev('ev_diffusion_velocity.i.Omega1(m.electroyle_conductivity,phil)', mode='qp')
        currents = ev('ev_diffusion_velocity.i.Omega2(m.electric_conductivity,phis)', mode='qp')
        phil_values = ev('ev_volume_integrate.i.Omega2(phil)',
                         mode='qp', verbose=False)
        phis_values = ev('ev_volume_integrate.i.Omega2(phis)',
                         mode='qp', verbose=False)
        eta = phis_values[0, 0, 0, 0] - phil_values[0, 0, 0, 0] - Eeq
        current = 1e2 * (exp(0.5 * F_const * eta / R_const / T) - exp(-0.5 * F_const * eta / R_const / T))
        # output('conductivity: min:', val.min(), 'max:', val.max())
        # current = -eta * 100
        # currents[:, 0, 0, 0] = currents[:, 0, 0, 0][::-1]
        # currents[:, 1, 0, 0] = currents[:, 1, 0, 0][::-1]
        # temp = np.array([currents[:, 0, 0, 0][::-1],currents[:, 1, 0, 0][::-1]])
        # temp.shape=(100,2,1,1)
        # currents = temp
        ik = -currents
        fk = ik - current
        ikp1 = ik - (fk * (ik - isk_1) / (fk - fsk_1))
        # if np.sum((fk - fsk_1) ** 2) > 1e-6:
        #     ikp1 = ik - (fk * (ik - isk_1) / (fk - fsk_1))
        # else:
        #     ikp1 = isk_1
        print(ikp1)
        # r = np.ones([100,2,1,1])*(-0.3)
        r = ikp1
        isk_1 = ik
        fsk_1 = fk
        r.shape = (r.shape[0] * r.shape[1], 1, 1)
        return {'val': r}
# def get_conductivity(ts, coors, problem, equations=None, mode=None, **kwargs):
#     """
#     Calculates the conductivity as 2+10*T and returns it.
#     This relation results in larger T gradients where T is small.
#     """
#     if mode == 'qp':
#         # T-field values in quadrature points coordinates given by integral i
#         # - they are the same as in `coors` argument.
#         T_values = problem.evaluate('ev_volume_integrate.i.Omega1(phil)',
#                                     mode='qp', verbose=False)
#         val = 2 + 10 * (T_values + 2)
#
#         output('conductivity: min:', val.min(), 'max:', val.max())
#
#         val.shape = (val.shape[0] * val.shape[1], 1, 1)
#         return {'val' : val}

functions = {
    'get_current_l': (get_current_l,),
    'get_current_s': (get_current_s,),
    # 'get_conductivity': (get_conductivity,)
}

materials = {
    'm': ({
              'electroyle_conductivity': 1.0,
              'electric_conductivity': 10.0,
          },),
    # 'coef' : 'get_conductivity',
    'il0': 'get_current_l',
    'is0': 'get_current_s',
    # 'il0': {'val': -0.3},
    # 'is0': {'val': -0.3},
}
fields = {
    'potential_l': ('real', 1, 'Omega', 1),
    'potential_s': ('real', 1, 'Omega2', 1),
    'current_l': ('real', 1, 'Omega', 1),
    'current_s': ('real', 1, 'Omega2', 1),
}
variables = {
    'phil': ('unknown field', 'potential_l', 1),
    'psil': ('test field', 'potential_l', 'phil'),
    'phis': ('unknown field', 'potential_s', 2),
    'psis': ('test field', 'potential_s', 'phis'),
    'il': ('unknown field', 'potential_l', 3),
    'ils': ('test field', 'potential_l', 'il'),
    'is': ('unknown field', 'potential_s', 4),
    'iss': ('test field', 'potential_s', 'is'),
}
ebcs = {
    'left': ('Gamma_left', {'phil.0': 0.0}),  # V_cell000
    # 'middle1': ('Gamma1', {'phil.0': -0.3}),
    # 'middle2': ('Gamma2', {'phis.0': 0.8}),
    # 'middle1': ('Gamma1', {'il.0': -0.3}),
    # 'middle2': ('Gamma2', {'is.0': 0.3}),
    'right': ('Gamma_right', {'phis.0': 0.7}),  # V_cell000
}
integrals = {
    'i': 2
}
equations = {
    'eq1': """dw_laplace.i.Omega( m.electroyle_conductivity, psil, phil )
           = dw_volume_integrate.i.Omega2(il0.val,psil)""",
    'eq2': """dw_laplace.i.Omega2( m.electric_conductivity, psis, phis )
           = -dw_volume_integrate.i.Omega2(is0.val,psis)""",
    # 'eq1': """dw_laplace.i.Omega1( m.electroyle_conductivity, psil, phil  )
    #        = dw_surface_dot.i.Gamma1(psil,il)""",
    # 'eq2': """dw_laplace.i.Omega2( m.electric_conductivity, psis, phis )
    #        = dw_surface_dot.i.Gamma2(psis,is)""",
    'eq3': """dw_volume_dot.i.Omega(ils, il )
               = dw_diffusion_coupling.i.Omega(m.electroyle_conductivity,phil,ils)""",
    'eq4': """dw_volume_dot.i.Omega2(iss, is )
               = -dw_diffusion_coupling.i.Omega2(m.electric_conductivity,phis,iss)""",
}
solvers = {
    # 'ls': ('ls.scipy_iterative', {}),
    'ls': ('ls.scipy_direct', {}),
    'newton': ('nls.newton', {
        'i_max': 10,
        'eps_a': 1e-10,
        'eps_r': 1.0,
        'problem': 'nonlinear',
    }),
    'ts': ('ts.simple', {
        't0': t0,
        't1': t1,
        'dt': None,
        'n_step': n_step,  # has precedence over dt!
        'quasistatic': True,
        'verbose': 1,
    }),
}
options.update({
    'post_process_hook': 'postproc',
})


# options = {
#     'post_process_hook': 'postproc',
# }
def postproc(out, pb, state, extend=False):
    from sfepy.base.base import Struct
    from sfepy.postprocess.probes_vtk import Probe
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # print("222")
    # ev = pb.evaluate
    # currentl = ev('ev_diffusion_velocity.i.Omega(m.electroyle_conductivity,phil)', mode='qp')
    # currents = ev('ev_diffusion_velocity.i.Omega2(m.electric_conductivity,phis)', mode='qp')
    # phil = ev('dw_volume_integrate.i.Omega(phil)', mode='qp')
    # phis = ev('dw_volume_integrate.i.Omega2(phis)', mode='qp')
    # out['currentl'] = Struct(name='output_data', mode='vertex',
    #                          data=currentl.reshape(currentl.shape[0] * currentl.shape[1], 1), dofs=None)
    # out['currents'] = Struct(name='output_data', mode='vertex',
    #                          data=currents.reshape(currentl.shape[0] * currentl.shape[1], 1), dofs=None)
    probe = Probe(out, pb.domain.mesh, probe_view=False)
    ps0 = [0.0, 0.0, 0.0]
    ps1 = [2.0, 0.0, 0.0]
    n_point = 20

    probes = []
    probes.append('line1')
    probe.add_line_probe('line1', ps0, ps1, n_point)
    fig = plt.figure()
    plt.clf()
    fig.subplots_adjust(hspace=0.4)
    plt.subplot(221)
    pars, vals = probe(probes[0], 'phil')
    plt.plot(pars * 2, vals[:], label=r'$u_{phil}$',
             lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('phil')
    plt.xlabel('x')
    plt.legend(loc='best', prop=fm.FontProperties(size=10))
    plt.subplot(222)
    pars, vals = probe(probes[0], 'phis')
    plt.plot(pars * 2, vals[:], label=r'$u_{phis}$',
             lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('phis')
    # plt.ylabel('phi')
    plt.xlabel('x')
    plt.legend(loc='best', prop=fm.FontProperties(size=10))
    plt.subplot(223)
    pars, vals = probe(probes[0], 'il')
    plt.plot(pars * 2, vals[:], label=r'$i_{l}$',
             lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('il')
    # plt.ylabel('phi')
    plt.xlabel('x')
    plt.legend(loc='best', prop=fm.FontProperties(size=10))
    plt.subplot(224)
    pars, vals = probe(probes[0], 'is')
    plt.plot(pars * 2, vals[:], label=r'$i_{s}$',
             lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('is')
    # plt.ylabel('phi')
    plt.xlabel('x')
    plt.legend(loc='best', prop=fm.FontProperties(size=10))
    plt.show()
    return out
