import sys

import numpy as np
from numpy.random import random
from scipy.optimize import bisect


class MergerOutcome:
    InClusterInspiral = "incluster_inspiral"
    GWCapture = "gw_capture"
    Ejected = "ejected"


def get_sync_time(cbh, fbh0, Mcl_i, M3ej, t):
    if fbh0 * Mcl_i - M3ej >= cbh.Mbh_interp(t / 1e6):
        return t
    elif fbh0 * Mcl_i - M3ej >= cbh.Mbh0:
        return t
    elif fbh0 * Mcl_i - M3ej <= cbh.Mbh_interp(cbh.tend):
        return cbh.tend * 1e6
    else:
        t_bisect = bisect(lambda t_param: cbh.Mbh_interp(t_param / 1e6) - (fbh0 * Mcl_i - M3ej),
                          1, cbh.tend * 1e6 - 1,
                          xtol=1e2)
        return max(t, t_bisect)


def get_mbh_params(nbh, bhv, t):
    m_d = np.where(bhv[:, 2] <= t, bhv[:, 0], np.zeros_like(bhv[:, 0]) * np.nan)
    nbh_core = len(m_d[~np.isnan(m_d)])
    kmax = len(m_d) - np.nanargmax(np.flip(m_d)) - 1
    # The numpy flip is only needed when there are multiple
    # BHs with the same mass, otherwise we would have kmin == kmax
    kmin = np.nanargmin(m_d)
    mbhmax = m_d[kmax]
    mbhmin = m_d[kmin]

    if m_d[kmax] != m_d[np.nanargmax(m_d)]:
        raise RuntimeError(f"Error in determining {kmax=}")

    return m_d, mbhmax, mbhmin, nbh_core, kmin, kmax


def tbalanced(Mbh, Mcl, rh, m_mean):
    N_rh = 3.21
    coulomb_log = 10
    G = 4490  # pc^3 M_sun^-1 Gyr^-1
    a1 = 1.47

    fbh = Mbh / Mcl
    psi = 1. + a1 * fbh / 0.01
    trh0 = 0.138 * np.sqrt(Mcl * rh ** 3 / G) * (1 / (psi * m_mean * coulomb_log))
    return N_rh * trh0  # Gyr


def dotp(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def recoil(m1, m2, S1, S2):
    q = m2 / m1  # q<0
    eta = q / (1 + q) ** 2

    theta = np.arccos(-1. + random() * (1. + 1.))
    phi = 0. + random() * (2. * np.pi + 0.0)
    chi1 = np.array([S1 * np.cos(theta), S1 * np.sin(theta) * np.sin(phi), S1 * np.sin(theta) * np.cos(phi)])

    theta = np.arccos(-1. + random() * (1. + 1.))
    phi = 0. + random() * (2. * np.pi + 0.0)
    chi2 = np.array([S2 * np.cos(theta), S2 * np.sin(theta) * np.sin(phi), S2 * np.sin(theta) * np.cos(phi)])

    theta = np.arccos(-1. + random() * (1. + 1.))
    phi = 0. + random() * (2. * np.pi + 0.0)
    j = np.array([np.cos(theta), np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)])

    chit = (q ** 2 * chi2 + chi1) / (1. + q) ** 2
    chit2 = chit[0] ** 2 + chit[1] ** 2 + chit[2] ** 2
    delta = (chi1 - q * chi2) / (1. + q)

    #     parallel components
    chip = dotp(chit, j)
    deltap = dotp(delta, j)

    #     perp components
    chi_cross = cross(chit, j)
    delta_cross = cross(delta, j)
    chiL = np.sqrt(chi_cross[0] ** 2 + chi_cross[1] ** 2 + chi_cross[2] ** 2)
    deltaL = np.sqrt(delta_cross[0] ** 2 + delta_cross[1] ** 2 + delta_cross[2] ** 2)

    #     recoil velocity
    A = 1.2e4
    B = -0.93
    H = 6.9e3
    V11 = 3677.76
    VA = 2481.21
    VB = 1792.45
    VC = 1506.52
    C2 = 1140.
    C3 = 2481.

    vm = A * eta ** 2 * (1. - q) / (1. + q) * (1. + B * eta)
    vsL = H * eta ** 2 * deltap
    Dphi = -1. + random() * (1. + 1.)
    vsp = 16. * eta ** 2 * (
            deltaL * (V11 + 2. * VA * chip + 4. * VB * chip ** 2 + 8. * VC * chip ** 3) + 2. * chiL * deltap * (
            C2 + 2. * C3 * chip)) * Dphi
    v_kick = np.sqrt(vm ** 2 + 2. * vm * vsL * np.cos(145. * np.pi / 180.) + vsL ** 2 + vsp ** 2)
    #     recoil velocity
    #     Compute new spin
    t0 = -2.8904
    t2 = -3.51712
    t3 = 2.5763
    s4 = -0.1229
    s5 = 0.4537
    ell = 2. * np.sqrt(3.) + t2 * eta + t3 * eta ** 2 + s4 * (1. + q) ** 4 / (1. + q ** 2) ** 2 * chit2 + (
            s5 * eta + t0 + 2.) * (1. + q) ** 2 / (1. + q ** 2) * chip

    chiv = chit + q / (1 + q) ** 2 * ell * j
    chi_f = min(1., np.sqrt(chiv[0] ** 2 + chiv[1] ** 2 + chiv[2] ** 2))
    return v_kick, chi_f


def evolve_eccentricity(a0, e0, m1, m2, f=10):
    # f in Hz
    if e0 == 0:
        return 0
    elif e0 == 1:
        return 1
    G = 3.964e-14  # In AU^3 M_sun^-1 s^-2

    try:
        return bisect(lambda e: -((e0 / e) ** 0.631578947368421 * (
                (1 + (121 * e0 ** 2) / 304) / (1 + (121 * e ** 2) / 304)) ** 0.3784254023488473) + (
                                        a0 * (1 - e0 ** 2) * (f * np.pi) ** (2 / 3)) / (
                                        (1 + e) ** 0.7969333333333333 * (G * (m1 + m2)) ** (1 / 3)),
                      1e-300, 1 - 1e-300)
    except Exception as e:
        print(e, file=sys.stderr)
        return np.nan
