import math
import time

import astropy.cosmology
import numpy as np
from numpy.random import random

import funcs
from clusterbh import clusterBH
from funcs import MergerOutcome

DEBUG_MODE = False


def run_model(input_params, supernova_model="RAPID", rho_h_i=1e5):  # Initial density MSun/pc3
    tf0, Mcl_i, [Z_file, Z] = input_params
    tf0 *= 1e9
    try:
        ret_val = _run_model(input_params, supernova_model, rho_h_i)
        if DEBUG_MODE and len(ret_val) == 0:
            print(f"The cluster with mass = {Mcl_i} M_sun and metallicity = {Z} ({Z_file}) did not produce any merger")
        return ret_val
    except Exception as err:
        print(f"Error in model with", flush=True)
        print(f"\t Mass = {Mcl_i} M_sun", flush=True)
        print(f"\t Metallicity = {Z} ({Z_file})", flush=True)
        print(f"\t Formation redshift = {tf0 / 1e9} Gyr", flush=True)
        print(err, flush=True)
        raise err


def _run_model(input_params, supernova_model, rho_h_i):
    tf0, Mcl_i, [Z_file, Z] = input_params
    tf0 *= 1e9

    if DEBUG_MODE:
        print(f"Generating new model with")
        print(f"\t Mass = {Mcl_i:.2g} M_sun")
        print(f"\t Metallicity = {Z:.2e}")
        print(f"\t Formation redshift = {tf0 / 1e9 :.3g} Gyr")

    c = 1.e4  # sol

    Mcl0 = 1e8 if supernova_model == "RAPID" else (
        1e7 if supernova_model == "DELAY" else -1)  # Only clusters below this mass are included

    # Cluster evolution
    # Some other model parameters
    csi = 0.09  # Relaxation coefficient
    a1 = 1.47  # Coefficient relating f_bh to t_relax (see Antonini+Gieles)
    m_mean = 0.638  # Mean mass
    beta = 2.80e-3  # BH loss rate

    a_poly = 8.53174266519503e+16
    b_poly = -7177288764328684
    c_poly = 238181873067096.75
    d_poly = -3958278869787.77
    e_poly = 34364808989.07675
    f_poly = -145642408.80801892
    g_poly = 228858.39425097185
    h_poly = 54.322223269234705
    i_poly = -0.1954553717385152

    alpha_samp = a_poly * (Z ** 8) + b_poly * (Z ** 7) + c_poly * (Z ** 6) + d_poly * (Z ** 5) + e_poly * (
            Z ** 4) + f_poly * (Z ** 3) + g_poly * (Z ** 2) + h_poly * (Z ** 1) + i_poly

    bhout = []
    mbh0 = []
    vk0 = []
    with open(f"data/BHs/{supernova_model}/{Z_file}", "r") as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        for line in lines:
            mbh0_i, vk0_i = line.replace("\n", "").strip().split()
            mbh0.append(float(mbh0_i))
            vk0.append(float(vk0_i))

    v_esc_i = 50 * (Mcl_i / 1e5) ** (1 / 3) * (rho_h_i / 1e5) ** (1 / 6)

    # Build array with BH properties for retained BHs
    bhv = []
    mtot = 0
    j = 0
    ibh = len(mbh0)

    for j, (mbh0_i, vk0_i) in enumerate(zip(mbh0, vk0)):
        if not j < math.floor(ibh * Mcl_i / Mcl0):
            break
        if vk0_i < v_esc_i:
            mtot += mbh0_i
            spin0 = 0
            bhv.append([mbh0_i, spin0, 0, 1])

    bhv = np.array(bhv)
    nbh = len(bhv)

    if DEBUG_MODE:
        print(f"\t Number of BHs before natal kicks: {j}")
        print(f"\t Number of BHs after natal kicks: {nbh}")
    if nbh < 4:
        return bhout
    fbh0 = mtot / Mcl_i
    if DEBUG_MODE:
        print(f"\t Averaged BH mass after kicks: {mtot / nbh:.2f}")

    cbh = clusterBH(Mcl_i / m_mean, rho_h_i, kick=False, f0=fbh0, dtout=None, dense_output=True, tend=15e3)

    # compute time when balanced evolution starts (alla AG'19)
    tcc = cbh.tcc / 1e3

    #   start integration
    t = tcc * 1e9  # yr from this on
    N3ej = 0  # Ejected BHs (+ BHs destroyed in mergers)
    M3ej = 0  # Ejected BH mass

    if M3ej > 100:
        raise RuntimeError("Initial M3ej is too big")

    while t <= tf0:
        m_d, mbhmax, mbhmin, nbh_core, kmin, kmax = funcs.get_mbh_params(nbh, bhv, t)

        if nbh_core < 4:
            if len(bhv[~np.isnan(bhv[:, 0])]) < 4:
                return bhout  # exit if not BHs in the core
            else:
                t += np.min(bhv[:, 2][bhv[:, 2] > 0])
                continue

        if np.isnan(mbhmax) or np.isnan(mbhmin) or kmin == kmax or kmin is None or kmax is None:
            raise ValueError(f"Error in determining m range, found {mbhmax=} ({kmax=}) and {mbhmin=} ({kmin=})")

        # use a distribution for m1
        alpha_1 = 8 + 2 * alpha_samp
        m1_t = ((mbhmax ** (1 + alpha_1) - mbhmin ** (1 + alpha_1)) * random() + mbhmin ** (1 + alpha_1)) ** (
                1 / (1 + alpha_1))

        m_d[kmin] = np.nan
        k1 = np.nanargmin(np.abs(m_d - m1_t))
        m_d[kmin] = mbhmin

        m1, S1, t1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 2], bhv[k1, 3]

        if k1 == kmin or np.isnan(m1):
            raise RuntimeError(f"Error in {k1=}, {m1=} determination")

        # use a distribution for q
        alpha_2 = 3.5 + alpha_samp

        m_d[k1] = np.nan
        qmax = np.nanmax((m_d[m_d <= m1])) / m1
        qmin = mbhmin / m1

        q_t = ((qmax ** (1 + alpha_2) - qmin ** (1 + alpha_2)) * random() + qmin ** (1 + alpha_2)) ** (
                1 / (1 + alpha_2))
        k2 = np.nanargmin(np.abs(m_d - q_t * m1))
        m_d[k1] = m1

        m2, S2, t2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 2], bhv[k2, 3]

        if k2 == k1 or np.isnan(m2):  # just avoid to select the same black hole as 1
            raise RuntimeError(f"Error in {k2=}, {m2=} determination")

        if m2 > m1:  # Force that the primary is always the most massive
            k1, k2 = k2, k1
            m1, S1, t1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 2], bhv[k1, 3]
            m2, S2, t2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 2], bhv[k2, 3]

        #      compute spins and recoil kick
        v_kick, chi_f = funcs.recoil(m1, m2, S1, S2)

        #      evolved cluster properties
        if t / 1e6 > cbh.tend:
            return bhout
        rh = cbh.rh_interp(t / 1e6)
        Mcl = cbh.M_interp(t / 1e6)
        Mbh = max(cbh.Mbh_interp(t / 1e6), 0)

        lastBH = False
        if Mbh <= 0 or Mcl <= 0:  # check that cluster did not evaporate
            lastBH = True
        rho_h = 3 * Mcl / (8 * np.pi * rh ** 3)  # TODO: check these equations
        v_esc = 3.69e-3 * np.sqrt(Mcl / rh) * 30
        vd = v_esc / 4.77

        #      hard radius, and some other quantities
        mu = m1 * m2 / (m1 + m2)
        ah = 887.1278675 * mu / vd ** 2  # in AU if sigma in Km/s and M in solar masses
        Eh = (1 / 2) * (m1 + m2) * vd ** 2  # in Msun*(km/s)^2
        fbh = Mbh / (Mcl + Mbh)
        psi = 1. + a1 * fbh / 1e-2  # relaxation alla Antonini+Gieles 2019
        trh = 2.06e5 * np.sqrt((Mcl + Mbh) * rh ** 3) / (psi * m_mean)  # TODO: change to function
        Edot = 1.53e-7 * csi * (Mcl ** 2 / rh) / trh  # convert to M=M_sun, L=1AU, G=1

        # Do Montecarlo of three-body encounters
        # da = 1 / (dE + 1)
        a = ah
        Ebin = Eh
        merger_type = None

        while not lastBH:
            m_d, mbhmax, mbhmin, nbh_core, kmin, kmax = funcs.get_mbh_params(nbh, bhv, t)

            if nbh_core < 4:
                if len(bhv[~np.isnan(bhv[:, 0])]) < 4:
                    return bhout  # exit if not BHs in the core
                else:
                    t += np.min(bhv[:, 2][bhv[:, 2] > 0])
                    break

            if np.isnan(mbhmax) or np.isnan(mbhmin) or kmin == kmax or kmin is None or kmax is None:
                raise ValueError(f"Error in determining m range, found {mbhmax=} ({kmax=}) and {mbhmin=} ({kmin=})")

            if N3ej + 3 >= nbh:
                return bhout  # exit if runs out of BHs during binary hardening sequence

            # use a distribution for m3
            alpha_3 = alpha_samp + 0.5

            m3_t = ((mbhmax ** (1 + alpha_3) - mbhmin ** (1 + alpha_3)) * random() + mbhmin ** (1 + alpha_3)) ** (
                    1 / (1 + alpha_3))

            # Do this for performance reasons
            m_d[k1] = np.nan
            m_d[k2] = np.nan
            k3 = np.nanargmin(np.abs(m_d - m3_t))
            m3 = bhv[k3, 0]
            m_d[k1] = m1
            m_d[k2] = m2

            if k3 == k1 or k3 == k2 or np.isnan(m3) or bhv[k3, 2] > t:
                print(f"{k1=}, {k2=}, {k3=}, {m_d=}, {m3_t=}", flush=True)
                raise ValueError(f"Error in {k3=}, {m3=} determination")

            dE = 0.2  # fractional energy change per interaction
            Nrs = 20  # number of resonant states per 2-1 interaction

            # Resonant encounters
            is_capture = False
            Rs = 4 * (m1 + m2) / c ** 2
            ell_cap = (Rs / a) ** (5 / 14)
            ell_b = 1e11  # FIXME
            for i in range(Nrs):
                e_b = np.sqrt(random())
                ell_b = np.sqrt(1 - e_b ** 2)
                if ell_b < ell_cap:
                    is_capture = True
                    merger_type = MergerOutcome.GWCapture
                    break
            if is_capture:
                break

            vbin = np.sqrt(dE * Ebin * (2 / (m1 + m2)) * (m3 / (m1 + m2 + m3)))  # recoil in km/s

            # Recalculate E and SMA after interaction
            Ebin = Ebin * (1 + dE)
            a = 887.1278675 * m1 * m2 / (2 * Ebin)

            # Ejection of interlopers
            q3 = m3 / (m1 + m2)

            v3 = vbin / q3
            if v3 > v_esc:
                N3ej += 1
                M3ej += m3
                bhv[k3, 0] = np.nan

            # Ejection of binaries
            if vbin > v_esc:
                merger_type = MergerOutcome.Ejected
                break

            # In-cluster inspirals
            ell_gw = 1.3 * ((m1 * m2) ** 2 * (m1 + m2) / (c ** 5 * Edot)) ** (1 / 7) * a ** (-5 / 7)
            if ell_b < ell_gw:
                merger_type = MergerOutcome.InClusterInspiral
                break

        M3ej += m1 + m2

        if nbh_core < 4:
            continue
        if M3ej > fbh0 * Mcl_i:
            lastBH = True

        if m1 == 0 or m2 == 0:
            raise ValueError("Error in sampling, BH masses should not be 0!")

        #      GW timescale
        R = (1 + 73 / 24 * e_b ** 2 + 37 / 96 * e_b ** 4)
        t_gw = 5 * c ** 5 * a ** 4 * (1 - e_b ** 2) ** (7 / 2) / (64 * m1 * m2 * (m1 + m2) * R) * 58 / 365

        if np.isnan(t_gw):
            raise ValueError(f"Error in {t_gw=}")

        #      hardening sequence timescale

        #     compute dynamical friction timescale
        t_sim = t
        if v_kick < v_esc and merger_type != MergerOutcome.Ejected:  # if the binary is retained then make new BH
            #    recompute hardening timescale if retained

            t = funcs.get_sync_time(cbh, fbh0, Mcl_i, M3ej - m1 - m2, t)

            rin = rh * np.sqrt((v_esc ** 2 / (v_esc ** 2 - v_kick ** 2)) ** 2 - 1)
            tfric = 7.6e8 * (rin / 1.) ** 2 * (vd / 200.) * (10. / (m1 + m2))

            N3ej += 1

            # set such that the tot BH mass at that time is the same as in the cluster model
            tform = t + tfric + t_gw
            bhv[k1, 0] = np.nan  # remove one
            bhv[k2, 0] = m1 + m2  # and make a new one
            bhv[k2, 1] = chi_f  # new spin
            bhv[k2, 2] = tform  # reinclude in core only after this time
            bhv[k2, 3] = max(bhv[k1, 3], bhv[k2, 3]) + 1  # increase the BH generation by one

        else:  # if the binary is ejected or the end product is ejected then remove members
            M3ej += m1 + m2
            if M3ej > fbh0 * Mcl_i:
                lastBH = True
                # return bhout

            t = funcs.get_sync_time(cbh, fbh0, Mcl_i, M3ej, t)

            N3ej += 2

            # set such that the tot BH mass at that time is the same as in the cluster model
            bhv[k1, 0] = np.nan  # remove one
            bhv[k2, 0] = np.nan  # remove two

        tmerge = t + t_gw  # time of merger

        if lastBH:
            merger_type = MergerOutcome.Ejected

        if merger_type is None:
            raise ValueError("Merger type has not been set!")

        bhout.append([tmerge,  # 0 merger time
                      t_sim,  # 1 simulation time
                      m1,  # 2 mass of component 1
                      m2,  # 3 mass of component 2
                      e_b,  # 4 eccentricity of binary just before GW radiation takes over
                      Mcl_i,  # 5 cluster mass
                      merger_type,  # 6 merger type
                      tf0,  # 7 look-back time of formation
                      rho_h,  # 8 half mass radius
                      v_kick,  # 9 recoil kick
                      v_esc,  # 10 escape velocity
                      chi_f,  # 11 chi_f final remnant spin
                      S1,  # 12 S1 spin of component 1
                      S2,  # 13 S2 spin of component 2
                      Z,  # 14 metallicity
                      a,  # 15 semimajor axis
                      round(max(gen1, gen2)),  # 16 generation of merger
                      Mbh,  #
                      # z,  #
                      ])

        if lastBH:
            return bhout
    return bhout


if __name__ == "__main__":
    DEBUG_MODE = True
    tf0_tst = astropy.cosmology.Planck18.lookback_time(np.array([3, ])).value[0]

    print(f"---------------- TEST RUN BEGIN ----------------")

    seeds = [1, 2, 3, 4, 5, 6, 7]
    for seed in seeds:
        np.random.seed(seed)
        tst_start_time = time.time()

        bhout_tst = run_model(
            [13, 1e6, ["mbh.1005", 3.9810717874921620E-004]])  # [tf0, Mcl_i, [Z_idx, Z]]

        tst_end_time = time.time()

        merge_subchann_tst = {}
        for merger_tst in bhout_tst:
            lbl_tst = merger_tst[6]
            if lbl_tst not in merge_subchann_tst:
                merge_subchann_tst[lbl_tst] = 0
            merge_subchann_tst[lbl_tst] += 1
        print(f"Number of mergers:")
        for lbl_tst, merge_count_tst in merge_subchann_tst.items():
            print(f"\t{lbl_tst}: {merge_count_tst}")

        SAVE_TEST_OUTPUT = False
        if SAVE_TEST_OUTPUT:
            with open(f"temp/test_{seed}.txt", "w") as f:
                for bhout_tst_i in bhout_tst:
                    f.write(" ".join([str(el) for el in bhout_tst_i]) + "\n")

    print(f"\n Test run time: {tst_end_time - tst_start_time:.1f} s")
    print(f"---------------- TEST RUN END ----------------")
