from __future__ import division

import numpy
from pylab import pi, sqrt, log
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1


class clusterBH:
    def __init__(self, N, rhoh, **kwargs):
        self.G = 0.004499  # pc^3 /Msun /Myr^2

        # Cluster ICs
        self.m0 = 0.638  # For Kroupa (2001) IMF 0.1-100 Msun
        self.N = N
        self.M0 = self.m0 * N
        self.rh0 = (3 * self.M0 / (8 * pi * rhoh)) ** (1. / 3)
        self.fc = 1  # equation (50)
        self.rg = 8  # [kpc]

        # BH MF
        self.mlo = 3
        self.mup = 30
        self.alpha = 0.5

        # Model parameters
        self.zeta = 0.1
        self.a0 = 1  # fix zeroth order
        self.a2 = 0  # ignore 2nd order for now
        self.f0 = 0.06  # for "metal-poor" GCs
        self.kick = False
        self.fretm = 1
        self.tsev = 2.

        # Parameters that were fit to N-body
        self.ntrh = 3.21
        self.beta = 0.00280
        self.nu = 0.0823
        self.a1 = 1.47

        self.sigmans = 265  # km/s
        self.mns = 1.4  # Msun

        # Some integration params
        self.tend = 12e3
        self.dtout = 2  # Myr
        self.Mst_min = 100  # [Msun] stop criterion
        self.dense_output = False
        self.integration_method = "DOP853"

        self.Mbh_interp = None
        self.Mst_interp = None
        self.Mbh_interp = None
        self.rh_interp = None

        self.output = False
        self.outfile = "cluster.txt"

        # Mass loss mechanism
        self.tidal = True
        self.Rht = 0.125  # ratio of rh/rt to give correct Mdot [17/3/22]
        self.Vc = 142.  # [km/s] circular velocity of singular isothermal galaxy

        # Check input parameters
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.vesc0 = 50 * (self.M0 / 1e5) ** (1. / 3) * (rhoh / 1e5) ** (1. / 6)

        self.vesc0 *= self.fc

        if (self.kick):
            mb = (9 * pi / 2) ** (1. / 6) * self.sigmans * self.mns / self.vesc0
            self.mb = mb
            mu, ml, a, a2 = self.mup, self.mlo, self.alpha, self.alpha + 2

            qul, qub, qlb = mu / ml, mu / mb, ml / mb

            b = a2 / 3

            h1 = hyp2f1(1, b, b + 1, -qub ** 3)
            h2 = hyp2f1(1, b, b + 1, -qlb ** 3)

            if a != -2:
                self.fretm = 1 - (qul ** a2 * h1 - h2) / (qul ** a2 - 1)
            else:
                self.fretm = log((qub ** 3 + 1) / (qlb ** 3 + 1)) / log(qul ** 3)

        self.Mbh0 = self.fretm * self.f0 * self.M0

        self.trh0 = self._trh(self.M0, self.rh0, self.f0 * self.fretm)
        self.tcc = self.ntrh * self.trh0

        self.evolve()

    def _rt(self, M):
        O2 = (self.Vc * 1.023 / (self.rg * 1e3)) ** 2
        return (self.G * M / (2 * O2)) ** (1. / 3)

    def _psi(self, fbh):
        psi = self.a0 + self.a1 * abs(fbh) / 0.01 + self.a2 * (abs(fbh) / 0.01) ** 2
        return psi

    def _trh(self, M, rh, fbh):
        m = self.M0 / self.N  # changed to M0 to keep m constant [17/3/22]
        if M > 0 and rh > 0:
            return 0.138 * sqrt(M * rh ** 3 / self.G) / (m * self._psi(fbh) * 10)
        else:
            return 1e-99

    def find_mmax(self, Mbh):
        a2 = self.alpha + 2

        # Note that a warning for alpha = -2 is needed
        if (self.kick):
            def integr(mm, qmb, qlb):
                a2 = self.alpha + 2
                b = a2 / 3
                h1 = hyp2f1(1, b, b + 1, -qmb ** 3)
                h2 = hyp2f1(1, b, b + 1, -qlb ** 3)

                return mm ** a2 * (1 - h1) - self.mlo ** a2 * (1 - h2)

            # invert eq. 52 from AG20
            Np = 1000
            mmax_ = numpy.linspace(self.mlo, self.mup, Np)
            qml, qmb, qlb = mmax_ / self.mlo, mmax_ / self.mb, self.mlo / self.mb

            A = Mbh[0] / integr(self.mup, self.mup / self.mb, qlb)

            Mbh_ = A * integr(mmax_, qmb, qlb)
            mmax = numpy.interp(Mbh, Mbh_, mmax_)
        else:
            # eq 51 in AG20
            mmax = (Mbh / self.Mbh0 * (self.mup ** a2 - self.mlo ** a2) + self.mlo ** a2) ** (1. / a2)

        # TBD: Set to 0 when MBH = 0
        return mmax

    def odes(self, t, y):
        Mst = y[0]
        Mbh = y[1]
        rh = y[2]

        if Mst <= 0 or rh <= 0 or Mst + Mbh <= 0:
            return 0, 0, 0

        M = Mst + Mbh
        fbh = Mbh / M

        trh = self._trh(M, rh, fbh)
        tcc = self.tcc
        tsev = self.tsev

        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0

        # Stellar mass loss
        if t >= tsev:
            Mst_dot -= self.nu * Mst / t
            rh_dot -= Mst_dot / M * rh

        # Add tidal mass loss
        if (self.tidal):
            xi = 0.6 * self.zeta * (rh / self._rt(M) / self.Rht) ** 1.5
            Mst_dot -= xi * M / trh

        # BH escape
        if t > tcc:
            rh_dot += self.zeta * rh / trh
            rh_dot += 2 * Mst_dot / M * rh

            if Mbh > 0:
                Mbh_dot = -self.beta * M / trh
                rh_dot += 2 * Mbh_dot / M * rh

        derivs = [Mst_dot]
        derivs.append(Mbh_dot)
        derivs.append(rh_dot)

        return numpy.array(derivs)

    def evolve(self):
        Mst = [self.M0]  # MG 29/1/2020 should be self.M0-self.Mbh0 ???
        Mbh = [self.Mbh0]
        rh = [self.rh0]

        y = [Mst[0], Mbh[0], rh[0]]

        def Mst_min_event(t, y):  # [17/3/22] stop when stars are lost
            return y[0] - self.Mst_min

        Mst_min_event.terminal = True
        Mst_min_event.direction = -1

        t_eval = numpy.arange(0, self.tend, self.dtout) if self.dtout is not None else None
        sol = solve_ivp(self.odes, [0, self.tend], y, events=Mst_min_event,
                        method=self.integration_method, t_eval=t_eval, dense_output=self.dense_output)

        self.t = sol.t
        self.Mst = sol.y[0]
        self.Mbh = sol.y[1]
        self.rh = sol.y[2]

        if len(sol.t_events[0]) != 0:
            self.tend = sol.t_events[0][0]

        if self.dense_output:
            self.Mst_interp = lambda t: sol.sol(t)[0]
            self.Mbh_interp = lambda t: sol.sol(t)[1]
            self.rh_interp = lambda t: sol.sol(t)[2]

        cbh = (self.Mbh > 0)
        self.mmax = numpy.zeros_like(self.Mbh)
        self.mmax[cbh] = self.find_mmax(self.Mbh[cbh])

        # Some derived quantities
        self.M = self.Mst + self.Mbh
        self.rt = self._rt(self.M)
        self.fbh = self.Mbh / self.M

        self.M_interp = lambda t: self.Mst_interp(t) + self.Mbh_interp(t)

        self.E = -self.G * self.M ** 2 / (2 * self.rh)

        m = self.M0 / self.N  # changed to M0 to keep m constant [17/3/22]
        self.trh = 0.138 * sqrt(self.M * self.rh ** 3 / self.G) / (m * self._psi(self.fbh) * 10)
        self.trh = numpy.where(numpy.logical_and(self.M > 0, self.rh > 0), self.trh, 1e-99)

        if self.output:
            f = open(self.outfile, "w")
            for i in range(len(self.t)):
                f.write("%12.5e %12.5e %12.5e %12.5e %12.5e\n" % (self.t[i] / 1e3, self.Mbh[i],
                                                                  self.M[i], self.rh[i],
                                                                  self.mmax[i]))

            f.close()


if __name__ == "__main__":
    clusterBH(1e6/0.638, 1e3)
