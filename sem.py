import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import math
import time
import json
import importlib.resources as pkg_resources


class SEM:

    def __init__(self, y, U, uu, vv=None, ww=None, uv=None, Linf=None, del_99=None, Dens=100., fsh='quadratic'):
        """Initialise with target Reynolds stress profile.

        We normalise velocity with friction velocity, length with BL thickness."""

        # Default to homogenous isotropic if only uu provided
        if vv is None:
            vv = uu
        if ww is None:
            ww = uu
        if uv is None:
            uv = np.zeros_like(y)

        # Assign attributes
        self.y_t = y
        self.uu = uu
        self.vv = vv
        self.ww = ww
        self.uv = uv
        self.U = np.trapz(U, y) / np.trapz(np.ones_like(y), y)

        # Blend to the free-stream length scale
        # Assume y is normalised on BL thickness
        if Linf is None:
            L = np.interp(y, [0., 1., 2.], [0., 0.41, 0.41])
        else:
            L = np.interp(y, [0., del_99, 2 * Linf], [0., del_99 * 0.41, Linf])
        self.L = L[:, None, None, None]

        # Assemble Reynolds stress tensor
        self.R = np.zeros((np.size(y), 3, 3))
        self.R[:, 0, 0] = self.uu
        self.R[:, 1, 1] = self.vv
        self.R[:, 2, 2] = self.ww
        self.R[:, 1, 0] = self.uv
        self.R[:, 0, 1] = self.uv

        # Make interpolator for Reynolds stress
        self.fR = scipy.interpolate.interp1d(y, self.R, axis=0)

        # Set bounding box
        # Square centered around z=0, from 0 to ymax, at x=0, with Lmax space around it
        Lmax = np.max(self.L)
        side = np.max(self.y_t)
        bb = np.array([[-Lmax, Lmax],
                       [0, side + Lmax],
                       [-(side + Lmax) / 2, (side + Lmax) / 2]])
        self.box = bb

        # Calculate number of eddies
        self.Vol = np.prod(np.diff(self.box, 1, 1))
        self.Nk = np.int(Dens * self.Vol / Lmax ** 3.)

        print("Eddy box dimensions: ", np.ptp(bb, 1))
        print("Number of eddies: ", self.Nk)

        # Initialise eddies uniformly over box
        self.xk = np.empty((1, 1, self.Nk, 3))
        for i in range(3):
            self.xk[..., i] = np.random.uniform(bb[i, 0], bb[i, 1], (self.Nk,))

        # Get length scales associated with each eddy
        self.lk = np.interp(self.xk[..., 1], self.y_t.flatten(), self.L.flatten())[..., None]
        self.Uk = np.ones_like(self.xk[..., 1]) * self.U

        # Choose eddy orientations
        self.ek = np.random.choice((-1, 1), self.xk.shape)

        # Choose shape function normalisation factor
        # = (int from -1 to 1 of fsh**2) **2
        if fsh == 'gaussian':
            self.fac_norm = math.erf(np.sqrt(np.pi * 2.)) / np.sqrt(2.)
        elif fsh == 'quadratic':
            self.fac_norm = 16. / 15.
        elif fsh == 'triangular':
            self.fac_norm = 2. / 3.
        self.fac_norm = 1. / np.sqrt(self.fac_norm)
        self.fsh = fsh

    def evaluate(self, yg, zg):
        """Evaluate fluctuating velocity field associated with the current eddies."""

        # Assemble input grid vector
        xg = np.stack((np.zeros_like(yg), yg, zg), 2)[:, :, None, :]

        # Get distances to all eddies
        dxksq = ((xg - self.xk) / self.lk) ** 2.

        # Get Reynolds stresses at grid points of interest
        Rg = self.fR(yg)
        ag = np.linalg.cholesky(Rg)[:, :, None, ...]

        # Evaluate components shape function
        if self.fsh == 'gaussian':
            f = np.where(dxksq < 1., np.exp(-np.pi * dxksq), 0.)
        elif self.fsh == 'quadratic':
            f = np.where(dxksq < 1., 1. - dxksq, 0.)
        elif self.fsh == 'triangular':
            f = np.where(dxksq < 1., 1. - np.sqrt(dxksq), 0.)
        else:
            raise Exception('Invalid shape function')

        # Normalise
        f = f * self.fac_norm

        fsig = np.prod(f, -1, keepdims=True) * np.sqrt(self.Vol / self.lk ** 3.)

        # Compute sum
        u = np.einsum('...kij,...kj,...kl->...i', ag, self.ek, fsig) / np.sqrt(self.Nk)

        return u

    def convect(self, dt):
        """Move the eddies."""

        # Time step is normalised by dt_hat = dt * u_tau / delta
        self.xk[..., 0] = self.xk[..., 0] + dt * self.Uk

        # Check if any eddies have left the box
        has_left = self.xk[..., 0] > self.box[0, 1]
        Nk_new = np.sum(has_left)
        xk_new = np.zeros((1, 1, Nk_new, 3))
        for i in [1, 2]:
            xk_new[..., i] = np.random.uniform(self.box[i, 0], self.box[i, 1], (Nk_new,))
        xk_new[..., 0] = self.box[0, 0]

        # Get length scales associated with each eddy
        lk_new = np.interp(xk_new[..., 1], self.y_t.flatten(), self.L.flatten())[..., None]
        Uk_new = np.ones_like(xk_new[..., 1]) * self.U
        ek_new = np.random.choice((-1, 1), xk_new.shape)

        self.xk[has_left, :] = xk_new
        self.Uk[0, 0, has_left.flatten()] = Uk_new
        self.lk[0, 0, has_left.flatten()] = lk_new
        self.ek[has_left, :] = ek_new

    def loop(self, yg, zg, dt, Nt):

        start_time = time.perf_counter()

        if np.min(zg) < self.box[2, 0]:
            raise Exception('Output grid too wide.')

        u = np.zeros(np.shape(yg) + (3, Nt))
        print('Time step %d/%d' % (0, Nt), end="")
        for i in range(Nt):
            if not np.mod(i, 50):
                print('\r', end="")
                print('Time step %d/%d' % (i, Nt), end="")
            u[..., i] = self.evaluate(yg, zg)
            self.convect(dt)

        print('\nElapsed time:', time.perf_counter() - start_time, "seconds")

        self.u = u
        return

    def plot_input(self):

        f, a = plt.subplots(1, 3, sharey=True)
        a[0].plot(self.uu, self.y_t, 'x', label="$\overline{u\'u\'}$")
        a[0].plot(self.vv, self.y_t, 'x', label="$\overline{v\'v\'}$")
        a[0].plot(self.ww, self.y_t, 'x', label="$\overline{w\'w\'}$")
        a[0].plot(-self.uv, self.y_t, 'x', label="$-\overline{u\'v\'}$")

        a[1].plot(self.L.flatten(), self.y_t, 'kx')

        a[0].set_ylabel("$y/\delta$")
        a[0].set_xlabel("Reynolds Stress, $\overline{u_i\'u_j\'}$")
        a[1].set_xlabel("Length Scale, $\ell/\delta$")
        a[2].set_xlabel(r"Mean velocity, $U/U_{\tau}$")

        a[0].legend()
        plt.tight_layout()
        plt.show()

    def plot_output(self, yg):

        f, a = plt.subplots(1, 3)
        a[0].plot(self.uu, self.y_t, 'x', label="$\overline{u\'u\'}$")
        a[0].plot(self.vv, self.y_t, 'x', label="$\overline{v\'v\'}$")
        a[0].plot(self.ww, self.y_t, 'x', label="$\overline{w\'w\'}$")
        a[0].plot(-self.uv, self.y_t, 'x', label="-$\overline{u\'v\'}$")

        # Calculate stats
        uu = np.mean(self.u[..., 0, :] ** 2., (-2, -1))
        vv = np.mean(self.u[..., 1, :] ** 2., (-2, -1))
        ww = np.mean(self.u[..., 2, :] ** 2., (-2, -1))
        uv = np.mean(self.u[..., 0, :] * self.u[..., 1, :], (-2, -1))

        a[0].set_prop_cycle(None)

        a[0].plot(uu, yg[:, 0], '-')
        a[0].plot(vv, yg[:, 0], '-')
        a[0].plot(ww, yg[:, 0], '-')
        a[0].plot(-uv, yg[:, 0], '-')

        # a[1].plot(uu / self.uu, self.y_t, '-')
        # a[1].plot(vv / self.vv, self.y_t, '-')
        # a[1].plot(ww / self.ww, self.y_t, '-')
        # a[1].plot(uv / self.uv, self.y_t, '-')

        a[2].plot(self.u[1, 1, 0, :].flatten())

        a[0].set_ylabel("$y/\delta$")
        a[0].set_xlabel("Reynolds Stress, $\overline{u_i\'u_j\'}$")
        a[1].set_xlabel("Length Scale, $\ell/\delta$")
        a[2].set_xlabel(r"Mean velocity, $U/U_{\tau}$")

        a[1].legend()
        plt.tight_layout()
        plt.show()


def main_old():
    # Load data and interpolate onto a common grid
    Dat = np.genfromtxt('Ziefle2013_Re_stress.csv', delimiter=',', skip_header=2)
    ien = np.ones((np.size(Dat, 1, )), dtype=np.int) * np.size(Dat, 0)
    for n in range(np.size(Dat, 1)):
        for m in range(np.size(Dat, 0)):
            if np.isnan(Dat[m, n]):
                Dat[m:, n] = Dat[m - 1, n]
                ien[n] = m
                break

    # Reformat the input data into a dict, save as json
    d = {'uu': Dat[:ien[0], 0:2].T.tolist(),
         'vv': Dat[:ien[2], 2:4].T.tolist(),
         'ww': Dat[:ien[4], 4:6].T.tolist(),
         'uv': Dat[:ien[6], 6:8].T.tolist()}
    with open('Re_stress.json', 'w') as fp:
        json.dump(d, fp, indent=4)

    y_in = np.arctanh(np.linspace(0.005, .95, 17))
    y_in = y_in / np.max(y_in) * 1.0
    U_in = np.where(y_in < 1., y_in ** (1. / 7.), 1.0) * 22.

    uu_in = np.interp(y_in, Dat[:, 0], Dat[:, 1])
    vv_in = np.interp(y_in, Dat[:, 2], Dat[:, 3])
    ww_in = np.interp(y_in, Dat[:, 4], Dat[:, 5])
    uv_in = np.interp(y_in, Dat[:, 6], Dat[:, 7])

    uv_in[y_in > 1.] = 0.0
    uu_in[y_in > 1.] = 0.005 * 22.
    vv_in[y_in > 1.] = 0.005 * 22.
    ww_in[y_in > 1.] = 0.005 * 22.

    theSEM = SEM(y_in, U_in, uu_in, vv_in, ww_in, -uv_in, .75, Dens=100., fsh='quadratic')

    zgv_in = np.linspace(-.5, .5, 3)
    ygv_in = np.linspace(0.2, 0.5, 11)

    zg_in, yg_in = np.meshgrid(zgv_in, ygv_in)

    print(theSEM.loop(yg_in, zg_in, .001, 10000))
    print(theSEM.Nk)
    theSEM.plot_output(yg_in)


class BoundaryLayer(SEM):
    """Setup the SEM for a boundary layer flow."""

    def __init__(self, del_99, h, Tu_inf, cf, L_inf, Dens):
        """Initialise with boundary layer thickness and main-stream information.

        We take unit main-stream velocity, and an arbitrary length scale."""

        # Read Reynolds stress data (uu_rms as fn of y/del_99)
        with pkg_resources.open_text('tugen.data', 'Re_stress.json') as fid:
            Re_stress = json.load(fid)

        # Rescale the Reynolds stress data
        # To get uu normalised by main-stream velocity, need
        # (uu_rms/v_tau)^2 * (v_tau/v_inf)^2 = Data * cf/2
        Re_stress_inf = {'uu': Tu_inf ** 2., 'vv': Tu_inf ** 2., 'ww': Tu_inf ** 2., 'uv': 0.}
        Re_stress_min = {'uu': 1e-9, 'vv': 1e-9, 'ww': 1e-9, 'uv': 0.}
        for k in Re_stress:
            ui = np.array(Re_stress[k])
            ui[0, :] = ui[0, :] * del_99
            ui[1, :] = ui[1, :] ** 2. * cf / 2.
            ui[1, ui[0, :] > del_99] = Re_stress_inf[k]
            ui[0, 0] = 0.
            ui[1, 0] = Re_stress_min[k]
            Re_stress[k] = ui

        # Create wall-normal grid vector
        A = 2.
        zeta = np.linspace(0., 1., 41)
        yg_bl = (1. + np.tanh(A * (zeta - 1.)) / np.tanh(A)) * del_99
        yg_inf = np.array((yg_bl[-1] + np.diff(yg_bl, 1)[-1], 2. * L_inf, h))
        yg = np.concatenate((yg_bl, yg_inf))

        # Interpolate all points to same grid
        Re_stress_g = {}
        for k in Re_stress:
            Re_stress_g[k] = np.interp(yg, Re_stress[k][0, :], Re_stress[k][1, :])

        super().__init__(yg, np.ones_like(yg),
                         Re_stress_g['uu'],
                         Re_stress_g['vv'],
                         Re_stress_g['ww'],
                         -Re_stress_g['uv'],
                         L_inf,
                         del_99,
                         Dens=Dens)

        return


if __name__ == '__main__':
    BL = BoundaryLayer(0.5, 8., 0.05, 4.2e-3, 1., 100.)

    zgv_in = np.linspace(-2., .2, 3)
    ygv_in = np.linspace(0., 2.2, 6)

    zg_in, yg_in = np.meshgrid(zgv_in, ygv_in)

    BL.loop(yg_in, zg_in, .02, 3200)

    BL.plot_output(yg_in)

    # main_old()
