import numpy as np
import matplotlib.pyplot as plt


class SEM:

    def __init__(self, y, U, uu, vv=None, ww=None, uv=None, Linf=None):
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
        self.U = U

        # Blend to the free-stream length scale
        # Assume y is normalised on BL thickness
        if Linf is None:
            L = np.interp(y, [0., 1., 2.], [0., 0.41, 0.41])
        else:
            L = np.interp(y, [0., 1., 2 * Linf], [0., 0.41, Linf])
        self.L = L[:, None, None, None]

        # Assemble Reynolds stress tensor
        self.R = np.zeros((np.size(y), 3, 3))
        self.R[:, 0, 0] = self.uu
        self.R[:, 1, 1] = self.vv
        self.R[:, 2, 2] = self.ww
        self.R[:, 1, 0] = self.uv
        self.R[:, 0, 1] = self.uv
        self.a = np.linalg.cholesky(self.R)[:, None, None, ...]

        # Set bounding box
        # Square centered around z=0, from 0 to ymax, at x=0, with Lmax space around it
        Lmax = np.max(self.L)
        side = np.max(self.y_t)
        bb = np.array([[-Lmax, Lmax],
                       [0, side + Lmax],
                       [-(side + Lmax) / 2, (side + Lmax) / 2]])
        self.box = bb

        # Calculate number of eddies
        Dens = 100.
        self.Vol = np.prod(np.diff(self.box, 1, 1))
        self.Nk = np.int(Dens * self.Vol / Lmax ** 3.)

        # Initialise eddies uniformly over box
        self.xk = np.empty((1, 1, self.Nk, 3))
        for i in range(3):
            self.xk[..., i] = np.random.uniform(bb[i, 0], bb[i, 1], (self.Nk,))

        # Get length scales associated with each eddy
        self.lk = np.interp(self.xk[..., 1], self.y_t.flatten(), self.L.flatten())[..., None]
        self.Uk = np.interp(self.xk[..., 1], self.y_t.flatten(), self.U)

        # Choose eddy orientations
        self.ek = np.random.choice((-1, 1), self.xk.shape)

    def evaluate(self, yg, zg):
        """Evaluate fluctuating velocity field associated with the current eddies."""

        # Assemble input grid vector
        sg = np.shape(yg)
        xg = np.stack((np.zeros_like(yg), yg, zg), 2)[:, :, None, :]

        # Get distances to all eddies
        dxk = (xg - self.xk) / self.lk

        # Evaluate components shape function
        f = np.where(np.abs(dxk) < 1., 1. - dxk ** 2., 0.)

        # Take product of all compotents and normalise
        fsig = np.prod(f, -1, keepdims=True) / self.lk ** 3. * np.sqrt(self.Vol)

        # Compute sum
        u = np.einsum('...kij,...kj,...kl->...i', self.a, self.ek, fsig) / np.sqrt(self.Nk)

        return u

    def convect(self, dt):
        """Move the eddies."""

        # Time step is normalised by dt_hat = dt * u_tau / delta
        self.xk[..., 1] = self.xk[..., 1] + dt * self.Uk

        # Check if any eddies have left the box
        has_left = self.xk[..., 0] > self.box[0, 1]
        Nk_new = np.sum(has_left)
        xk_new = np.zeros((1, 1, Nk_new, 3))
        for i in range(3):
            xk_new[..., i] = np.random.uniform(self.box[i, 0], self.box[i, 1], (Nk_new,))

        # Get length scales associated with each eddy
        lk_new = np.interp(xk_new[..., 1], self.y_t.flatten(), self.L.flatten())[..., None]
        Uk_new = np.interp(xk_new[..., 1], self.y_t.flatten(), self.U)
        ek_new = np.random.choice((-1, 1), xk_new.shape)

        self.xk[has_left, :] = xk_new
        self.Uk[0, 0, has_left.flatten()] = Uk_new
        self.lk[0, 0, has_left.flatten()] = lk_new
        self.ek[has_left, :] = ek_new

    def loop(self, yg, zg, dt, Nt):

        u = np.zeros(np.shape(yg) + (3, Nt))
        print('Time step %d/%d' % (1, Nt), end="")
        for i in range(Nt):
            print('\r', end="")
            print('Time step %d/%d' % (i + 1, Nt), end="")
            t = np.linspace(0, dt * (Nt - 1000), Nt)
            u[..., i] = self.evaluate(yg, zg)
            self.convect(dt)

        self.u = u
        self.yg = yg
        self.zg = zg

    def plot_input(self):

        f, a = plt.subplots(1, 3, sharey=True)
        a[0].plot(self.uu, self.y_t, 'x', label="$\overline{u\'u\'}$")
        a[0].plot(self.vv, self.y_t, 'x', label="$\overline{v\'v\'}$")
        a[0].plot(self.ww, self.y_t, 'x', label="$\overline{w\'w\'}$")
        a[0].plot(self.uv, self.y_t, 'x', label="$\overline{u\'v\'}$")

        a[1].plot(self.L.flatten(), self.y_t, 'kx')

        a[2].plot(self.U, self.y_t, 'kx')

        a[0].set_ylabel("$y/\delta$")
        a[0].set_xlabel("Reynolds Stress, $\overline{u_i\'u_j\'}$")
        a[1].set_xlabel("Length Scale, $\ell/\delta$")
        a[2].set_xlabel(r"Mean velocity, $U/U_{\tau}$")

        a[0].legend()
        plt.tight_layout()
        plt.show()

    def plot_output(self):

        f, a = plt.subplots(1, 3)
        a[0].plot(self.uu, self.y_t, 'x', label="$\overline{u\'u\'}$")
        a[0].plot(self.vv, self.y_t, 'x', label="$\overline{v\'v\'}$")
        a[0].plot(self.ww, self.y_t, 'x', label="$\overline{w\'w\'}$")
        a[0].plot(self.uv, self.y_t, 'x', label="$\overline{u\'v\'}$")

        # Calculate stats
        uu = np.mean(self.u[..., 0, :] ** 2., (-2, -1))
        vv = np.mean(self.u[..., 1, :] ** 2., (-2, -1))
        ww = np.mean(self.u[..., 2, :] ** 2., (-2, -1))
        uv = np.mean(self.u[..., 0, :] * self.u[..., 1, :], (-2, -1))

        a[0].plot(uu, self.y_t, '-')
        a[0].plot(vv, self.y_t, '-')
        a[0].plot(ww, self.y_t, '-')
        a[0].plot(uv, self.y_t, '-')

        a[1].plot(self.L.flatten(), self.y_t, 'kx')

        a[2].plot(self.u[10, 10, 0, :].flatten())

        a[0].set_ylabel("$y/\delta$")
        a[0].set_xlabel("Reynolds Stress, $\overline{u_i\'u_j\'}$")
        a[1].set_xlabel("Length Scale, $\ell/\delta$")
        a[2].set_xlabel(r"Mean velocity, $U/U_{\tau}$")

        a[0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Load data and interpolate onto a common grid
    Dat = np.genfromtxt('Ziefle2013_Re_stress.csv', delimiter=',', skip_header=2)
    for n in range(np.size(Dat, 1)):
        for m in range(np.size(Dat, 0)):
            if np.isnan(Dat[m, n]):
                Dat[m:, n] = Dat[m - 1, n]
                break

    y_in = np.arctanh(np.linspace(0.005, .95, 51))
    y_in = y_in / np.max(y_in) * 1.0
    U_in = np.where(y_in < 1., y_in ** (1. / 7.), 1.0) * 100.

    uu_in = np.interp(y_in, Dat[:, 0], Dat[:, 1])
    vv_in = np.interp(y_in, Dat[:, 2], Dat[:, 3])
    ww_in = np.interp(y_in, Dat[:, 4], Dat[:, 5])
    uv_in = np.interp(y_in, Dat[:, 6], Dat[:, 7])

    theSEM = SEM(y_in, U_in, uu_in, vv_in, ww_in, uv_in, .75)

    # theSEM.plot_input()

    zgv_in = np.linspace(-1., 1., 11)
    ygv_in = y_in

    zg_in, yg_in = np.meshgrid(zgv_in, ygv_in)

    theSEM.loop(yg_in, zg_in, 0.5, 1000)

    theSEM.plot_output()
