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
        bb = np.array([[0., 0.],
                       [0., side],
                       [-side / 2, side / 2]])
        bb[:, 0] = bb[:, 0] - Lmax
        bb[:, 1] = bb[:, 1] + Lmax
        self.box = bb

        # Calculate number of eddies
        Dens = 1.
        self.Vol = np.prod(np.diff(self.box, 1, 1))
        self.Nk = np.int(Dens * self.Vol / Lmax ** 3.)

        # Initialise eddies uniformly over box
        self.xk = np.empty((1, 1, self.Nk, 3))
        for i in range(3):
            self.xk[..., i] = np.random.uniform(bb[i, 0], bb[i, 1], (self.Nk,))

        # Choose eddy orientations
        self.ek = np.random.choice((-1, 1), self.xk.shape)

    def evaluate(self, yg, zg):
        """Evaluate fluctuating velocity field associated with the current eddies."""

        # Assemble input grid vector
        sg = np.shape(yg)
        xg = np.stack((np.zeros_like(yg), yg, zg), 2)[:, :, None, :]

        # Get distances to all eddies
        dxk = (xg - self.xk) / self.L

        # Evaluate shape function
        f = np.prod(np.where(dxk < 1., 0., 1. - dxk ** 2.), -1, keepdims=True) / (self.L ** 3.) * np.sqrt(self.Vol)

        # Compute sum over all components of aij ekj
        u = np.einsum('...kij,...kj,...kl->...i', self.a, self.ek, f) / np.sqrt(self.Nk)

        lev = np.linspace(-1.0, 1.0, 11)
        f, a = plt.subplots()
        a.contourf(yg, zg, u[..., 0], levels=lev, cmap='RdBu')
        a.axis('equal')
        plt.show()

        return u

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


if __name__ == '__main__':
    # Load data and interpolate onto a common grid
    Dat = np.genfromtxt('Ziefle2013_Re_stress.csv', delimiter=',', skip_header=2)
    for n in range(np.size(Dat, 1)):
        for m in range(np.size(Dat, 0)):
            if np.isnan(Dat[m, n]):
                Dat[m:, n] = Dat[m - 1, n]
                break

    eps = 0.1
    Dat[Dat < eps] = eps

    y_in = np.arctanh(np.linspace(0.005, .95, 51))
    y_in = y_in / np.max(y_in) * 2.0
    U_in = np.where(y_in < 1., y_in ** (1. / 7.), 1.0) * 100.

    uu_in = np.interp(y_in, Dat[:, 0], Dat[:, 1])
    vv_in = np.interp(y_in, Dat[:, 2], Dat[:, 3])
    ww_in = np.interp(y_in, Dat[:, 4], Dat[:, 5])
    uv_in = np.interp(y_in, Dat[:, 6], Dat[:, 7])

    theSEM = SEM(y_in, U_in, uu_in, vv_in, ww_in, uv_in, .75)

    theSEM.plot_input()

    zgv_in = np.linspace(-1., 1., 11)
    ygv_in = y_in

    yg_in, zg_in = np.meshgrid(zgv_in, ygv_in)

    u_out = theSEM.evaluate(yg_in, zg_in)
