import numpy as np
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt
import time
import json
from context import RE_DATA
import numexpr

quadratic_Cf = np.sqrt(4./3.)
quadratic_norm = 1. / np.sqrt(16. / 15.)

class AnisoSEM:
    """Generate correlated velocity fluctuations using the synthetic eddies."""

    def __init__(self, y, z, uu, vv, ww, uv, L, Dens=100., fsh='quadratic'):
        """Initialise using target Reynolds stresses and length scales."""

        # Correct length scale
        L = L* quadratic_Cf

        # Store input data
        self.y = y
        self.L = L

        # Assemble Reynolds stress tensor
        self.R = np.zeros((np.size(y), 3, 3))
        self.R[:, 0, 0] = uu
        self.R[:, 1, 1] = vv
        self.R[:, 2, 2] = ww
        self.R[:, 1, 0] = uv
        self.R[:, 0, 1] = uv

        # Make interpolator for Reynolds stress
        self.fR = scipy.interpolate.interp1d(y, self.R, axis=0)

        # Set bounding box
        # Square centered around z=0, from 0 to ymax, at x=0, with Lmax space around it
        Lmax = np.amax(L)
        bb = np.array([[-Lmax, Lmax],
                       [y[0] - Lmax, y[-1] + Lmax],
                       [z[0] - Lmax, z[-1] + Lmax]])
        self.box = bb

        # Uniform probability density in homogenous dirns
        self.px = 1. / (bb[0, 1] - bb[0, 0])
        self.pz = 1. / (bb[2, 1] - bb[2, 0])

        # Extend length scale distribution outside of box
        self.y_ext = np.concatenate(([bb[1,0]], y, [bb[1,1]]))
        self.L_ext = np.concatenate(([L[0]], L, [L[-1]]))

        # Determine the eddy probability density in y dirn
        # Proportional to reciprocal of L 
        py_raw = 1. / self.L_ext
        self.py = py_raw / np.trapz(py_raw, self.y_ext)
        self.cpy = np.insert(
                scipy.integrate.cumtrapz(self.py, self.y_ext), 0, 0)


        # Calculate number of eddies
        self.Vol = np.prod(np.diff(self.box, 1, 1))
        self.Nk = np.int(Dens * self.Vol /  np.amax(self.L) ** 3.)

        print("Eddy box dimensions: ", np.ptp(bb, 1))
        print("Number of eddies: ", self.Nk)

        # Initialise eddies positions using probability distributions
        self.xk = np.empty((self.Nk, 3))
        for i in [0, 2]:
            self.xk[:, i] = np.random.uniform(bb[i, 0], bb[i, 1], (self.Nk,))
        self.xk[:, 1] = np.interp(np.random.rand(self.Nk), self.cpy, self.y_ext)

        # Get length scales associated with each eddy
        self.lk = np.interp(self.xk[:, 1], self.y_ext, self.L_ext)[..., None]

        # Choose eddy orientations
        self.ek = np.random.choice((-1, 1), self.xk.shape)

        # Evaluate total scaling factor for each eddy
        pyk = np.interp(self.xk[:, 1], self.y_ext, self.py)
        self.sfk = np.sqrt(1. / pyk[None, None, :, None]
                / self.px / self.pz / self.lk ** 3.) * self.ek

    def set_grid(self, yg, zg):
        """Prepare to evaluate eddies over a particular grid."""

        # Assemble input grid vector
        self.xg = np.stack((np.zeros_like(yg), yg, zg), 2)[:, :, None, :]

        # Get Reynolds stresses at grid points of interest
        self.ag = np.linalg.cholesky( self.fR(yg) )

    def evaluate(self, nproc=None):
        """Evaluate fluctuating velocity associated with current eddies."""

        # Get distances to all eddies
        xg = self.xg
        xk = self.xk[None, None, ...]
        lk = self.lk[None, None, ...]
        dxksq = numexpr.evaluate('((xg-xk)/lk)**2.0')

        # # Evaluate shape function
        f = numexpr.evaluate( ('prod(where(dxksq < 1.0,'
                'quadratic_norm * (1.0 - dxksq) , 0.0),3)'))[...,None]

        # Normalise shape function
        sfk = self.sfk
        fsig_ek = numexpr.evaluate('sum(f * sfk, 2)')

        # Compute sum
        u = np.squeeze( np.einsum('...ij,...j', self.ag, fsig_ek)
                ) / np.sqrt(self.Nk)

        return u

    def convect(self, dt):
        """Move the eddies."""

        # Time step is normalised by dt_hat = dt * u_tau / delta
        self.xk[:, 0] = self.xk[:, 0] + dt

        # Check if any eddies have left the box
        has_left = self.xk[:, 0] > self.box[0, 1]
        Nk_new = np.sum(has_left)

        # Get new positions for fresh eddies
        xk_new = np.zeros((Nk_new, 3))
        xk_new[:, 0] = self.box[0, 0]  # x at upstream side
        # y with nonuniform dist
        xk_new[:, 1] = np.interp(np.random.rand(Nk_new), self.cpy, self.y_ext)
        # z with uniform dist
        xk_new[:, 2] = np.random.uniform(
                self.box[2, 0], self.box[2, 1], (Nk_new,))

        # Get length scales and orientations for new eddies
        lk_new = np.interp(xk_new[:, 1], self.y_ext, self.L_ext)[..., None]
        ek_new = np.random.choice((-1, 1), xk_new.shape)
        pyk_new = np.interp(xk_new[:, 1], self.y_ext, self.py)

        # Insert into grid
        self.xk[has_left, :] = xk_new
        self.lk[has_left, :] = lk_new
        self.ek[has_left, :] = ek_new

        self.sfk[:, :, has_left, :] = np.sqrt(
                1. / pyk_new[None, None, :, None]
                / self.px / self.pz / lk_new ** 3.) * ek_new

    def loop(self, dt, Nt):
        start_time = time.perf_counter()

        u = np.zeros(np.shape(self.xg)[:2] + (3, Nt))
        print('Time step %d/%d' % (0, Nt), end="")
        for i in range(Nt):
            if not np.mod(i, 50):
                print('\r', end="")
                print('Time step %d/%d' % (i, Nt), end="")
            u[..., i] = self.evaluate()
            self.convect(dt)

        print('\nElapsed time:', time.perf_counter() - start_time, "seconds")

        self.u = u
        return

    def plot_input(self):
        f, a = plt.subplots(1, 3, sharey=True)
        a[0].plot(self.R[:,0,0], self.y, 'x', label="$\overline{u\'u\'}$")
        a[0].plot(self.R[:,1,1], self.y, 'x', label="$\overline{v\'v\'}$")
        a[0].plot(self.R[:,2,2], self.y, 'x', label="$\overline{w\'w\'}$")
        a[0].plot(-self.R[:,0,1], self.y, 'x', label="$-\overline{u\'v\'}$")

        a[1].plot(self.L.flatten(), self.y, 'kx')

        a[0].set_ylabel("$y/\delta$")
        a[0].set_xlabel("Reynolds Stress, $\overline{u_i\'u_j\'}$")
        a[1].set_xlabel("Length Scale, $\ell/\delta$")

        a[2].plot(self.py, self.y)
        a[2].plot(self.cpy, self.y)
        a[2].set_xlabel(r"Probability Density, $p(y)$")

        a[0].legend()
        plt.tight_layout()
        plt.show()

    def plot_output(self,yg,nstep_cycle):
        # yg = self.xg[0,:,0]

        f, a = plt.subplots(1, 2)
        a[0].plot(self.R[:,0,0], self.y, 'x', label="$\overline{u\'u\'}$")
        a[0].plot(self.R[:,1,1], self.y, 'x', label="$\overline{v\'v\'}$")
        a[0].plot(self.R[:,2,2], self.y, 'x', label="$\overline{w\'w\'}$")
        a[0].plot(-self.R[:,0,1], self.y, 'x', label="$-\overline{u\'v\'}$")

        # Calculate stats
        uu = np.mean(self.u[..., 0, :] ** 2., (-2, -1))
        vv = np.mean(self.u[..., 1, :] ** 2., (-2, -1))
        ww = np.mean(self.u[..., 2, :] ** 2., (-2, -1))
        uv = np.mean(self.u[..., 0, :] * self.u[..., 1, :], (-2, -1))

        a[0].set_prop_cycle(None)

        a[0].plot(uu, yg, 'o')
        a[0].plot(vv, yg, 'o')
        a[0].plot(ww, yg, 'o')
        a[0].plot(-uv, yg, 'o')

        # ux = self.u[0,0,0,:]
        # Nt = len(ux)
        # Rxx = (np.correlate(ux,ux,mode='full')/np.mean(ux**2.)/Nt)[Nt-1:]
        # a[1].plot(Rxx,'-x')
        # a[1].set_xlim([0.,50.])

        # Lam = np.trapz(Rxx)
        # print(Lam)

        ux = self.u[:,:,0,:]
        shu = np.shape(ux)
        Nt = shu[2]
        Ny = shu[0]
        Nz = shu[1]
        print(shu)

        Lam = np.empty((Ny,Nz))
        for i in range(Ny):
            for j in range(Nz):
                uxnow = ux[i,j,:]
                Rxxnow = (np.correlate(
                        uxnow,uxnow,mode='full')/np.mean(uxnow**2.)/Nt)[Nt-1:]
                Lam[i,j] = np.trapz(Rxxnow[:nstep_cycle*5])

        a[1].plot(np.mean(Lam,1)/nstep_cycle,yg,'-x')
        a[1].set_xlim([0.,2.])

        print(np.mean(Lam)/nstep_cycle)

        plt.tight_layout()
        #plt.show()


class BoundaryLayer(AnisoSEM):
    """Setup the SEM for a boundary layer flow."""

    def __init__(self, del_99, Re_theta, w, h, Tu_inf, L_inf, Dens):
        """Initialise with boundary layer thickness and main-stream information.

        We take unit main-stream velocity, and an arbitrary length scale."""

        # Read Reynolds stress data (uu_rms as fn of y/del_99)
        dat = np.loadtxt(RE_DATA)

        # Use skin friction coefficient to scale velocity fluc on main stream
        # vtau/vinf = sqrt(cf/2)
        cf = 0.025*Re_theta**-0.25 # Kays (1980)
        dat[:,1:] = dat[:,1:] * np.sqrt(cf/2.)

        # # Raise Reynolds stresses in main-stream
        iinf = dat[:,0] > 0.2
        dat[iinf,1:4] = np.where(dat[iinf,1:4]<Tu_inf,Tu_inf,dat[iinf,1:4])

        # Use bl thickness to scale y
        dat[:,0] = dat[:,0] * del_99

        f,a = plt.subplots()
        for i in range(1,5):
            a.plot(dat[:,i],dat[:,0],'-x')
        a.set_ylim([0.,2.*del_99])

        plt.show()

        super().__init__(yg,
                         Re_stress_g['uu'],
                         Re_stress_g['vv'],
                         Re_stress_g['ww'],
                         -Re_stress_g['uv'],
                         L_inf,
                         del_99,
                         Dens=Dens)

        return


def worker_sum(x):
    return np.sum(np.prod(x[0], -1, keepdims=True) * x[1], 2)


if __name__ == '__main__':
    # main_old()
    # quit()

    #def __init__(self, del_99, Re_theta, w, h, Tu_inf, L_inf, Dens):
    D = 0.005
    BL = BoundaryLayer(D, 800., 2.*D, 4.*D, 0.05, D, 10.)

    BL.plot_input()
    zgv_in = np.linspace(-0.1, .2, 2) * 0.005
    ygv_in = np.linspace(0.0001, .003, 17)
    #ygv_in = BL.y

    zg_in, yg_in = np.meshgrid(zgv_in, ygv_in)

    BL.loop(yg_in, zg_in, .02 * .005, 3200)

    BL.plot_output(yg_in)
