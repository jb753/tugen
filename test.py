import numpy as np
import sem
import matplotlib.pyplot as plt
#import cProfile

yv = np.linspace(0.,1.,7)
zv = np.linspace(0.,1.,7)
Tu = 0.1 * np.ones_like(yv)
uv = np.zeros_like(yv)
L = np.ones_like(yv)*0.1

theSEM = sem.AnisoSEM(yv, zv, Tu, Tu, Tu, uv, L, 1.)
# theSEM.plot_input()

zg, yg = np.meshgrid(zv, yv)

theSEM.set_grid(yg, zg)

nstep_cycle = 20
ncycle = 1000
dt = L[0]/nstep_cycle
tau = L[0]*ncycle
nt = int(tau/dt)

theSEM.loop(dt, nt)
#cProfile.run("theSEM.loop(dt, nt)",sort="cumtime")

theSEM.plot_output(yv, nstep_cycle)

plt.show()
