import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from plotting import Plotting

data_idn = scipy.io.loadmat("../../Data/burgers_sine.mat")

t_idn = data_idn["t"].flatten()[:, None]
x_idn = data_idn["x"].flatten()[:, None]
Exact_idn = np.real(data_idn["usol"])
# print(np.shape(Exact_idn))
T_idn, X_idn = np.meshgrid(t_idn, x_idn)

t_idn_star = T_idn.flatten()[:, None]
x_idn_star = X_idn.flatten()[:, None]
X_idn_star = np.hstack((t_idn_star, x_idn_star))

nx = 256
dx = 16.0 / nx
nt = 200
dt = 10 / nt
c = 0.1
d = 0.02
x_all = np.linspace(-8, 8, nx + 1)
t_all = np.linspace(0, 10, nt + 1)


def fun(u_im1, u_i, u_ia1):
    # # just first derivative of x
    # u_na1 = u_i - c * (dt / dx) * u_i * np.subtract(u_i, u_im1)
    u_na1 = (
        u_i
        - c * (dt / dx) * u_i * np.subtract(u_i, u_im1)
        + d * (np.subtract(u_ia1, 2 * u_i) + u_im1) * (dt / (dx * dx))
    )
    return u_na1


u0 = np.zeros((nx + 1))
for i in range(nx + 1):
    u0[i] = -np.sin(np.pi * (x_all[i] / 8))
    # u0[i] = np.exp(-((x_all[i] + 2) ** 2))
u0[0] = 0  # np.sin(np.pi) = 1.22e-16 but not 0
u0[-1] = u0[0]

plt.plot(x_all, u0, label="initial")

u_tx = np.zeros((nx + 1, nt + 1801))
u_tx[:, 0] = u0
for n in range(1, nt + 1801):
    u_tx[:, n] = u_tx[:, n - 1]
    u_tx[-1][n - 1] = u_tx[0][n - 1]
    for i in range(1, nx):
        u_tx[i][n] = fun(u_tx[i - 1][n - 1], u_tx[i][n - 1], u_tx[i + 1][n - 1])
    u_tx[0][n] = fun(u_tx[-2][n - 1], u_tx[0][n - 1], u_tx[1][n - 1])
    # u_tx[0][n] = 0.0

plt.plot(x_all[:-1], u_tx[:-1, -1], label="converged")
plt.plot(x_all[:-1], Exact_idn[:, -1], label="Exact_idn")
plt.legend(loc="right")
plt.savefig("./test")

# data_idn["usol"] = u_tx
# scipy.io.savemat("./test_1.mat", data_idn)

plot = Plotting("test_gen", [0.0, -8.0], [100.0, 8.0], X_idn_star, (T_idn, X_idn))
# plot.draw_n_save(Exact_idn, u_tx[:-1, -201:])
plot.gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
plot.draw_subplot("test_gen", u_tx, loc=0)
plt.savefig("../figures/test_gen")
plt.close()
