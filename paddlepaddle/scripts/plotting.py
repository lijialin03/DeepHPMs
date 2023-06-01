import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


class Plotting(object):
    def __init__(self, figname, lb, ub, points, xi) -> None:
        self.figname = figname
        self.fig = plt.figure(figname, figsize=(10, 6))
        self.gs = gridspec.GridSpec(1, 2)
        self.lb = lb
        self.ub = ub
        self.points = points
        self.xi = xi

    def grid_data(self, values):
        return griddata(self.points, values.flatten(), self.xi, method="cubic")

    def draw_subplot(self, subfigname, figdata, loc):
        ax = plt.subplot(self.gs[:, loc])
        h = ax.imshow(
            figdata,
            interpolation="nearest",
            cmap="jet",
            extent=[self.lb[0], self.ub[0], self.lb[1], self.ub[1]],
            origin="lower",
            aspect="auto",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        self.fig.colorbar(h, cax=cax)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_aspect("auto", "box")
        ax.set_title(subfigname, fontsize=10)

        # line = np.linspace(self.lb[1], self.ub[1], 2)[:, None]
        # ax.plot(t_idn[index] * np.ones((2, 1)), line, "w-", linewidth=1)

    def draw_n_save(self, data_exact, data_learned):
        self.gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
        # Exact p(t,x,y)
        self.draw_subplot("Exact Dynamics", data_exact, loc=0)
        # Predicted p(t,x,y)
        self.draw_subplot("Learned Dynamics", self.grid_data(data_learned), loc=1)
        plt.savefig("../figures/" + self.figname)
        plt.close()

    def draw_debug(self, data_debug):
        self.gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
        self.draw_subplot("Debug Dynamics", self.grid_data(data_debug), loc=0)
        plt.savefig("../figures/debug/" + self.figname)
        plt.close()

    def draw_t_2d(self, data_exact, data_t):
        import numpy as np

        nx, nt = np.shape(data_exact)
        plt.plot(np.linspace(-8, 8, nx + 1)[:-1], data_exact[:, -1], label="exact")
        plt.plot(
            np.linspace(-8, 8, nx + 1)[:-1],
            data_exact[:, int(nt / 2)],
            label="exact_helf_t",
        )
        data_t = np.reshape(data_t, (nx, nt))
        plt.plot(
            np.linspace(-8, 8, nx + 1)[:-1],
            data_t[:, int(nt / 2)],
            label="learned_helf_t",
        )
        plt.plot(np.linspace(-8, 8, nx + 1)[:-1], data_t[:, -1], label="learned")
        plt.legend(loc="right")
        plt.savefig("../figures/debug/test_t_" + self.figname)
        plt.close()
