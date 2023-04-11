import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotting(object):
    def __init__(self, figname, lb, ub) -> None:
        self.figname = figname
        self.fig = plt.figure(figname, figsize=(15, 6))
        self.gs = gridspec.GridSpec(1, 3)
        self.lb = lb
        self.ub = ub

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

    def draw_n_save(self, data_exact, data_pinns, data_learned):
        self.gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
        # Exact p(t,x,y)
        self.draw_subplot("Exact Dynamics", data_exact, loc=0)
        # PINNs Predicted p(t,x,y)
        self.draw_subplot("PINNs Learned Dynamics", data_pinns, loc=1)
        # Predicted p(t,x,y)
        self.draw_subplot("Learned Dynamics", data_learned, loc=2)
        plt.savefig("../figures/" + self.figname)
