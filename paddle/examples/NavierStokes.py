import numpy as np
import time
import paddle
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

# import imp

paddle.device.set_device("gpu:1")


class Network(paddle.nn.Layer):
    """Network"""

    def __init__(self, layer_sizes):
        super().__init__()
        self.initializer_w = paddle.nn.initializer.XavierNormal()
        self.initializer_b = paddle.nn.initializer.Constant(value=0.0)

        self.linears = paddle.nn.LayerList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.initializer_w(self.linears[-1].weight)
            self.initializer_b(self.linears[-1].bias)

        self.activation = paddle.sin

    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        outputs = self.linears[-1](x)
        return outputs


class DeepHPM(object):
    def __init__(self, lb, ub) -> None:
        self.lb = paddle.to_tensor(lb, dtype="float32")
        self.ub = paddle.to_tensor(ub, dtype="float32")
        self.net_idn = None
        self.net_pde = None
        self.net_sol = None
        self.opt_idn = None
        self.opt_pde = None
        self.opt_sol = None
        self.loss_fn = None

    def init_idn(self, net_w, t, x, y, u, v, w) -> None:
        self.net_idn = net_w
        self.t_idn = paddle.to_tensor(t, dtype="float32")
        self.x_idn = paddle.to_tensor(x, dtype="float32")
        self.y_idn = paddle.to_tensor(y, dtype="float32")
        self.u_idn = paddle.to_tensor(u, dtype="float32")
        self.v_idn = paddle.to_tensor(v, dtype="float32")
        self.w_idn = paddle.to_tensor(w, dtype="float32")

    def init_pde(self, net) -> None:
        self.net_pde = net

    def init_sol(self, t_b, x_b, y_b, w_b, t_f, x_f, y_f, u_f, v_f, net=None) -> None:
        self.net_sol = net if net is not None else self.net_idn  # net

        # Initial and Boundary Data (4 boundaries)
        self.t_b = paddle.to_tensor(t_b, dtype="float32")  # time
        self.x_b = paddle.to_tensor(x_b, dtype="float32")  # space - x
        self.y_b = paddle.to_tensor(y_b, dtype="float32")  # space - y
        self.w_b = paddle.to_tensor(w_b, dtype="float32")  # vorticity

        # Collocation Points
        self.t_f = paddle.to_tensor(t_f, dtype="float32")  # time
        self.x_f = paddle.to_tensor(x_f, dtype="float32")  # space - x
        self.y_f = paddle.to_tensor(y_f, dtype="float32")  # space - y
        self.u_f = paddle.to_tensor(u_f, dtype="float32")  # space - u
        self.v_f = paddle.to_tensor(v_f, dtype="float32")  # space - v

    def mean_squared_error(self, error):
        mse_loss = paddle.nn.MSELoss(reduction="mean")
        label = paddle.to_tensor(np.zeros(error.shape), dtype="float32")
        return mse_loss(error, label)

    def compile(self, optimizer="adam", lr=[0.01, 0.01, 0.01], loss="MSE"):
        if optimizer == "adam":
            self.opt_idn = (
                paddle.optimizer.Adam(
                    learning_rate=lr[0], parameters=self.net_idn.parameters()
                )
                if self.net_idn is not None
                else None
            )
            self.opt_pde = (
                paddle.optimizer.Adam(
                    learning_rate=lr[1], parameters=self.net_pde.parameters()
                )
                if self.net_pde is not None
                else None
            )
            self.opt_sol = (
                paddle.optimizer.Adam(
                    learning_rate=lr[2], parameters=self.net_sol.parameters()
                )
                if self.net_sol is not None
                else None
            )
        if loss == "MSE":
            self.loss_fn = self.mean_squared_error

    def train(self, N_iter, mode="idn"):
        """
        Args:
            mode (str): idn/pde/sol
        """
        start_time = time.time()
        for iter in range(N_iter):
            if mode is "idn":
                opt = self.opt_idn
                w_pred_idn = self.forward_net_w(
                    self.t_idn, self.x_idn, self.y_idn, self.net_idn
                )
                losses = self.loss_fn(w_pred_idn - self.w_idn)
            elif mode is "pde":
                opt = self.opt_pde
                f_pred_pde = self.forward_net_f(
                    self.t_idn,
                    self.x_idn,
                    self.y_idn,
                    self.u_idn,
                    self.v_idn,
                    self.net_idn,
                )
                losses = self.loss_fn(f_pred_pde)
            elif mode == "sol":
                opt = self.opt_sol
                w_pred_sol = self.forward_net_w(
                    self.t_b, self.x_b, self.y_b, self.net_sol
                )
                f_pred_sol = self.forward_net_f(
                    self.t_f,
                    self.x_f,
                    self.y_f,
                    self.u_f,
                    self.v_f,
                    self.net_sol,
                )
                losses = self.loss_fn(self.w_b - w_pred_sol) + self.loss_fn(f_pred_sol)

            if iter % 1000 == 0:
                elapsed = time.time() - start_time
                print("It: %d, Time: %.2f, Loss: %.3e" % (iter, elapsed, losses))
                start_time = time.time()

            losses.backward()
            opt.step()
            opt.clear_grad()

    def predict(self, t, x, y, mode):
        t = paddle.to_tensor(t, dtype="float32")
        x = paddle.to_tensor(x, dtype="float32")
        y = paddle.to_tensor(y, dtype="float32")

        if mode == "idn":
            net = self.net_idn
        elif mode == "sol":
            net = self.net_sol
        pred_w = self.forward_net_w(t, x, y, net)
        return pred_w.numpy()

    def predict_pde(self, terms):
        pred_ped = self.forward_net_pde(terms)
        return pred_ped

    def forward_net_pde(self, terms):
        pde = self.net_pde.forward(terms)
        return pde

    def forward_net_w(self, t, x, y, net):
        t.stop_gradient = False
        x.stop_gradient = False
        y.stop_gradient = False
        X = paddle.concat([t, x, y], axis=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        w = net.forward(H)
        return w

    def forward_net_f(self, t, x, y, u, v, net):
        u.stop_gradient = False
        v.stop_gradient = False

        w = self.forward_net_w(t, x, y, net)

        w_t = paddle.grad(w, t, create_graph=True)[0]

        w_x = paddle.grad(w, x, create_graph=True)[0]
        w_y = paddle.grad(w, y, create_graph=True)[0]

        w_xx = paddle.grad(w_x, x, create_graph=True)[0]
        w_xy = paddle.grad(w_x, y, create_graph=True)[0]
        w_yy = paddle.grad(w_y, y, create_graph=True)[0]

        # terms = paddle.concat([w, w_x, w_y, w_xx, w_xy, w_yy], 1)
        terms = paddle.concat([u, v, w, w_x, w_y, w_xx, w_xy, w_yy], 1)
        f = w_t - self.net_pde(terms)

        return f

    def save(self, path, mode="pde"):
        if mode == "idn":
            net = self.net_idn
            opt = self.opt_idn
        elif mode == "pde":
            net = self.net_pde
            opt = self.opt_pde
        elif mode == "sol":
            net = self.net_sol
            opt = self.opt_sol
        paddle.save(net.state_dict(), path + mode + ".pdparams")
        paddle.save(opt.state_dict(), path + mode + ".pdopt")

    def load(self, path, mode="pde"):
        if mode == "idn":
            net = self.net_idn
            opt = self.opt_idn
        elif mode == "pde":
            net = self.net_pde
            opt = self.opt_pde
        elif mode == "sol":
            net = self.net_sol
            opt = self.opt_sol

        net_state_dict = paddle.load(path + mode + ".pdparams")
        opt_state_dict = paddle.load(path + mode + ".pdopt")
        net.set_state_dict(net_state_dict)
        opt.set_state_dict(opt_state_dict)


class Config(object):
    def __init__(self) -> None:
        self.lb = [0.0, 1, -1.7]
        self.ub = [30.0, 7.5, 1.7]

        self.N_train = 50000
        self.N_f = 50000

        self.layers_idn = [3, 200, 200, 200, 200, 1]
        self.layers_pde = [8, 100, 100, 1]
        self.layers_sol = [3, 200, 200, 200, 200, 1]


class DataLoader(object):
    def __init__(self) -> None:
        self.file = None
        self.cfg = Config()

        # outputs

    def set_bounds(self):
        """Doman bounds"""
        self.lb = np.array(self.cfg.lb)
        self.ub = np.array(self.cfg.ub)

    def prepare_training_data(self):
        """Load and Prepare data"""
        # Load data
        data = scipy.io.loadmat(self.file)
        t_data = data["t_star"]
        self.X_data = data["X_star"]
        U_data = data["U_star"]
        self.w_data = data["w_star"]

        t_star = np.tile(t_data.T, (2310, 1))
        x_star = np.tile(self.X_data[:, 0:1], (1, 151))
        y_star = np.tile(self.X_data[:, 1:2], (1, 151))
        u_star = U_data[:, 0, :]
        v_star = U_data[:, 1, :]
        w_star = self.w_data

        t_star = np.reshape(t_star, (-1, 1))
        x_star = np.reshape(x_star, (-1, 1))
        y_star = np.reshape(y_star, (-1, 1))
        u_star = np.reshape(u_star, (-1, 1))
        v_star = np.reshape(v_star, (-1, 1))
        w_star = np.reshape(w_star, (-1, 1))

        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.u_star = u_star
        self.v_star = v_star
        self.w_star = w_star

        # Prepare data
        # For identification
        idx = np.random.choice(t_star.shape[0], self.cfg.N_train, replace=True)
        self.t_train = t_star[idx, :]
        self.x_train = x_star[idx, :]
        self.y_train = y_star[idx, :]
        self.u_train = u_star[idx, :]
        self.v_train = v_star[idx, :]
        self.w_train = w_star[idx, :]

        # For solution
        t_b0 = t_star[t_star == t_star.min()][:, None]
        x_b0 = x_star[t_star == t_star.min()][:, None]
        y_b0 = y_star[t_star == t_star.min()][:, None]
        w_b0 = w_star[t_star == t_star.min()][:, None]

        t_b1 = t_star[x_star == x_star.min()][:, None]
        x_b1 = x_star[x_star == x_star.min()][:, None]
        y_b1 = y_star[x_star == x_star.min()][:, None]
        w_b1 = w_star[x_star == x_star.min()][:, None]

        t_b2 = t_star[x_star == x_star.max()][:, None]
        x_b2 = x_star[x_star == x_star.max()][:, None]
        y_b2 = y_star[x_star == x_star.max()][:, None]
        w_b2 = w_star[x_star == x_star.max()][:, None]

        t_b3 = t_star[y_star == y_star.min()][:, None]
        x_b3 = x_star[y_star == y_star.min()][:, None]
        y_b3 = y_star[y_star == y_star.min()][:, None]
        w_b3 = w_star[y_star == y_star.min()][:, None]

        t_b4 = t_star[y_star == y_star.max()][:, None]
        x_b4 = x_star[y_star == y_star.max()][:, None]
        y_b4 = y_star[y_star == y_star.max()][:, None]
        w_b4 = w_star[y_star == y_star.max()][:, None]

        self.t_b_train = np.concatenate((t_b0, t_b1, t_b2, t_b3, t_b4))
        self.x_b_train = np.concatenate((x_b0, x_b1, x_b2, x_b3, x_b4))
        self.y_b_train = np.concatenate((y_b0, y_b1, y_b2, y_b3, y_b4))
        self.w_b_train = np.concatenate((w_b0, w_b1, w_b2, w_b3, w_b4))

        idx = np.random.choice(t_star.shape[0], self.cfg.N_f, replace=True)
        self.t_f_train = t_star[idx, :]
        self.x_f_train = x_star[idx, :]
        self.y_f_train = y_star[idx, :]
        self.u_f_train = u_star[idx, :]
        self.v_f_train = v_star[idx, :]

    def __call__(self, file):
        # set config
        self.file = file

        # load data
        self.set_bounds()
        self.prepare_training_data()


class Plotting(object):
    def __init__(self, figname, X_data) -> None:
        self.figname = figname
        self.fig = plt.figure(figname, figsize=(10, 6))
        self.gs = gridspec.GridSpec(1, 2)
        self.lb = X_data.min(0)
        self.ub = X_data.max(0)
        self.X_data = X_data
        self.snap = 120
        self.nn = 200
        x_plot = np.linspace(self.lb[0], self.ub[0], self.nn)
        y_plot = np.linspace(self.lb[1], self.ub[1], self.nn)
        self.X_plot, self.Y_plot = np.meshgrid(x_plot, y_plot)

    def grid_data(self, values):
        return griddata(
            self.X_data,
            values[:, self.snap].flatten(),
            (self.X_plot, self.Y_plot),
            method="cubic",
        )

    def draw_subplot(self, subfigname, figdata, loc):
        ax = plt.subplot(self.gs[:, loc])
        h = ax.imshow(
            figdata,
            interpolation="nearest",
            cmap="seismic",
            extent=[self.lb[0], self.ub[0], self.lb[1], self.ub[1]],
            origin="lower",
            aspect="auto",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        self.fig.colorbar(h, cax=cax)
        # ax.axis("off")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_aspect("auto", "box")
        ax.set_title(subfigname, fontsize=10)

    def draw_n_save(self, data_exact, data_learned):
        self.gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
        # Exact p(t,x,y)
        self.draw_subplot("Exact Dynamics", self.grid_data(data_exact), loc=0)
        # Predicted p(t,x,y)
        self.draw_subplot("Learned Dynamics", self.grid_data(data_learned), loc=1)
        plt.savefig("../figures/" + self.figname)
        plt.close()

    def plot_solution(self, w_data, index):
        W_data = self.griddata(w_data)

        plt.figure(index)
        plt.pcolor(self.X_plot, self.Y_plot, W_data, cmap="jet")
        plt.colorbar()

    def plot_vorticity(self, x_star, y_star):
        data = scipy.io.loadmat("../../Data/cylinder_vorticity.mat")
        XX = data["XX"]
        YY = data["YY"]
        WW = data["WW"]
        WW[XX**2 + YY**2 < 0.25] = 0

        fig = plt.figure("cylinder_vorticity", figsize=(10, 6))

        gs0 = gridspec.GridSpec(1, 1)
        gs0.update(top=0.85, bottom=0.2, left=0.25, right=0.8, wspace=0.15)

        ax = plt.subplot(gs0[0:1, 0:1])
        h = ax.pcolormesh(
            XX, YY, WW, cmap="seismic", shading="gouraud", vmin=-5, vmax=5
        )
        ax.axis("off")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title("Vorticity", fontsize=10)
        fig.colorbar(h)

        ax.plot([x_star.min(), x_star.max()], [y_star.min(), y_star.min()], "r--")
        ax.plot([x_star.min(), x_star.max()], [y_star.max(), y_star.max()], "r--")
        ax.plot([x_star.min(), x_star.min()], [y_star.min(), y_star.max()], "r--")
        ax.plot([x_star.max(), x_star.max()], [y_star.min(), y_star.max()], "r--")

        plt.savefig("../figures/Cylinder_vorticity")
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


class Example(object):
    def __init__(self) -> None:
        pass

    def run(
        self,
        file,
        figname="test",
        lr=0.001,
        n_train_idn=10000,
        n_train_pde=10000,
        n_train_sol=10000,
        mode=["train"],
    ):
        print("########## Starting", figname, "##########")
        # Load Data
        data = DataLoader()
        data(file)
        plot = Plotting(figname, data.X_data)

        # Save Path
        net_path = "../saved_nets/" + figname + "_opt"

        # Training
        net_idn = Network(data.cfg.layers_idn)
        net_pde = Network(data.cfg.layers_pde)
        net_sol = Network(data.cfg.layers_sol)

        model = DeepHPM(data.lb, data.ub)
        model.init_idn(
            net_idn,
            data.t_train,
            data.x_train,
            data.y_train,
            data.u_train,
            data.v_train,
            data.w_train,
        )
        model.init_pde(net_pde)
        model.init_sol(
            data.t_b_train,
            data.x_b_train,
            data.y_b_train,
            data.w_b_train,
            data.t_f_train,
            data.x_f_train,
            data.y_f_train,
            data.u_f_train,
            data.v_f_train,
            net_sol,
        )

        # train idn and pde
        model.compile(lr=lr)
        if "load_gen_pde" in mode:
            model.load(net_path, "idn")
            model.load(net_path, "pde")
        if "train_gen_pde" in mode:
            model.train(n_train_idn, "idn")
            model.train(n_train_pde, "pde")
        # save nets
        if "save_gen_pde" in mode:
            model.save(net_path, "idn")
            model.save(net_path, "pde")
        # predict and print error
        w_pred_identifier = model.predict(data.t_star, data.x_star, data.y_star, "idn")
        error_w_identifier = np.linalg.norm(
            data.w_star - w_pred_identifier, 2
        ) / np.linalg.norm(data.w_star, 2)
        print("Error w idn-idn: %e" % (error_w_identifier))
        w_pred_identifier = np.reshape(w_pred_identifier, (-1, 151))
        if "debug_gen_pde" in mode:
            plot.figname = figname + "_debug_NS_pde"
            plot.draw_t_2d(data.w_data, w_pred_identifier)
            plot.draw_n_save(data.w_data, w_pred_identifier)
        #    step = 71
        #    plot.plot_solution(data.X_data,w_pred_identifier[:,step],1)
        #    plot.plot_solution(data.X_data,data.w_data[:,step],2)
        #    plot.plot_solution(data.X_data,np.abs(w_pred_identifier[:,step]-data.w_data[:,step]),3)

        # train sol
        if "load_pinns" in mode:
            model.load(net_path, "sol")
        if "train_pinns" in mode:
            model.train(n_train_sol, "sol")
        # save net_sol to resume training
        if "save_pinns" in mode:
            model.save(net_path, "sol")
        # predict and print error
        w_pred = model.predict(data.t_star, data.x_star, data.y_star, "sol")
        error_w = np.linalg.norm(data.w_star - w_pred, 2) / np.linalg.norm(
            data.w_star, 2
        )
        print("Error w sol-idn: %e" % (error_w))
        w_pred = np.reshape(w_pred, (-1, 151))
        #    step = 71
        #    plot.plot_solution(data.X_data,w_pred[:,step],4)
        #    plot.plot_solution(data.X_data,data.w_data[:,step],5)
        #    plot.plot_solution(data.X_data,np.abs(w_pred[:,step]-data.w_data[:,step]),6)

        # Plotting
        plot.figname = figname
        plot.draw_t_2d(data.w_data, w_pred)
        plot.draw_n_save(data.w_data, w_pred)
        plot.plot_vorticity(data.x_star, data.y_star)


if __name__ == "__main__":
    lr = [1e-4, 1e-4, 1e-4]
    N_train = [50000, 50000, 50000]
    example = Example()
    import paddle

    paddle.device.set_device("gpu:1")
    paddle.fluid.core.set_prim_eager_enabled(True)
    # NavierStokes
    example.run(
        "../../Data/cylinder.mat",
        "NavierStokes",
        # "NavierStokes_normal",
        lr,
        N_train[0],
        N_train[1],
        N_train[2],
        mode=[
            "load_gen_pde",
            # "train_gen_pde",
            # "debug_gen_pde",
            # "save_gen_pde",
            # "load_pinns",
            "train_pinns",
            # "save_pinns",
        ],
    )
