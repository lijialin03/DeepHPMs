import numpy as np
import time
import paddle


class DeepHPM(object):
    def __init__(self) -> None:
        self.net_idn = []
        self.net_pde = []
        self.net_sol = []
        self.opt_idn = None
        self.opt_pde = None
        self.opt_sol = None
        self.loss_fn = None
        self.max_grad = None
        self.num_var = 1

    def init_idn(self, net_uv, t, x, vars, lb, ub) -> None:
        self.net_idn = net_uv
        self.t_idn = paddle.to_tensor(t, dtype="float32")
        self.x_idn = paddle.to_tensor(x, dtype="float32")
        self.vars_idn = paddle.to_tensor(vars, dtype="float32")
        self.num_var = len(vars)
        self.lb_idn = paddle.to_tensor(lb, dtype="float32")
        self.ub_idn = paddle.to_tensor(ub, dtype="float32")

    def init_pde(self, net_uv) -> None:
        self.net_pde = net_uv

    def init_sol(self, tb, x0, vars0, lb, ub, X_f, net_uv=None) -> None:
        self.net_sol = net_uv if net_uv is not None else self.net_idn
        self.lb_sol = paddle.to_tensor(lb, dtype="float32")
        self.ub_sol = paddle.to_tensor(ub, dtype="float32")

        X0 = np.concatenate((0 * x0, x0), 1)  # (0, x0)
        X_lb = np.concatenate((tb, 0 * tb + lb[1]), 1)  # (tb, lb[1])
        X_ub = np.concatenate((tb, 0 * tb + ub[1]), 1)  # (tb, ub[1])

        # Initial Data
        self.t0_sol = paddle.to_tensor(X0[:, 0:1], dtype="float32")  # time
        self.x0_sol = paddle.to_tensor(X0[:, 1:2], dtype="float32")  # space
        self.vars0_sol = paddle.to_tensor(vars0, dtype="float32")

        # Boundary Data
        self.t_lb_sol = paddle.to_tensor(X_lb[:, 0:1], dtype="float32")  # time -- lb
        self.t_ub_sol = paddle.to_tensor(X_ub[:, 0:1], dtype="float32")  # time -- ub
        self.x_lb_sol = paddle.to_tensor(X_lb[:, 1:2], dtype="float32")  # space -- lb
        self.x_ub_sol = paddle.to_tensor(X_ub[:, 1:2], dtype="float32")  # space -- ub

        # Collocation Points
        self.X_f_sol = paddle.to_tensor(X_f, dtype="float32")
        self.t_f_sol = paddle.to_tensor(X_f[:, 0:1], dtype="float32")  # time
        self.x_f_sol = paddle.to_tensor(X_f[:, 1:2], dtype="float32")  # space

    def mean_squared_error(self, error):
        mse_loss = paddle.nn.MSELoss(reduction="mean")
        label = paddle.to_tensor(np.zeros(error.shape), dtype="float32")
        return mse_loss(error, label)

    def compile(self, optimizer="adam", lr=[0.01, 0.01, 0.01], loss="MSE", max_grad=2):
        if optimizer == "adam":
            if self.net_idn is not []:
                params = self.net_idn[0].parameters()
                for i in range(1, self.num_var):
                    params += self.net_idn[i].parameters()
                self.opt_idn = paddle.optimizer.Adam(
                    learning_rate=lr[0], parameters=params
                )
            if self.net_pde is not []:
                params = self.net_pde[0].parameters()
                for i in range(1, self.num_var):
                    params += self.net_pde[i].parameters()
                self.opt_pde = paddle.optimizer.Adam(
                    learning_rate=lr[1], parameters=params
                )
            if self.net_sol is not []:
                params = self.net_sol[0].parameters()
                for i in range(1, self.num_var):
                    params += self.net_sol[i].parameters()
                self.opt_sol = paddle.optimizer.Adam(
                    learning_rate=lr[2], parameters=params
                )
        if loss == "MSE":
            self.loss_fn = self.mean_squared_error
        self.max_grad = max_grad

    def train(self, N_iter, mode="idn"):
        """
        Args:
            mode (str): idn/pde/sol
        """
        start_time = time.time()
        for iter in range(N_iter):
            if mode is "idn":
                opt = self.opt_idn
                u_pred, v_pred, _, _ = self.forward_net_u(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )  # [u,v,u_x,v_x]
                losses = self.loss_fn(u_pred - self.vars_idn[0]) + self.loss_fn(
                    v_pred - self.vars_idn[1]
                )
            elif mode is "pde":
                opt = self.opt_pde
                f_pred, g_pred = self.forward_net_f(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )
                losses = self.loss_fn(f_pred) + self.loss_fn(g_pred)
            elif mode == "sol":
                opt = self.opt_sol
                u0, v0, _, _ = self.forward_net_u(
                    self.t0_sol, self.x0_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                u_lb, v_lb, u_x_lb, v_x_lb = self.forward_net_u(
                    self.t_lb_sol, self.x_lb_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                u_ub, v_ub, u_x_ub, v_x_ub = self.forward_net_u(
                    self.t_ub_sol, self.x_ub_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                f_p, g_p = self.forward_net_f(
                    self.t_f_sol, self.x_f_sol, self.lb_sol, self.ub_sol, self.net_sol
                )

                loss_u = self.loss_fn(self.vars0_sol[0] - u0)
                loss_v = self.loss_fn(self.vars0_sol[1] - v0)
                loss_ub = self.loss_fn(u_lb - u_ub)
                loss_vb = self.loss_fn(v_lb - v_ub)
                loss_uxb = self.loss_fn(u_x_lb - u_x_ub)
                loss_vxb = self.loss_fn(v_x_lb - v_x_ub)
                loss_f = self.loss_fn(f_p)
                loss_g = self.loss_fn(g_p)
                losses = (
                    loss_u
                    + loss_v
                    + loss_ub
                    + loss_vb
                    + loss_uxb
                    + loss_vxb
                    + loss_f
                    + loss_g
                )
                losses_list = [
                    loss_u,
                    loss_v,
                    loss_ub,
                    loss_vb,
                    loss_uxb,
                    loss_vxb,
                    loss_f,
                    loss_g,
                ]
            if iter % 1000 == 0:
                elapsed = time.time() - start_time
                print(
                    "It: %d, Time: %.2f, Loss: %.3e" % (iter, elapsed, losses), end=""
                )
                if mode == "sol":
                    for i in range(len(losses_list)):
                        print(", Loss%d: %.3e" % (i, losses_list[i]), end="")
                print("\n")
                start_time = time.time()
            losses.backward()
            opt.step()
            opt.clear_grad()

    def predict(self, t, x, mode):
        t = paddle.to_tensor(t, dtype="float32")
        x = paddle.to_tensor(x, dtype="float32")

        if mode == "idn":
            lb = self.lb_idn
            ub = self.ub_idn
            net = self.net_idn
        elif mode == "sol":
            lb = self.lb_sol
            ub = self.ub_sol
            net = self.net_sol

        pred_u, pred_v, _, _ = self.forward_net_u(t, x, lb, ub, net)
        pred_f, pred_g = self.forward_net_f(t, x, lb, ub, net)
        return pred_u.numpy(), pred_v.numpy(), pred_f.numpy(), pred_g.numpy()

    def forward_net_pde(self, terms):
        pde_u = self.net_pde[0].forward(terms)
        pde_v = self.net_pde[1].forward(terms)
        return pde_u, pde_v

    def forward_net_u(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False
        X = paddle.concat([t, x], axis=1)
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        u = net[0].forward(H)
        v = net[1].forward(H)

        u_x = paddle.grad(u, x, create_graph=True)[0]
        v_x = paddle.grad(v, x, create_graph=True)[0]

        return u, v, u_x, v_x

    def forward_net_f(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False

        u, v, u_x, v_x = self.forward_net_u(t, x, lb, ub, net)

        u_t = paddle.grad(u, t, create_graph=True)[0]
        v_t = paddle.grad(v, t, create_graph=True)[0]

        u_xx = paddle.grad(u_x, x, create_graph=True)[0]
        v_xx = paddle.grad(v_x, x, create_graph=True)[0]

        terms = paddle.concat([u, v, u_x, v_x, u_xx, v_xx], 1)
        pde_u, pde_v = self.forward_net_pde(terms)
        f = u_t - pde_u
        g = v_t - pde_v
        return f, g

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
        for i in range(self.num_var):
            paddle.save(net[i].state_dict(), path + mode + str(i) + ".pdparams")
        paddle.save(opt.state_dict(), path + mode + ".pdopt")

    def load(self, path, mode="pde"):
        for i in range(self.num_var):
            net_state_dict = paddle.load(path + mode + str(i) + ".pdparams")
            if mode == "idn":
                self.net_idn[i].set_state_dict(net_state_dict)
            elif mode == "pde":
                self.net_pde[i].set_state_dict(net_state_dict)
            elif mode == "sol":
                self.net_sol[i].set_state_dict(net_state_dict)

        opt_state_dict = paddle.load(path + mode + ".pdopt")
        if mode == "idn":
            self.opt_idn.set_state_dict(opt_state_dict)
        elif mode == "pde":
            self.opt_pde.set_state_dict(opt_state_dict)
        elif mode == "sol":
            self.opt_sol.set_state_dict(opt_state_dict)
