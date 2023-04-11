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

    def init_idn(self, net_uv, t, x, vars, lb, ub) -> None:
        self.net_idn = net_uv
        self.t_idn = paddle.to_tensor(t, dtype="float32")
        self.x_idn = paddle.to_tensor(x, dtype="float32")
        self.vars_idn = paddle.to_tensor(vars, dtype="float32")
        self.lb_idn = paddle.to_tensor(lb, dtype="float32")
        self.ub_idn = paddle.to_tensor(ub, dtype="float32")

    def init_pde(self, net_uv) -> None:
        self.net_pde = net_uv

    def init_sol(self, net, tb, x0, vars0, lb, ub, X_f) -> None:
        self.net_sol = net

        X0 = np.concatenate((0 * x0, x0), 1)  # (0, x0)
        X_lb = np.concatenate((tb, 0 * tb + lb[1]), 1)  # (tb, lb[1])
        X_ub = np.concatenate((tb, 0 * tb + ub[1]), 1)  # (tb, ub[1])

        self.t0_sol = paddle.to_tensor(
            X0[:, 0:1], dtype="float32"
        )  # Initial Data (time)
        self.x0_sol = paddle.to_tensor(
            X0[:, 1:2], dtype="float32"
        )  # Initial Data (space)
        self.vars0_sol = paddle.to_tensor(vars0, dtype="float32")  # Boundary Data
        self.X_f_sol = paddle.to_tensor(X_f, dtype="float32")  # Collocation Points
        self.lb_sol = paddle.to_tensor(lb, dtype="float32")
        self.ub_sol = paddle.to_tensor(ub, dtype="float32")

        self.t_lb_sol = paddle.to_tensor(
            X_lb[:, 0:1], dtype="float32"
        )  # Boundary Data (time) -- lower boundary
        self.t_ub_sol = paddle.to_tensor(
            X_ub[:, 0:1], dtype="float32"
        )  # Boundary Data (time) -- upper boundary
        self.x_lb_sol = paddle.to_tensor(
            X_lb[:, 1:2], dtype="float32"
        )  # Boundary Data (space) -- lower boundary
        self.x_ub_sol = paddle.to_tensor(
            X_ub[:, 1:2], dtype="float32"
        )  # Boundary Data (space) -- upper boundary
        self.t_f_sol = paddle.to_tensor(
            X_f[:, 0:1], dtype="float32"
        )  # Collocation Points (time)
        self.x_f_sol = paddle.to_tensor(
            X_f[:, 1:2], dtype="float32"
        )  # Collocation Points (space)

    def mean_squared_error(self, error):
        return paddle.sum(paddle.square(error))

    def compile(self, optimizer="adam", lr=None, loss="MSE", max_grad=2):
        if optimizer == "adam":
            self.opt_idn = (
                paddle.optimizer.Adam(
                    learning_rate=lr, parameters=self.net_idn.parameters()
                )
                if self.net_idn is not None
                else None
            )
            self.opt_pde = (
                paddle.optimizer.Adam(
                    learning_rate=lr, parameters=self.net_pde.parameters()
                )
                if self.net_pde is not None
                else None
            )
            self.opt_sol = (
                paddle.optimizer.Adam(
                    learning_rate=lr, parameters=self.net_sol.parameters()
                )
                if self.net_sol is not None
                else None
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
                for i in range(2):
                    vars_pred_idn = self.forward_net_u(
                        self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                    )[0]
                    error = vars_pred_idn - self.vars_idn
                    losses = 0
                    for e in error:
                        losses += self.loss_fn(e)
            elif mode is "pde":
                pde_pred = self.forward_net_f(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )
                losses = 0
                for e in pde_pred:
                    losses += self.loss_fn(e)
            elif mode == "sol":
                self.vars0_pred = self.forward_net_u(
                    self.t0_sol, self.x0_sol, self.lb_sol, self.ub_sol, self.net_sol
                )[0]
                vars_lb_pred_list = self.forward_net_u(
                    self.t_lb_sol, self.x_lb_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                vars_ub_pred_list = self.forward_net_u(
                    self.t_ub_sol, self.x_ub_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                self.sol_pde_pred = self.forward_net_f(
                    self.t_f_sol, self.x_f_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                error = self.vars0_sol - self.vars0_pred
                losses = 0
                for i in range(len(error)):
                    losses = self.loss_fn(error[i]) + self.loss_fn(self.sol_pde_pred[i])
                for j in range(self.max_grad):
                    for i in range(len(error)):
                        losses += self.loss_fn(
                            vars_lb_pred_list[j][i] - vars_ub_pred_list[j][i]
                        )
            if iter % 1000 == 0:
                elapsed = time.time() - start_time
                print("It: %d, Loss: %.3e, Time: %.2f" % (iter, losses, elapsed))
                start_time = time.time()
            losses.backward()
            if mode == "idn":
                self.opt_idn.step()
                self.opt_idn.clear_grad()
            elif mode == "pde":
                self.opt_pde.step()
                self.opt_pde.clear_grad()
            elif mode == "sol":
                self.opt_sol.step()
                self.opt_sol.clear_grad()

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

        pred_vars = self.forward_net_u(t, x, lb, ub, net)[0]
        pred_fg = self.forward_net_f(t, x, lb, ub, net)
        return pred_vars.numpy(), pred_fg.numpy()

    def forward_net_pde(self, terms):
        pde = []
        for i in range(2):
            pde.append(self.net_pde[i].forward(terms))
        return pde

    def forward_net_u(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False
        X = paddle.concat([t, x], axis=1)
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        vars = []
        for i in range(2):
            vars.append(net[i].forward(H))
        res = [vars]
        grad = res[0]
        for i in range(1, self.max_grad - 1):
            new_grad = []
            for j in range(len(grad)):
                new_grad.append(paddle.grad(grad[j], x, create_graph=True)[0])
            res.append(new_grad)
            grad = new_grad
        new_grad = []
        for j in range(len(grad)):
            new_grad.append(paddle.grad(grad[j], x, create_graph=False)[0])
        res.append(new_grad)
        return res

    def forward_net_f(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False

        vars = self.forward_net_u(t, x, lb, ub, net)[0]
        vars_t = []
        for i in range(2):
            vars_t.append(paddle.grad(vars[i], t, create_graph=True)[0])
        terms_list = [vars]
        grad = terms_list[0]
        for i in range(1, self.max_grad):
            new_grad = []
            for j in range(len(grad)):
                new_grad.append(paddle.grad(grad[j], x, create_graph=True)[0])
            terms_list.append(new_grad)
            grad = new_grad
        new_grad = []
        for j in range(len(grad)):
            new_grad.append(paddle.grad(grad[j], x, create_graph=False)[0])
        terms_list.append(new_grad)
        terms = paddle.concat(terms_list, 1)
        pde = self.forward_net_pde(terms)
        fg = []
        for i in range(2):
            fg.append(vars_t[i] - pde[i])
        return fg

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
        net_state_dict = paddle.load(path + mode + ".pdparams")
        opt_state_dict = paddle.load(path + mode + ".pdopt")
        if mode == "idn":
            self.net_idn.set_state_dict(net_state_dict)
            self.opt_idn.set_state_dict(opt_state_dict)
        elif mode == "pde":
            self.net_pde.set_state_dict(net_state_dict)
            self.opt_pde.set_state_dict(opt_state_dict)
        elif mode == "sol":
            self.net_sol.set_state_dict(net_state_dict)
            self.opt_sol.set_state_dict(opt_state_dict)
