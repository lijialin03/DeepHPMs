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

    def init_sol(self, net_uv, tb, x0, vars0, lb, ub, X_f) -> None:
        self.net_sol = net_uv

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
        mse_loss = paddle.nn.MSELoss(reduction="mean")
        label = paddle.to_tensor(np.zeros(error.shape), dtype="float32")
        return mse_loss(error, label)

    def compile(self, optimizer="adam", lr=None, loss="MSE", max_grad=2):
        if optimizer == "adam":
            if self.net_idn is not []:
                params = self.net_idn[0].parameters()
                for i in range(1, self.num_var):
                    params += self.net_idn[i].parameters()
                self.opt_idn = paddle.optimizer.Adam(
                    learning_rate=lr, parameters=params
                )
            if self.net_pde is not []:
                params = self.net_pde[0].parameters()
                for i in range(1, self.num_var):
                    params += self.net_pde[i].parameters()
                self.opt_pde = paddle.optimizer.Adam(
                    learning_rate=lr, parameters=params
                )
            if self.net_sol is not []:
                params = self.net_sol[0].parameters()
                for i in range(1, self.num_var):
                    params += self.net_sol[i].parameters()
                self.opt_sol = paddle.optimizer.Adam(
                    learning_rate=lr, parameters=params
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
                vars_pred_idn = self.forward_net_u(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )  # [u,u_x,u_xx,...,v,v_x,v_xx,...]
                losses = 0
                for i in range(self.num_var):
                    error = vars_pred_idn[i * self.max_grad] - self.vars_idn[i]
                    losses += self.loss_fn(error)
            elif mode is "pde":
                pde_pred = self.forward_net_f(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )
                losses = 0
                for error in pde_pred:
                    losses += self.loss_fn(error)
            elif mode == "sol":
                self.vars0_pred = self.forward_net_u(
                    self.t0_sol, self.x0_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                vars_lb_pred_list = self.forward_net_u(
                    self.t_lb_sol, self.x_lb_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                vars_ub_pred_list = self.forward_net_u(
                    self.t_ub_sol, self.x_ub_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                self.sol_pde_pred = self.forward_net_f(
                    self.t_f_sol, self.x_f_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                losses = 0
                for i in range(self.num_var):
                    error = self.vars0_sol[i] - self.vars0_pred[i * self.max_grad]
                    losses += self.loss_fn(error) + self.loss_fn(self.sol_pde_pred[i])
                    for j in range(self.max_grad - 1):
                        losses += self.loss_fn(
                            vars_lb_pred_list[i * self.max_grad + j + 1]
                            - vars_ub_pred_list[i * self.max_grad + j + 1]
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

        pred_u = self.forward_net_u(t, x, lb, ub, net)[0]
        pred_f = self.forward_net_f(t, x, lb, ub, net)[0]
        if self.num_var >= 2:
            pred_v = self.forward_net_u(t, x, lb, ub, net)[self.max_grad]
            pred_g = self.forward_net_f(t, x, lb, ub, net)[1]
            return [pred_u.numpy(), pred_f.numpy(), pred_v.numpy(), pred_g.numpy()]

        return [pred_u.numpy(), pred_f.numpy()]

    def forward_net_pde(self, terms):
        pde = []
        for i in range(self.num_var):
            pde.append(self.net_pde[i].forward(terms))
        return pde

    def forward_net_u(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False
        X = paddle.concat([t, x], axis=1)
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        u = net[0].forward(H)
        if self.num_var >= 2:
            v = net[1].forward(H)

        res = [u]
        grad = u
        for i in range(1, self.max_grad):
            if i < 2:
                new_grad = paddle.grad(grad, x, create_graph=True)[0]
            else:
                new_grad = paddle.grad(grad, x, create_graph=False)[0]
            res.append(new_grad)
            grad = new_grad

        if self.num_var >= 2:
            res.append(v)
            grad = v
            for i in range(1, self.max_grad):
                if i < 2:
                    new_grad = paddle.grad(grad, x, create_graph=True)[0]
                else:
                    new_grad = paddle.grad(grad, x, create_graph=False)[0]
                res.append(new_grad)
                grad = new_grad

        return res

    def forward_net_f(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False

        terms_list = []
        vars = self.forward_net_u(t, x, lb, ub, net)
        u = vars[0]
        u_t = paddle.grad(u, t, create_graph=True)[0]
        terms_list = [u]
        grad = u
        for i in range(1, self.max_grad):
            new_grad = paddle.grad(grad, x, create_graph=True)[0]
            terms_list.append(new_grad)
            grad = new_grad
        terms_list.append(paddle.grad(grad, x, create_graph=False)[0])

        if self.num_var >= 2:
            v = vars[self.max_grad]
            v_t = paddle.grad(v, t, create_graph=True)[0]
            terms_list.append(v)
            grad = v
            for i in range(1, self.max_grad):
                new_grad = paddle.grad(grad, x, create_graph=True)[0]
                terms_list.append(new_grad)
                grad = new_grad
            terms_list.append(paddle.grad(grad, x, create_graph=False)[0])

        # terms_list [u,u_x,u_xx,...,v,v_x,v_xx,...]
        terms = paddle.concat(terms_list, 1)
        pdes = self.forward_net_pde(terms)
        f = u_t - pdes[0]
        if self.num_var >= 2:
            g = v_t - pdes[1]
            return [f, g]
        return [f]

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
