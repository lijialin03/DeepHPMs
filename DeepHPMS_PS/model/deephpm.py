import numpy as np
import os
import sys
import time
import paddle


class DeepHPM(object):
    def __init__(self) -> None:
        self.net_idn = None
        self.net_pde = None
        self.net_sol = None
        self.opt_idn = None
        self.opt_pde = None
        self.opt_sol = None
        self.loss_fn = None
        self.max_grad = None

    def init_idn(self, net, t, x, u, lb, ub) -> None:
        self.net_idn = net
        self.t_idn = paddle.to_tensor(t, dtype="float32")
        self.x_idn = paddle.to_tensor(x, dtype="float32")
        self.u_idn = paddle.to_tensor(u, dtype="float32")
        self.lb_idn = paddle.to_tensor(lb, dtype="float32")
        self.ub_idn = paddle.to_tensor(ub, dtype="float32")

    def init_pde(self, net) -> None:
        self.net_pde = net

    def init_sol(self, net, tb, x0, u0, lb, ub, X_f) -> None:
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
        self.u0_sol = paddle.to_tensor(u0, dtype="float32")  # Boundary Data
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
                    learning_rate=0.00001, parameters=self.net_pde.parameters()
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
            N_iter (int): number of iteration
            mode (str): idn/pde/sol
        """
        start_time = time.time()
        for iter in range(N_iter):
            if mode is "idn":
                opt = self.opt_idn
                u_pred_idn = self.forward_net_u(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )[0]
                losses = self.loss_fn(u_pred_idn - self.u_idn)
            elif mode is "pde":
                opt = self.opt_pde
                pde_pred = self.forward_net_f(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )
                losses = self.loss_fn(pde_pred)
            elif mode == "sol":
                opt = self.opt_sol
                self.u0_pred = self.forward_net_u(
                    self.t0_sol, self.x0_sol, self.lb_sol, self.ub_sol, self.net_sol
                )[0]
                u_lb_pred_list = self.forward_net_u(
                    self.t_lb_sol, self.x_lb_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                u_ub_pred_list = self.forward_net_u(
                    self.t_ub_sol, self.x_ub_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                self.sol_f_pred = self.forward_net_f(
                    self.t_f_sol, self.x_f_sol, self.lb_sol, self.ub_sol, self.net_sol
                )

                losses = self.loss_fn(self.u0_sol - self.u0_pred) + self.loss_fn(
                    self.sol_f_pred
                )
                for i in range(self.max_grad):
                    losses += self.loss_fn(u_lb_pred_list[i] - u_ub_pred_list[i])

            if iter % 1000 == 0:
                elapsed = time.time() - start_time
                print("It: %d, Loss: %.3e, Time: %.2f" % (iter, losses, elapsed))
                start_time = time.time()

            losses.backward()
            opt.step()
            opt.clear_grad()

    def predict(self, t, x, mode):
        t = paddle.to_tensor(t, dtype="float32")
        x = paddle.to_tensor(x, dtype="float32")

        if mode in ["idn", "pde"]:
            lb = self.lb_idn
            ub = self.ub_idn
            net = self.net_idn
        elif mode == "sol":
            lb = self.lb_sol
            ub = self.ub_sol
            net = self.net_sol

        pred_u = self.forward_net_u(t, x, lb, ub, net)[0]
        pred_f = self.forward_net_f(t, x, lb, ub, net)
        return pred_u.numpy(), pred_f.numpy()

    def forward_net_pde(self, terms):
        pde = self.net_pde.forward(terms)
        return pde

    def forward_net_u(self, t, x, lb, ub, net):
        t.stop_gradient = False
        x.stop_gradient = False
        X = paddle.concat([t, x], axis=1)
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        u = net.forward(H)
        res = [u]
        grad = u
        for i in range(1, self.max_grad - 1):
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

        u = self.forward_net_u(t, x, lb, ub, net)[0]
        u_t = paddle.grad(u, t, create_graph=True)[0]
        terms_list = [u]
        grad = u
        for i in range(1, self.max_grad):
            if i < 2:
                new_grad = paddle.grad(grad, x, create_graph=True)[0]
            else:
                new_grad = paddle.grad(grad, x, create_graph=False)[0]
            terms_list.append(new_grad)
            grad = new_grad
        terms = paddle.concat(terms_list, 1)
        f = u_t - self.forward_net_pde(terms)
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
        paddle.save(net.state_dict(), path + "_" + mode + ".pdparams")
        paddle.save(opt.state_dict(), path + "_" + mode + ".pdopt")

    def load(self, path, mode="pde"):
        net_path = path + "_" + mode + ".pdparams"
        opt_path = path + "_" + mode + ".pdopt"
        if not os.path.exists(net_path) or not os.path.exists(opt_path):
            sys.exit(
                f"No {mode} model found in the path, "
                "please check your path or swith to 'train' mode."
            )

        net_state_dict = paddle.load(net_path)
        opt_state_dict = paddle.load(opt_path)
        if mode == "idn":
            self.net_idn.set_state_dict(net_state_dict)
            self.opt_idn.set_state_dict(opt_state_dict)
        elif mode == "pde":
            self.net_pde.set_state_dict(net_state_dict)
            self.opt_pde.set_state_dict(opt_state_dict)
        elif mode == "sol":
            self.net_sol.set_state_dict(net_state_dict)
            self.opt_sol.set_state_dict(opt_state_dict)

    def train_debug(self, N_iter, t, x, lb, ub, points, xi, mode="idn"):
        """Predict and save plot every 1000 iters. Only avaliable for 'sol'.
        Args:
            N_iter (int): number of iteration
            mode (str): idn/pde/sol
        """
        sys.path.append("../scripts/")
        from plotting import Plotting

        start_time = time.time()
        for iter in range(N_iter):
            if mode is "idn":
                opt = self.opt_idn
                u_pred_idn = self.forward_net_u(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )[0]
                losses = self.loss_fn(u_pred_idn - self.u_idn)
            elif mode is "pde":
                opt = self.opt_pde
                pde_pred = self.forward_net_f(
                    self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
                )
                losses = self.loss_fn(pde_pred)
            elif mode == "sol":
                opt = self.opt_sol
                self.u0_pred = self.forward_net_u(
                    self.t0_sol, self.x0_sol, self.lb_sol, self.ub_sol, self.net_sol
                )[0]
                u_lb_pred_list = self.forward_net_u(
                    self.t_lb_sol, self.x_lb_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                u_ub_pred_list = self.forward_net_u(
                    self.t_ub_sol, self.x_ub_sol, self.lb_sol, self.ub_sol, self.net_sol
                )
                self.sol_f_pred = self.forward_net_f(
                    self.t_f_sol, self.x_f_sol, self.lb_sol, self.ub_sol, self.net_sol
                )

                losses = self.loss_fn(self.u0_sol - self.u0_pred) + self.loss_fn(
                    self.sol_f_pred
                )
                for i in range(self.max_grad):
                    losses += self.loss_fn(u_lb_pred_list[i] - u_ub_pred_list[i])

            # if iter < 10000:
            #     opt.learning_rate = 0.001
            if iter % 1000 == 0:
                elapsed = time.time() - start_time
                print("It: %d, Loss: %.3e, Time: %.2f" % (iter, losses, elapsed))
                u_debug, _ = self.predict(t, x, mode)
                # Plotting
                plot = Plotting("iter_" + str(iter), lb, ub, points, xi)
                plot.draw_debug(u_debug)
                start_time = time.time()

            losses.backward()
            opt.step()
            opt.clear_grad()
