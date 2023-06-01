import numpy as np
import os
import sys
import time
import paddle

np.set_printoptions(threshold=np.inf)


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
        self.opt_op = "adam"

    def init_idn(self, net, t, x, u, lb, ub) -> None:
        self.net_idn = net
        self.t_idn = paddle.to_tensor(t, dtype="float32")
        self.x_idn = paddle.to_tensor(x, dtype="float32")
        self.u_idn = paddle.to_tensor(u, dtype="float32")
        self.lb_idn = paddle.to_tensor(lb, dtype="float32")
        self.ub_idn = paddle.to_tensor(ub, dtype="float32")

    def init_pde(self, net) -> None:
        self.net_pde = net

    def init_sol(self, net, tb, x0, u0, lb, ub, X_f, ts=None, xs=None, us=None) -> None:
        self.net_sol = net if net is not None else self.net_idn  # net
        self.lb_sol = paddle.to_tensor(lb, dtype="float32")
        self.ub_sol = paddle.to_tensor(ub, dtype="float32")

        X0 = np.concatenate((0 * x0, x0), 1)  # (0, x0)
        X_lb = np.concatenate((tb, 0 * tb + lb[1]), 1)  # (tb, lb[1])
        X_ub = np.concatenate((tb, 0 * tb + ub[1]), 1)  # (tb, ub[1])

        # Initial Data
        self.t0_sol = paddle.to_tensor(X0[:, 0:1], dtype="float32")  # time
        self.x0_sol = paddle.to_tensor(X0[:, 1:2], dtype="float32")  # space
        self.u0_sol = paddle.to_tensor(u0, dtype="float32")

        # Boundary Data
        self.t_lb_sol = paddle.to_tensor(X_lb[:, 0:1], dtype="float32")  # time -- lb
        self.t_ub_sol = paddle.to_tensor(X_ub[:, 0:1], dtype="float32")  # time -- ub
        self.x_lb_sol = paddle.to_tensor(X_lb[:, 1:2], dtype="float32")  # space -- lb
        self.x_ub_sol = paddle.to_tensor(X_ub[:, 1:2], dtype="float32")  # space -- ub

        # Collocation Points
        self.X_f_sol = paddle.to_tensor(X_f, dtype="float32")
        self.t_f_sol = paddle.to_tensor(X_f[:, 0:1], dtype="float32")  # time
        self.x_f_sol = paddle.to_tensor(X_f[:, 1:2], dtype="float32")  # space

        # supervision points
        self.has_suv_point = False
        if ts is not None and xs is not None and us is not None:
            self.t_s = paddle.to_tensor(ts, dtype="float32")
            self.x_s = paddle.to_tensor(xs, dtype="float32")
            self.u_s = paddle.to_tensor(us, dtype="float32")
            self.has_suv_point = True

    def mean_squared_error(self, error):
        mse_loss = paddle.nn.loss.MSELoss(reduction="sum")
        label = paddle.to_tensor(np.zeros(error.shape), dtype="float32")
        return mse_loss(error, label)

    def compile(self, optimizer="adam", lr=[0.01, 0.01, 0.01], loss="MSE", max_grad=2):
        self.opt_op = optimizer
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
        elif optimizer == "lbfgs":
            self.opt_idn = (
                paddle.optimizer.LBFGS(
                    # learning_rate=lr[0],
                    # tolerance_grad=1e-16,
                    # tolerance_change=0,
                    line_search_fn="strong_wolfe",
                    parameters=self.net_idn.parameters(),
                )
                if self.net_idn is not None
                else None
            )
            self.opt_pde = (
                paddle.optimizer.LBFGS(
                    # learning_rate=lr[1],
                    # tolerance_grad=1e-16,
                    # tolerance_change=0,
                    line_search_fn="strong_wolfe",
                    parameters=self.net_pde.parameters(),
                )
                if self.net_pde is not None
                else None
            )
            self.opt_sol = (
                paddle.optimizer.LBFGS(
                    # learning_rate=lr[2],
                    # tolerance_grad=1e-16,
                    # tolerance_change=0,
                    line_search_fn="strong_wolfe",
                    parameters=self.net_sol.parameters(),
                )
                if self.net_sol is not None
                else None
            )
        if loss == "MSE":
            self.loss_fn = self.mean_squared_error
        self.max_grad = max_grad

    def get_losses(self, mode):
        losses_list = []
        if mode is "idn":
            u_pred_idn = self.forward_net_u(
                self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
            )[0]
            # print("input:",float(self.t_idn[0]),float(self.x_idn[0]))
            # print("out,label:",float(u_pred_idn[0]),float(self.u_idn[0]))
            losses = self.loss_fn(u_pred_idn - self.u_idn)
        elif mode is "pde":
            pde_pred = self.forward_net_f(
                self.t_idn, self.x_idn, self.lb_idn, self.ub_idn, self.net_idn
            )
            losses = self.loss_fn(pde_pred)
        elif mode == "sol":
            u0_pred = self.forward_net_u(
                self.t0_sol, self.x0_sol, self.lb_sol, self.ub_sol, self.net_sol
            )[0]
            u_lb_pred_list = self.forward_net_u(
                self.t_lb_sol, self.x_lb_sol, self.lb_sol, self.ub_sol, self.net_sol
            )
            u_ub_pred_list = self.forward_net_u(
                self.t_ub_sol, self.x_ub_sol, self.lb_sol, self.ub_sol, self.net_sol
            )
            sol_f_pred = self.forward_net_f(
                self.t_f_sol, self.x_f_sol, self.lb_sol, self.ub_sol, self.net_sol
            )
            # print(self.t_f_sol[0].numpy())
            loss0 = self.loss_fn(sol_f_pred)
            loss1 = self.loss_fn(u0_pred - self.u0_sol)
            losses = loss0 + loss1
            # losses = loss1
            losses_list = [loss0.item(), loss1.item()]
            loss2 = 0
            for i in range(self.max_grad):
                loss_i = self.loss_fn(u_lb_pred_list[i] - u_ub_pred_list[i])
                # losses += loss_i
                # losses_list.append(loss_i.item())
                loss2 += loss_i
            losses_list.append(loss2.item())
            losses += loss2
            # if self.has_suv_point:
            #     u_suv = self.forward_net_u(
            #         self.t_s, self.x_s, self.lb_sol, self.ub_sol, self.net_sol
            #     )[0]
            #     loss_suv = self.loss_fn(u_suv - self.u_s)
            #     losses += loss_suv
            #     losses_list.append(loss_suv.item())
            # losses = loss0 + loss1
        return losses, losses_list

    def train_adam(self, N_iter, mode="idn"):
        """
        Args:
            N_iter (int): number of iteration
            mode (str): idn/pde/sol
        """
        if mode == "idn":
            opt = self.opt_idn
        elif mode == "pde":
            opt = self.opt_pde
        elif mode == "sol":
            opt = self.opt_sol

        start_time = time.time()
        for iter in range(1, N_iter+1):
            # with open('/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/params1.txt','w') as f:
            #     print("######### params",file=f)
            #     params = self.net_idn.state_dict()
            #     for k in params:
            #         print(params[k].numpy(),file=f)
            losses, losses_list = self.get_losses(mode)
            if iter == 1 or iter % 1000 == 0:
                elapsed = time.time() - start_time
                # print(
                #     "It: %d, Time: %.2f, Loss: %.3e" % (iter, elapsed, losses), end=""
                # )
                print(
                    "It:",iter," Loss:",float(losses), end=""
                )
                if mode == "sol":
                    for i in range(len(losses_list)):
                        print(", Loss%d: %.3e" % (i, losses_list[i]), end="")
                print("")
                start_time = time.time()

            losses.backward()

            # with open('/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/grad1.txt','w') as f:
            #     print("######### grad",file=f)
            #     for name, param in self.net_idn.named_parameters():
            #         print(param.grad.numpy(),file=f)
            #         # print(f"{k} {param.grad.mean().item():.10f}",file=f)
            #         np.save("/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/grad1.npy",param.grad.numpy())

            opt.step()

            # with open('/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/params_back1.txt','w') as f:
            #     print("######### params",file=f)
            #     params = self.net_idn.state_dict()
            #     for k in params:
            #         print(params[k].numpy(),file=f) 
            #         np.save("/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/params_back1.npy",params[k].numpy())

            opt.clear_grad()

            # with open('/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/params_back1.txt','w') as f:
            #     print("######### params",file=f)
            #     params = self.net_idn.state_dict()
            #     for k in params:
            #         print(params[k].numpy(),file=f) 
            #         np.save("/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/params_back1.npy",params[k].numpy())

        return losses

    def train_lbfgs(self, N_iter, mode="idn"):
        self.iter = 0

        def closure():
            loss, _ = self.get_losses(mode)
            opt.clear_grad()
            loss.backward()
            if self.iter % 1000 == 0:
                print("It: %d, Loss: %.3e" % (self.iter, loss))
                # for layer in self.net_sol.linears:
                #     w, b = layer.weight, layer.bias
                #     print("w,b :", w.mean(), b.mean())
            self.iter += 1
            return loss

        if mode == "idn":
            opt = self.opt_idn
        elif mode == "pde":
            opt = self.opt_pde
        elif mode == "sol":
            opt = self.opt_sol

        opt.max_iter = N_iter
        opt.max_eval = N_iter * 5 / 4
        loss = opt.step(closure)
        return loss

    def train(self, N_iter=10000, mode="idn"):
        if self.opt_op == "adam":
            return self.train_adam(N_iter, mode)
        elif self.opt_op == "lbfgs":
            return self.train_lbfgs(N_iter, mode)

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
        # lb.stop_gradient = True
        # ub.stop_gradient = True
        X = paddle.concat([t, x], axis=1)
        H = 2.0 * (X - lb) * paddle.pow((ub - lb), -1) - 1.0
        u = net.forward(H)
        res = [u]
        grad = u
        for _ in range(1, self.max_grad):
            new_grad = paddle.grad(grad, x, create_graph=True)[0]
            res.append(new_grad)
            grad = new_grad
        
        # with open('/home/lijialin03/workspaces/DeepHPMs/paddle_science/log/output1.txt','w') as f:
        #     print("######### output",file=f)
        #     print_out = u.reshape([100,100])
        #     for out in print_out.numpy():
        #         print(out,file=f)

        return res

    def forward_net_f(self, t, x, lb, ub, net):
        # t.stop_gradient = False
        # x.stop_gradient = False

        # print("#### t,x:", t[0].numpy(),x[0].numpy())
        u = self.forward_net_u(t, x, lb, ub, net)[0]
        u_t = paddle.grad(u, t, create_graph=True)[0]
        terms_list = [u]
        grad = u
        for _ in range(1, self.max_grad + 1):
            new_grad = paddle.grad(grad, x, create_graph=True)[0]
            terms_list.append(new_grad)
            grad = new_grad
        terms = paddle.concat(terms_list, 1)
        pde = self.forward_net_pde(terms)
        # print("u_t[0], pde[0]:", u_t[0].numpy(),pde[0].numpy())
        f = u_t - pde
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

    def load_model_list(self):
        path = "../../paddle_science/outs_kdv_adam/checkpoints/test"
        param_dict = paddle.load(path + ".pdparams")
        # for k in param_dict:
        #     print(k)
        # print(param_dict["model_list.0.linears.0.bias"])
        # optim_dict = paddle.load(path + ".pdopt")
        param_dict_idn = {}
        for k in self.net_idn.state_dict():
            # print(k)
            if k.split('.')[1] == "4":
                # param_dict_idn[k] = param_dict["last_fc."+k.split('.')[-1]]
                param_dict_idn[k] = param_dict["model_list.0.last_fc."+k.split('.')[-1]]
            else:
                param_dict_idn[k] = param_dict["model_list.0."+k]
                # param_dict_idn[k] = param_dict[k]
        self.net_idn.set_state_dict(param_dict_idn)

        param_dict_pde = {}
        for k in self.net_pde.state_dict():
            if k.split('.')[1] == "2":
                param_dict_pde[k] = param_dict["model_list.1.last_fc."+k.split('.')[-1]]
            else:
                param_dict_pde[k] = param_dict["model_list.1."+k]
        self.net_pde.set_state_dict(param_dict_pde)

        param_dict_sol = {}
        for k in self.net_sol.state_dict():
            if k.split('.')[1] == "4":
                param_dict_sol[k] = param_dict["model_list.2.last_fc."+k.split('.')[-1]]
            else:
                param_dict_sol[k] = param_dict["model_list.2."+k]
        self.net_sol.set_state_dict(param_dict_sol)
