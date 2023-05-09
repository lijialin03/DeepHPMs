import paddle
import numpy as np
from deephpm import DeepHPM
import scipy.io
from pyDOE import lhs

paddle.fluid.core.set_prim_eager_enabled(True)
paddle.device.set_device("gpu:2")


class Network(paddle.nn.Layer):
    """Network"""

    def __init__(self, layer_sizes):
        super().__init__()
        self.initializer_w = paddle.nn.initializer.XavierNormal()
        # self.initializer_w = paddle.nn.initializer.Constant(value=1.0)
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


def show_pred(it, model, mode="idn"):
    if mode == "idn":
        u_pred, _ = model.predict(model.t_idn, model.x_idn, mode)
        u_pred = np.squeeze(u_pred)
        print("It: %d, u_pred[:10]:" % (it))
        print(u_pred[:10])
    elif mode == "pde":
        _, f_pred = model.predict(model.t_idn, model.x_idn, mode)
        f_pred = np.squeeze(f_pred)
        print("It: %d, f_pred[:10]:" % (it))
        print(f_pred[:10])
    else:
        u_pred, _ = model.predict(model.t0_sol, model.x0_sol, mode)
        _, f_pred = model.predict(model.t_f_sol, model.x_f_sol, mode)
        u_pred = np.squeeze(u_pred)
        print("It: %d, u_pred[:10]:" % (it))
        print(u_pred[:10])
        f_pred = np.squeeze(f_pred)
        print("It: %d, f_pred[:10]:" % (it))
        print(f_pred[:10])


def train_adam(model, N_iter, mode="idn"):
    """
    Args:
        N_iter (int): number of iteration
        mode (str): idn/pde/sol
    """
    if mode == "idn":
        opt = model.opt_idn
    elif mode == "pde":
        opt = model.opt_pde
    elif mode == "sol":
        opt = model.opt_sol
    print(f"\n———————————————— w, b for mode: {mode} ————————————————\n")
    for iter in range(N_iter):
        losses, _ = model.get_losses(mode)

        if iter % 1000 == 0 or iter >= N_iter - 10:
            show_pred(iter, model, mode)

        losses.backward()
        opt.step()
        opt.clear_grad()

    return losses


def load_init_wb(net, mode):
    with open("../../params_w_" + mode + ".txt", "r") as f:
        w_tmp = []
        w = []
        for line in f.readlines():
            if "layer " not in line:
                line_str = line.strip().split(",")[:-1]
                tmp = []
                for s in line_str:
                    tmp.append(float(s))
                w_tmp.append(tmp)
            else:
                w.append(w_tmp.copy())
                w_tmp = []
        w.append(w_tmp.copy())
        w = w[1:]

    with open("../../params_b_" + mode + ".txt", "r") as f:
        b_tmp = []
        b = []
        for line in f.readlines():
            if "layer " not in line:
                line_str = line.strip().split(",")[:-1]
                tmp = []
                for s in line_str:
                    tmp.append(float(s))
                b_tmp.append(tmp)
            else:
                b.append(b_tmp.copy())
                b_tmp = []
        b.append(b_tmp.copy())
        b = b[1:]

    for i in range(len(net.linears)):
        w_i = np.array(w[i]).astype("float32")
        b_i = np.array(b[i][0]).astype("float32")
        net.linears[i].weight = paddle.create_parameter(
            shape=w_i.shape,
            dtype=w_i.dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(w_i)),
        )
        net.linears[i].bias = paddle.create_parameter(
            shape=b_i.shape,
            dtype=b_i.dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(b_i)),
        )


data_idn = scipy.io.loadmat("../../Data/KS.mat")

t_idn = data_idn["t"].flatten()[:, None]
x_idn = data_idn["x"].flatten()[:, None]
Exact_idn = np.real(data_idn["usol"])

T_idn, X_idn = np.meshgrid(t_idn, x_idn)

t_idn_star = T_idn.flatten()[:, None]
x_idn_star = X_idn.flatten()[:, None]
u_idn_star = Exact_idn.flatten()[:, None]

idx = 10
t_train = t_idn_star[:idx, :]
x_train = x_idn_star[:idx, :]
u_train = u_idn_star[:idx, :]

#

data_sol = scipy.io.loadmat("../../Data/KS.mat")

t_sol = data_sol["t"].flatten()[:, None]
x_sol = data_sol["x"].flatten()[:, None]
Exact_sol = np.real(data_sol["usol"])

T_sol, X_sol = np.meshgrid(t_sol, x_sol)

t_sol_star = T_sol.flatten()[:, None]
x_sol_star = X_sol.flatten()[:, None]
X_sol_star = np.hstack((t_sol_star, x_sol_star))
u_sol_star = Exact_sol.flatten()[:, None]

N_f = 20

x0_train = x_sol
u0_train = Exact_sol[:, 0:1]

tb_train = t_sol

lb_sol = np.array([0.0, -10.0])
ub_sol = np.array([50.0, 10.0])
# X_f_train = lb_sol + (ub_sol - lb_sol) * lhs(2, N_f)
# print(X_f_train)
with open("../../X_f_train_forward.txt", "r") as f:
    X_f_train = []
    for line in f.readlines():
        line_str = line.strip().split(",")[:-1]
        tmp = []
        for s in line_str:
            tmp.append(float(s))
        X_f_train.append(tmp)
    f.close()
X_f_train = np.reshape(X_f_train, (N_f, 2))

lb = [0.0, -10.0]
ub = [50.0, 10.0]

model = DeepHPM()
net_idn = Network([2, 50, 50, 50, 50, 1])
load_init_wb(net_idn, "idn")
model.init_idn(net_idn, t_train, x_train, u_train, lb, ub)

net_pde = Network([5, 100, 100, 1])
load_init_wb(net_pde, "pde")
model.init_pde(net_pde)

net_sol = Network([2, 50, 50, 50, 50, 1])
load_init_wb(net_sol, "sol")
model.init_sol(
    net_sol,
    tb_train,
    x0_train,
    u0_train,
    lb,
    ub,
    X_f_train,
)

model.compile(optimizer="adam", lr=[1e-3, 1e-3, 1e-3], max_grad=4)

N_iter = 1
train_adam(model, N_iter, mode="idn")
paddle.fluid.core.set_prim_eager_enabled(True)
train_adam(model, N_iter, mode="pde")
# train_adam(model, N_iter, mode="sol")
