"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import time
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

###############################################################################
############################## Helper Functions ###############################
###############################################################################


def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = xavier_init(size=[layers[l], layers[l + 1]])
        # W = tf.Variable(
        #     tf.ones([layers[l], layers[l + 1]], dtype=tf.float32), dtype=tf.float32
        # )
        b = tf.Variable(
            tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32
        )
        weights.append(W)
        biases.append(b)
    return weights, biases


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(
        tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
        dtype=tf.float32,
    )


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


###############################################################################
################################ DeepHPM Class ################################
###############################################################################


class DeepHPM:
    def __init__(
        self,
        t,
        x,
        u,
        x0,
        u0,
        tb,
        X_f,
        u_layers,
        pde_layers,
        layers,
        lb_idn,
        ub_idn,
        lb_sol,
        ub_sol,
    ):
        self.step = 0
        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn

        self.lb_sol = lb_sol
        self.ub_sol = ub_sol

        # Init for Identification
        self.idn_init(t, x, u, u_layers, pde_layers)

        # Init for Solution
        self.sol_init(x0, u0, tb, X_f, layers)

        # tf session
        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        )

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def show_w_b_loss(self, it, w, b, loss):
        print("It: %d, Loss: %.3e" % (it, loss))
        # print(len(w), len(b))
        for i in range(len(w)):
            print("layer:", i, " w:", w[i][0][0], " b:", b[i][0][0])

    def save_w_b(self, saver, w, b, mode):
        saver.save(self.sess, "../tf_params/params_" + mode)
        with open("../params_w_" + mode + ".txt", "w") as f:
            for i in range(len(w)):
                f.write("layer " + str(i) + "\n")
                for j in range(len(w[i])):
                    for k in range(len(w[i][j])):
                        f.write(str(w[i][j][k]))
                        f.write(",")
                    f.write("\n")
            f.close()
        with open("../params_b_" + mode + ".txt", "w") as f:
            for i in range(len(b)):
                f.write("layer " + str(i) + "\n")
                for j in range(len(b[i])):
                    for k in range(len(b[i][j])):
                        f.write(str(b[i][j][k]))
                        f.write(",")
                    f.write("\n")
            f.close()

    def load_w_b(self, saver, mode):
        saver.restore(self.sess, "../tf_params/params_" + mode)

    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################

    def idn_init(self, t, x, u, u_layers, pde_layers):
        # Training Data for Identification
        self.t = t
        self.x = x
        self.u = u

        # Layers for Identification
        self.u_layers = u_layers
        self.pde_layers = pde_layers

        # Initialize NNs for Identification
        self.u_weights, self.u_biases = initialize_NN(u_layers)
        self.pde_weights, self.pde_biases = initialize_NN(pde_layers)

        # tf placeholders for Identification
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.terms_tf = tf.placeholder(tf.float32, shape=[None, pde_layers[0]])

        # tf graphs for Identification
        self.idn_u_pred = self.idn_net_u(self.t_tf, self.x_tf)
        self.pde_pred = self.net_pde(self.terms_tf)
        self.idn_f_pred = self.idn_net_f(self.t_tf, self.x_tf)

        # loss for Identification
        self.idn_u_loss = tf.reduce_sum(tf.square(self.idn_u_pred - self.u_tf))
        self.idn_f_loss = tf.reduce_sum(tf.square(self.idn_f_pred))

        # Optimizer for Identification
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.idn_u_loss,
            var_list=self.u_weights + self.u_biases,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps,
            },
        )

        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.idn_f_loss,
            var_list=self.pde_weights + self.pde_biases,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps,
            },
        )

        self.idn_u_optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.idn_u_train_op_Adam = self.idn_u_optimizer_Adam.minimize(
            self.idn_u_loss, var_list=self.u_weights + self.u_biases
        )

        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(
            self.idn_f_loss, var_list=self.pde_weights + self.pde_biases
        )

    def idn_net_u(self, t, x):
        X = tf.concat([t, x], 1)
        H = 2.0 * (X - self.lb_idn) / (self.ub_idn - self.lb_idn) - 1.0
        u = neural_net(H, self.u_weights, self.u_biases)
        return u

    def net_pde(self, terms):
        pde = neural_net(terms, self.pde_weights, self.pde_biases)
        return pde

    def idn_net_f(self, t, x):
        u = self.idn_net_u(t, x)

        u_t = tf.gradients(u, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]

        terms = tf.concat([u, u_x, u_xx, u_xxx, u_xxxx], 1)

        f = u_t - self.net_pde(terms)

        return f

    def idn_u_train(self, N_iter):
        print(f"\n———————————————— w, b for mode: idn ————————————————\n")
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.u_tf: self.u}
        saver_u = tf.train.Saver(var_list=self.u_weights + self.u_biases)

        self.load_w_b(saver_u, "idn")

        for it in range(N_iter):
            if it == 0:
                w, b = self.sess.run([self.u_weights, self.u_biases], tf_dict)
                self.save_w_b(saver_u, w, b, "idn")

            if it % 1000 == 0 or it >= N_iter - 10:
                w, b, loss = self.sess.run(
                    [self.u_weights, self.u_biases, self.idn_u_loss], tf_dict
                )
                self.show_w_b_loss(it, w, b, loss)

            self.sess.run(self.idn_u_train_op_Adam, tf_dict)

        w, b, loss = self.sess.run(
            [self.u_weights, self.u_biases, self.idn_u_loss], tf_dict
        )
        self.show_w_b_loss(N_iter, w, b, loss)

    def idn_f_train(self, N_iter):
        print(f"\n———————————————— w, b for mode: pde ————————————————\n")
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x}
        saver_f = tf.train.Saver(var_list=self.pde_weights + self.pde_biases)

        self.load_w_b(saver_f, "pde")

        for it in range(N_iter):
            if it == 0:
                w, b = self.sess.run([self.pde_weights, self.pde_biases], tf_dict)
                self.save_w_b(saver_f, w, b, "pde")

            if it % 1000 == 0 or it >= N_iter - 10:
                w, b, loss = self.sess.run(
                    [self.pde_weights, self.pde_biases, self.idn_f_loss], tf_dict
                )
                self.show_w_b_loss(it, w, b, loss)

            self.sess.run(self.idn_f_train_op_Adam, tf_dict)

        w, b, loss = self.sess.run(
            [self.pde_weights, self.pde_biases, self.idn_f_loss], tf_dict
        )
        self.show_w_b_loss(N_iter, w, b, loss)

    def idn_predict(self, t_star, x_star):
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}

        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        f_star = self.sess.run(self.idn_f_pred, tf_dict)

        return u_star, f_star

    def predict_pde(self, terms_star):
        tf_dict = {self.terms_tf: terms_star}

        pde_star = self.sess.run(self.pde_pred, tf_dict)

        return pde_star

    ###########################################################################
    ############################### Solver ####################################
    ###########################################################################

    def sol_init(self, x0, u0, tb, X_f, layers):
        # Training Data for Solution
        X0 = np.concatenate((0 * x0, x0), 1)  # (0, x0)
        X_lb = np.concatenate((tb, 0 * tb + self.lb_sol[1]), 1)  # (tb, lb[1])
        X_ub = np.concatenate((tb, 0 * tb + self.ub_sol[1]), 1)  # (tb, ub[1])

        self.X_f = X_f  # Collocation Points
        self.t0 = X0[:, 0:1]  # Initial Data (time)
        self.x0 = X0[:, 1:2]  # Initial Data (space)
        self.t_lb = X_lb[:, 0:1]  # Boundary Data (time) -- lower boundary
        self.x_lb = X_lb[:, 1:2]  # Boundary Data (space) -- lower boundary
        self.t_ub = X_ub[:, 0:1]  # Boundary Data (time) -- upper boundary
        self.x_ub = X_ub[:, 1:2]  # Boundary Data (space) -- upper boundary
        self.t_f = X_f[:, 0:1]  # Collocation Points (time)
        self.x_f = X_f[:, 1:2]  # Collocation Points (space)
        self.u0 = u0  # Boundary Data

        # Layers for Solution
        self.layers = layers

        # Initialize NNs for Solution
        self.weights, self.biases = initialize_NN(layers)

        # tf placeholders for Solution
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # tf graphs for Solution
        self.u0_pred, _, _, _ = self.sol_net_u(self.t0_tf, self.x0_tf)
        (
            self.u_lb_pred,
            self.u_x_lb_pred,
            self.u_xx_lb_pred,
            self.u_xxx_lb_pred,
        ) = self.sol_net_u(self.t_lb_tf, self.x_lb_tf)
        (
            self.u_ub_pred,
            self.u_x_ub_pred,
            self.u_xx_ub_pred,
            self.u_xxx_ub_pred,
        ) = self.sol_net_u(self.t_ub_tf, self.x_ub_tf)
        self.sol_f_pred = self.sol_net_f(self.t_f_tf, self.x_f_tf)

        # loss for Solution
        self.sol_loss = (
            tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred))
            + tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred))
            + tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred))
            + tf.reduce_mean(tf.square(self.u_xx_lb_pred - self.u_xx_ub_pred))
            + tf.reduce_mean(tf.square(self.u_xxx_lb_pred - self.u_xxx_ub_pred))
            + tf.reduce_mean(tf.square(self.sol_f_pred))
        )

        # Optimizer for Solution
        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.sol_loss,
            var_list=self.weights + self.biases,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps,
            },
        )

        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(
            self.sol_loss, var_list=self.weights + self.biases
        )

    def sol_net_u(self, t, x):
        X = tf.concat([t, x], 1)
        H = 2.0 * (X - self.lb_sol) / (self.ub_sol - self.lb_sol) - 1.0
        u = neural_net(H, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        return u, u_x, u_xx, u_xxx

    def sol_net_f(self, t, x):
        u, u_x, u_xx, u_xxx = self.sol_net_u(t, x)

        u_t = tf.gradients(u, t)[0]

        u_xxxx = tf.gradients(u_xxx, x)[0]

        terms = tf.concat([u, u_x, u_xx, u_xxx, u_xxxx], 1)

        f = u_t - self.net_pde(terms)

        return f

    def callback(self, loss):
        if self.step % 1000 == 0:
            print("It: %d, Loss: %e" % (self.step, loss))
        self.step += 1

    def sol_train(self, N_iter):
        print(f"\n———————————————— w, b for mode: sol ————————————————\n")
        tf_dict = {
            self.t0_tf: self.t0,
            self.x0_tf: self.x0,
            self.u0_tf: self.u0,
            self.t_lb_tf: self.t_lb,
            self.x_lb_tf: self.x_lb,
            self.t_ub_tf: self.t_ub,
            self.x_ub_tf: self.x_ub,
            self.t_f_tf: self.t_f,
            self.x_f_tf: self.x_f,
        }

        saver = tf.train.Saver(var_list=self.weights + self.biases)

        self.load_w_b(saver, "sol")

        for it in range(N_iter):
            if it == 0:
                w, b = self.sess.run([self.weights, self.biases], tf_dict)
                self.save_w_b(saver, w, b, "sol")

            if it % 1000 == 0 or it >= N_iter - 10:
                w, b, loss = self.sess.run(
                    [self.weights, self.biases, self.sol_loss], tf_dict
                )
                self.show_w_b_loss(it, w, b, loss)

            self.sess.run(self.sol_train_op_Adam, tf_dict)

        w, b, loss = self.sess.run([self.weights, self.biases, self.sol_loss], tf_dict)
        self.show_w_b_loss(N_iter, w, b, loss)

    def sol_predict(self, t_star, x_star):
        u_star = self.sess.run(self.u0_pred, {self.t0_tf: t_star, self.x0_tf: x_star})
        f_star = self.sess.run(
            self.sol_f_pred, {self.t_f_tf: t_star, self.x_f_tf: x_star}
        )

        return u_star, f_star


###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__":
    # Doman bounds
    lb_idn = np.array([0.0, -10.0])
    ub_idn = np.array([50.0, 10.0])

    lb_sol = np.array([0.0, -10.0])
    ub_sol = np.array([50.0, 10.0])

    ### Load Data ###

    data_idn = scipy.io.loadmat("../Data/KS.mat")

    t_idn = data_idn["t"].flatten()[:, None]
    x_idn = data_idn["x"].flatten()[:, None]
    Exact_idn = np.real(data_idn["usol"])

    T_idn, X_idn = np.meshgrid(t_idn, x_idn)

    # keep = 1
    # index = int(keep * t_idn.shape[0])
    # T_idn = T_idn[:, 0:index]
    # X_idn = X_idn[:, 0:index]
    # Exact_idn = Exact_idn[:, 0:index]

    t_idn_star = T_idn.flatten()[:, None]
    x_idn_star = X_idn.flatten()[:, None]
    X_idn_star = np.hstack((t_idn_star, x_idn_star))
    u_idn_star = Exact_idn.flatten()[:, None]

    #

    data_sol = scipy.io.loadmat("../Data/KS.mat")

    t_sol = data_sol["t"].flatten()[:, None]
    x_sol = data_sol["x"].flatten()[:, None]
    Exact_sol = np.real(data_sol["usol"])

    T_sol, X_sol = np.meshgrid(t_sol, x_sol)

    t_sol_star = T_sol.flatten()[:, None]
    x_sol_star = X_sol.flatten()[:, None]
    X_sol_star = np.hstack((t_sol_star, x_sol_star))
    u_sol_star = Exact_sol.flatten()[:, None]

    ### Training Data ###

    # For identification
    # N_train = 10000

    # idx = np.random.choice(t_idn_star.shape[0], N_train, replace=False)
    idx = 10000
    t_train = t_idn_star[:idx, :]
    x_train = x_idn_star[:idx, :]
    u_train = u_idn_star[:idx, :]

    # For solution
    N0 = Exact_sol.shape[0]
    N_b = Exact_sol.shape[1]
    N_f = 20000

    # idx_x = np.random.choice(x_sol.shape[0], N0, replace=False)
    x0_train = x_sol
    u0_train = Exact_sol[:, 0:1]

    # idx_t = np.random.choice(t_sol.shape[0], N_b, replace=False)
    tb_train = t_sol

    with open("../X_f_train.txt", "r") as f:
        X_f_train = []
        for line in f.readlines():
            line_str = line.strip().split(",")[:-1]
            tmp = []
            for s in line_str:
                tmp.append(float(s))
            X_f_train.append(tmp)
        f.close()
    X_f_train = np.reshape(X_f_train, (N_f, 2))

    # X_f_train = lb_sol + (ub_sol - lb_sol) * lhs(2, N_f)

    with open("../X_f_train.txt", "w") as f:
        for i in range(len(X_f_train)):
            for j in range(len(X_f_train[0])):
                f.write(str(X_f_train[i][j]))
                f.write(",")
            f.write("\n")
        f.close()

    # Layers
    u_layers = [2, 50, 50, 50, 50, 1]
    pde_layers = [5, 100, 100, 1]

    layers = [2, 50, 50, 50, 50, 1]

    # Model
    model = DeepHPM(
        t_train,
        x_train,
        u_train,
        x0_train,
        u0_train,
        tb_train,
        X_f_train,
        u_layers,
        pde_layers,
        layers,
        lb_idn,
        ub_idn,
        lb_sol,
        ub_sol,
    )

    N_it = 10
    # Train the identifier
    model.idn_u_train(N_iter=N_it)

    model.idn_f_train(N_iter=N_it)

    ### Solution ###

    # Train the solver
    model.sol_train(N_iter=N_it)
