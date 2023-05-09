import imp
import numpy as np
import scipy.io
from scipy.stats import qmc
from pyDOE import lhs
from config import Config


class DataLoader(object):
    def __init__(self) -> None:
        self.file_idn = None
        self.file_sol = None
        self.cfg = None

        # outputs
        self.lb_idn = None
        self.ub_idn = None
        self.lb_sol = None
        self.ub_sol = None

        self.t_idn_star = None
        self.x_idn_star = None
        self.X_idn_star = None
        self.u_idn_star = None

        self.t_sol_star = None
        self.x_sol_star = None
        self.X_sol_star = None
        self.u_sol_star = None

        self.t_train = None
        self.x_train = None
        self.u_train = None
        self.x0_train = None
        self.u0_train = None
        self.tb_train = None
        self.X_f_train = None

        self.T_sol = None
        self.X_sol = None
        self.Exact_sol = None

        # middle vars
        self.t_sol = None
        self.x_sol = None

        # debug
        self.T_idn_d = None
        self.X_idn_d = None
        self.t_idn_debug = None
        self.x_idn_debug = None
        self.X_idn_debug = None
        self.u_idn_debug = None
        self.Exact_idn_debug = None

    def set_bounds(self):
        """Doman bounds"""
        self.lb_idn = np.array(self.cfg.lb)
        self.ub_idn = np.array(self.cfg.ub)

        self.lb_sol = np.array(self.cfg.lb)
        self.ub_sol = np.array(self.cfg.ub)

    def load_idn(self):
        """Load identification data"""
        data_idn = scipy.io.loadmat(self.file_idn)

        t_idn = data_idn["t"].flatten()[:, None]
        x_idn = data_idn["x"].flatten()[:, None]
        Exact_idn = np.real(data_idn["usol"])

        T_idn, X_idn = np.meshgrid(t_idn, x_idn)

        index = int(self.cfg.keep * t_idn.shape[0])
        T_idn = T_idn[:, 0:index]
        X_idn = X_idn[:, 0:index]
        Exact_idn_star = Exact_idn[:, 0:index]

        self.t_idn_star = T_idn.flatten()[:, None]
        self.x_idn_star = X_idn.flatten()[:, None]
        self.X_idn_star = np.hstack((self.t_idn_star, self.x_idn_star))
        self.u_idn_star = Exact_idn_star.flatten()[:, None]

        # debug idn data
        self.T_idn_d, self.X_idn_d = np.meshgrid(t_idn, x_idn)
        self.t_idn_debug = self.T_idn_d.flatten()[:, None]
        self.x_idn_debug = self.X_idn_d.flatten()[:, None]
        self.X_idn_debug = np.hstack((self.t_idn_debug, self.x_idn_debug))
        self.u_idn_debug = Exact_idn.flatten()[:, None]
        self.Exact_idn_debug = Exact_idn

    def load_sol(self):
        """Load solution data"""
        data_sol = scipy.io.loadmat(self.file_sol)

        self.t_sol = data_sol["t"].flatten()[:, None]
        self.x_sol = data_sol["x"].flatten()[:, None]
        self.Exact_sol = np.real(data_sol["usol"])

        self.T_sol, self.X_sol = np.meshgrid(self.t_sol, self.x_sol)

        self.t_sol_star = self.T_sol.flatten()[:, None]
        self.x_sol_star = self.X_sol.flatten()[:, None]
        self.X_sol_star = np.hstack((self.t_sol_star, self.x_sol_star))
        self.u_sol_star = self.Exact_sol.flatten()[:, None]

    def prepare_training_data(self):
        """Prepare training and validation data"""
        # For identification
        idx = np.random.choice(
            self.t_idn_star.shape[0], self.cfg.N_train, replace=False
        )
        self.t_train = self.t_idn_star[idx, :]
        self.x_train = self.x_idn_star[idx, :]
        self.u_train = self.u_idn_star[idx, :]

        self.u_train = self.u_train + self.cfg.noise * np.std(
            self.u_train
        ) * np.random.randn(self.u_train.shape[0], self.u_train.shape[1])

        # For solution
        N0 = self.Exact_sol.shape[0]
        N_b = self.Exact_sol.shape[1]

        idx_x = np.random.choice(self.x_sol.shape[0], N0, replace=False)
        self.x0_train = self.x_sol[idx_x, :]
        self.u0_train = self.Exact_sol[idx_x, 0:1]

        idx_t = np.random.choice(self.t_sol.shape[0], N_b, replace=False)
        self.tb_train = self.t_sol[idx_t, :]

        # self.X_f_train = self.lb_sol + (self.ub_sol - self.lb_sol) * lhs(
        #     2, self.cfg.N_f
        # )
        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(self.cfg.N_f)
        self.X_f_train = qmc.scale(sample, self.lb_sol, self.ub_sol)
        # print(self.X_f_train[:, 0:1])
        # print(self.X_f_train[:, 1:2])

        # Add some supervision point
        idx_s = np.random.choice(
            self.t_sol_star.shape[0], int(N0 * N_b * self.cfg.suv_ratio), replace=False
        )
        self.t_suv = self.t_sol_star[idx_s, :]
        self.x_suv = self.x_sol_star[idx_s, :]
        self.u_suv = self.u_sol_star[idx_s, :]

    def __call__(self, file_idn, file_sol):
        # set config
        self.file_idn = file_idn
        self.file_sol = file_sol
        self.cfg = Config()

        # load data
        self.set_bounds()
        self.load_idn()
        self.load_sol()
        self.prepare_training_data()
