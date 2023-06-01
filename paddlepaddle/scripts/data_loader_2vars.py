import imp
import numpy as np
import scipy.io
from scipy.stats import qmc
from config import Config


class DataLoader(object):
    def __init__(self, num_var) -> None:
        self.file_idn = None
        self.file_sol = None
        self.cfg = Config(num_var)

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
        self.Exact_u_sol = None

        if self.cfg.num_var == 2:
            self.v_idn_star = None
            self.v_sol_star = None
            self.v_train = None
            self.v0_train = None
            self.Exact_v_sol = None
            self.Exact_uv_sol = None

        # middle vars
        self.t_sol = None
        self.x_sol = None

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
        T_idn, X_idn = np.meshgrid(t_idn, x_idn)
        index = int(self.cfg.keep * t_idn.shape[0])
        T_idn = T_idn[:, 0:index]
        X_idn = X_idn[:, 0:index]

        self.t_idn_star = T_idn.flatten()[:, None]
        self.x_idn_star = X_idn.flatten()[:, None]
        self.X_idn_star = np.hstack((self.t_idn_star, self.x_idn_star))

        Exact_u_idn = np.real(data_idn["usol"])
        Exact_u_idn = Exact_u_idn[:, 0:index]
        self.u_idn_star = Exact_u_idn.flatten()[:, None]
        if self.cfg.num_var == 2:
            Exact_v_idn = np.imag(data_idn["usol"])
            Exact_v_idn = Exact_v_idn[:, 0:index]
            # Exact_uv_idn = np.sqrt(Exact_u_idn**2 + Exact_v_idn**2)
            self.v_idn_star = Exact_v_idn.flatten()[:, None]

    def load_sol(self):
        """Load solution data"""
        data_sol = scipy.io.loadmat(self.file_sol)

        self.t_sol = data_sol["t"].flatten()[:, None]
        self.x_sol = data_sol["x"].flatten()[:, None]
        self.T_sol, self.X_sol = np.meshgrid(self.t_sol, self.x_sol)

        self.t_sol_star = self.T_sol.flatten()[:, None]
        self.x_sol_star = self.X_sol.flatten()[:, None]
        self.X_sol_star = np.hstack((self.t_sol_star, self.x_sol_star))

        self.Exact_u_sol = np.real(data_sol["usol"])
        self.u_sol_star = self.Exact_u_sol.flatten()[:, None]
        if self.cfg.num_var == 2:
            self.Exact_v_sol = np.imag(data_sol["usol"])
            self.Exact_uv_sol = np.sqrt(self.Exact_u_sol**2 + self.Exact_v_sol**2)
            self.v_sol_star = self.Exact_v_sol.flatten()[:, None]

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
        N0 = self.Exact_u_sol.shape[0]
        N_b = self.Exact_u_sol.shape[1]

        idx_x = np.random.choice(self.x_sol.shape[0], N0, replace=False)
        self.x0_train = self.x_sol[idx_x, :]
        self.u0_train = self.Exact_u_sol[idx_x, 0:1]

        if self.cfg.num_var == 2:
            # For identification
            self.v_train = self.v_idn_star[idx, :]
            self.v_train = self.v_train + self.cfg.noise * np.std(
                self.v_train
            ) * np.random.randn(self.v_train.shape[0], self.v_train.shape[1])
            # For solution
            self.v0_train = self.Exact_v_sol[idx_x, 0:1]

        idx_t = np.random.choice(self.t_sol.shape[0], N_b, replace=False)
        self.tb_train = self.t_sol[idx_t, :]

        # self.X_f_train = self.lb_sol + (self.ub_sol - self.lb_sol) * lhs(2, self.cfg.N_f)
        sampler = qmc.LatinHypercube(d=2, seed=1)
        sample = sampler.random(self.cfg.N_f)
        self.X_f_train = qmc.scale(sample, self.lb_sol, self.ub_sol)

    def __call__(self, file_idn, file_sol):
        # set config
        self.file_idn = file_idn
        self.file_sol = file_sol

        # load data
        self.set_bounds()
        self.load_idn()
        self.load_sol()
        self.prepare_training_data()
