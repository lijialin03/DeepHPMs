import imp
import numpy as np
import scipy.io
from scipy.stats import qmc
from pyDOE import lhs
from paddlepaddle.scripts.config import Config


class DataLoader(object):
    def __init__(self) -> None:
        self.file_idn = None
        self.file_sol = None
        self.cfg = Config()
        self.lb_idn = self.cfg.lb
        self.ub_idn = self.cfg.ub
        self.lb_sol = self.cfg.lb
        self.ub_sol = self.cfg.ub

    def load_idn(self):
        """Load identification data"""
        data_idn = scipy.io.loadmat(self.file_idn)

        self.t_idn_star = data_idn["t_train"].flatten()[:, None]
        self.x_idn_star = data_idn["x_train"].flatten()[:, None]
        self.X_idn_star = np.hstack((self.t_idn_star, self.x_idn_star))
        self.u_idn_star = data_idn["u_train"]

        self.t_train = self.t_idn_star
        self.x_train = self.x_idn_star
        self.u_train = self.u_idn_star

    def load_sol(self):
        """Load solution data"""
        data_sol = scipy.io.loadmat(self.file_sol)
        t_sol = data_sol["t_ori"]
        x_sol = data_sol["x_ori"]
        self.Exact_sol = data_sol["Exact_ori"]

        self.T_sol, self.X_sol = np.meshgrid(t_sol, x_sol)

        self.t_sol_star = self.T_sol.flatten()[:, None]
        self.x_sol_star = self.X_sol.flatten()[:, None]
        self.X_sol_star = np.hstack((self.t_sol_star, self.x_sol_star))
        self.u_sol_star = data_sol["u_ori"]

        self.x0_train = data_sol["x0"]
        self.u0_train = data_sol["u0"]

        len_half = int(len(data_sol["tb"]) / 2)
        self.tb_train = data_sol["tb"][:len_half]

        t_f_train = data_sol["t_f_train"]
        x_f_train = data_sol["x_f_train"]
        self.X_f_train = np.hstack((t_f_train, x_f_train))

    def __call__(self, file_idn, file_sol):
        # set config
        self.file_idn = file_idn
        self.file_sol = file_sol

        # load data
        self.load_idn()
        self.load_sol()
