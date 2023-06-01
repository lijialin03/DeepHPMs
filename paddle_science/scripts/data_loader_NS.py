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

    def load(self):
        """Load identification data"""
        data = scipy.io.loadmat(self.file_idn)

        self.X_data = data["X_star"]
        self.w_star = data["w_star"]
        self.w_data = data["w_data"]
        self.x_star = data["x_star"]
        self.y_star = data["y_star"]
        self.t_star = data["t_star"]
        self.u_star = data["u_star"]
        self.v_star = data["v_star"]

    def __call__(self, file_idn, file_sol):
        # set config
        self.file_idn = file_idn
        self.file_sol = file_sol

        # load data
        self.load()
