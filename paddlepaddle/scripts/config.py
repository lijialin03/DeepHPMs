import numpy as np


class Config(object):
    def __init__(self, num_var=1) -> None:
        self.num_var = num_var
        if self.num_var == 1:
            # Burgers
            # self.lb = [0.0, -8.0]
            # self.ub = [10.0, 8.0]
            # KdV
            self.lb = [0.0, -20.0]
            self.ub = [40.0, 20.0]
            self.keep = 2 / 3
            # KS
            # self.lb = [0.0, -10.0]
            # self.ub = [50.0, 10.0]
            # KS_chaotic
            # self.lb = [0.0, 0.0]
            # self.ub = [100.0, 32.0 * np.pi]
            # self.keep = 1
        elif self.num_var == 2:
            # 2 vars
            self.lb = [0.0, -5.0]
            self.ub = [np.pi / 2, 5.0]
            self.keep = 1

        self.N_train = 10000
        self.N_f = 20000
        self.noise = 0.00
        self.suv_ratio = 0.00

        self.layers_idn = [2, 50, 50, 50, 50, 1]
        self.layers_pde = [3 * self.num_var, 100, 100, 1]
        self.layers_sol = [2, 50, 50, 50, 50, 1]
