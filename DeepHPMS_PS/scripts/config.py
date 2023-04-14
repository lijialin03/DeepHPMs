class Config(object):
    def __init__(self, num_var=1) -> None:
        self.num_var = num_var
        if self.num_var == 1:
            self.lb = [0.0, -8.0]
            self.ub = [10.0, 8.0]
            self.keep = 2 / 3
        elif self.num_var == 2:
            # 2 vars
            import numpy as np

            self.lb = [0.0, -5.0]
            self.ub = [np.pi / 2, 5.0]
            self.keep = 1

        self.N_train = 10000
        self.N_f = 20000
        self.noise = 0.00

        self.layers_idn = [2, 50, 50, 50, 50, 1]
        self.layers_pde = [3 * self.num_var, 100, 100, 1]
        self.layers_sol = [2, 50, 50, 50, 50, 1]
