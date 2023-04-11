class Config(object):
    def __init__(self) -> None:
        self.lb = [0.0, -8.0]
        self.ub = [10.0, 8.0]
        self.keep = 2 / 3
        self.N_train = 10000
        self.N_f = 20000
        self.noise = 0.00

        self.layers_idn = [2, 50, 50, 50, 50, 1]
        self.layers_pde = [3, 100, 100, 1]
        self.layers_sol = [2, 50, 50, 50, 50, 1]
