import numpy as np
import scipy.io
from scipy.stats import qmc


class DataLoader(object):
    def __init__(self) -> None:
        np.random.seed(1)
        self.data_trans = {}

    def trans_data(self, data):
        t_ori = data["t"].flatten()[:, None]
        x_ori = data["x"].flatten()[:, None]
        Exact_u_ori = np.real(data["usol"])
        Exact_v_ori = np.imag(data["usol"])
        self.t_ori = t_ori
        self.x_ori = x_ori
        self.u_ori = Exact_u_ori.flatten()[:, None]
        self.v_ori = Exact_v_ori.flatten()[:, None]
        self.Exact_u_ori = Exact_u_ori
        self.Exact_v_ori = Exact_v_ori
        self.Exact_uv_ori = np.sqrt(Exact_u_ori**2 + Exact_v_ori**2)

        T, X = np.meshgrid(t_ori, x_ori)

        index = int(self.keep * t_ori.shape[0])
        T = T[:, 0:index]
        X = X[:, 0:index]
        Exact_u_train = Exact_u_ori[:, 0:index]
        Exact_v_train = Exact_v_ori[:, 0:index]
        Exact_uv_train = np.sqrt(Exact_u_train**2 + Exact_v_train**2)

        self.t_star = T.flatten()[:, None]
        self.x_star = X.flatten()[:, None]
        self.X_star = np.hstack((self.t_star, self.x_star))
        self.u_star = Exact_u_train.flatten()[:, None]
        self.v_star = Exact_v_train.flatten()[:, None]

        self.u_star = self.u_star + self.noise * np.std(self.u_star) * np.random.randn(
            self.u_star.shape[0], self.u_star.shape[1]
        )
        self.v_star = self.v_star + self.noise * np.std(self.v_star) * np.random.randn(
            self.v_star.shape[0], self.v_star.shape[1]
        )

        N0 = Exact_u_ori.shape[0]
        N_b = Exact_u_ori.shape[1]

        idx_x = np.random.choice(x_ori.shape[0], N0, replace=False)
        self.t0_train = np.zeros(np.shape(x_ori))
        self.x0_train = x_ori[idx_x, :]
        self.u0_train = Exact_u_ori[idx_x, 0:1]
        self.v0_train = Exact_v_ori[idx_x, 0:1]

        idx_t = np.random.choice(t_ori.shape[0], N_b, replace=False)
        self.tb_train = np.concatenate([t_ori[idx_t, :], t_ori[idx_t, :]], axis=0)
        x_lb_train = np.ones(np.shape(t_ori)) * min(x_ori)
        x_ub_train = np.ones(np.shape(t_ori)) * (-min(x_ori))
        self.xb_train = np.concatenate([x_lb_train, x_ub_train], axis=0)

    def random_select(self):
        idx = np.random.choice(self.t_star.shape[0], self.N_train, replace=False)
        self.t_train = self.t_star[idx, :]
        self.x_train = self.x_star[idx, :]
        self.u_train = self.u_star[idx, :]
        self.v_train = self.v_star[idx, :]

        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(self.N_f_train)
        X_f_train = qmc.scale(sample, self.lb, self.ub)
        self.t_f_train = X_f_train[:, 0:1]
        self.x_f_train = X_f_train[:, 1:2]

    def save_dict(self):
        self.data_trans = {}
        # original data: t(201) x(256) u(256,201)
        self.data_trans["t_ori"] = self.t_ori
        self.data_trans["x_ori"] = self.x_ori
        self.data_trans["u_ori"] = self.u_ori
        self.data_trans["v_ori"] = self.v_ori
        self.data_trans["Exact_u_ori"] = self.Exact_u_ori
        self.data_trans["Exact_v_ori"] = self.Exact_v_ori
        self.data_trans["Exact_uv_ori"] = self.Exact_uv_ori

        # original(flatten) data: t/x/u(201x256) X_star(201x256,2)
        self.data_trans["t_star"] = self.t_star
        self.data_trans["x_star"] = self.x_star
        self.data_trans["u_star"] = self.u_star
        self.data_trans["v_star"] = self.v_star
        self.data_trans["X_star"] = self.X_star

        # train data:  N_train pts random choiced t/x/u(N_train)
        self.data_trans["t_train"] = self.t_train
        self.data_trans["x_train"] = self.x_train
        self.data_trans["u_train"] = self.u_train
        self.data_trans["v_train"] = self.v_train

        # t0 data: shuffled t0/x0/u0(256)
        self.data_trans["t0"] = self.t0_train
        self.data_trans["x0"] = self.x0_train
        self.data_trans["u0"] = self.u0_train
        self.data_trans["v0"] = self.v0_train

        # lb/ub data: shuffled tb/xb(201+201)
        self.data_trans["tb"] = self.tb_train
        self.data_trans["xb"] = self.xb_train

        # solver train data: N_f_train pts random generated t_f/x_f(N_f_train)
        self.data_trans["t_f_train"] = self.t_f_train
        self.data_trans["x_f_train"] = self.x_f_train

        # print(self.data_trans)
        scipy.io.savemat(self.save_path, self.data_trans)

    def __call__(self, file_path, save_path, keep, noise, N_train, N_f_train, lb, ub):
        # set config
        self.save_path = save_path
        self.keep = keep
        self.noise = noise
        self.N_train = N_train
        self.N_f_train = N_f_train
        self.lb = lb
        self.ub = ub

        # run
        data_ori = scipy.io.loadmat(file_path)
        self.trans_data(data_ori)
        self.random_select()
        self.save_dict()


if __name__ == "__main__":
    data = DataLoader()
    data(
        "../../Data/NLS.mat",
        "../dataset_gen/NLS.mat",
        1,
        0,
        10000,
        20000,
        lb = [0.0, -5.0],
        ub = [np.pi / 2, 5.0],
    )
