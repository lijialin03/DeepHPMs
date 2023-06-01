import numpy as np
import scipy.io
from scipy.stats import qmc


class DataLoader(object):
    def __init__(self) -> None:
        np.random.seed(1)
        self.data_trans = {}

    def trans_data(self,data):
        """Load and Prepare data"""
        # Load data
        t_data = data["t_star"]
        self.X_data = data["X_star"]
        U_data = data["U_star"]
        self.w_data = data["w_star"]

        t_star = np.tile(t_data.T, (2310, 1))
        x_star = np.tile(self.X_data[:, 0:1], (1, 151))
        y_star = np.tile(self.X_data[:, 1:2], (1, 151))
        u_star = U_data[:, 0, :]
        v_star = U_data[:, 1, :]
        w_star = self.w_data

        t_star = np.reshape(t_star, (-1, 1))
        x_star = np.reshape(x_star, (-1, 1))
        y_star = np.reshape(y_star, (-1, 1))
        u_star = np.reshape(u_star, (-1, 1))
        v_star = np.reshape(v_star, (-1, 1))
        w_star = np.reshape(w_star, (-1, 1))

        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.u_star = u_star
        self.v_star = v_star
        self.w_star = w_star

        # Prepare data
        # For identification
        idx = np.random.choice(t_star.shape[0], self.N_train, replace=True)
        self.t_train = t_star[idx, :]
        self.x_train = x_star[idx, :]
        self.y_train = y_star[idx, :]
        self.u_train = u_star[idx, :]
        self.v_train = v_star[idx, :]
        self.w_train = w_star[idx, :]

        # For solution
        t_b0 = t_star[t_star == t_star.min()][:, None]
        x_b0 = x_star[t_star == t_star.min()][:, None]
        y_b0 = y_star[t_star == t_star.min()][:, None]
        w_b0 = w_star[t_star == t_star.min()][:, None]

        t_b1 = t_star[x_star == x_star.min()][:, None]
        x_b1 = x_star[x_star == x_star.min()][:, None]
        y_b1 = y_star[x_star == x_star.min()][:, None]
        w_b1 = w_star[x_star == x_star.min()][:, None]

        t_b2 = t_star[x_star == x_star.max()][:, None]
        x_b2 = x_star[x_star == x_star.max()][:, None]
        y_b2 = y_star[x_star == x_star.max()][:, None]
        w_b2 = w_star[x_star == x_star.max()][:, None]

        t_b3 = t_star[y_star == y_star.min()][:, None]
        x_b3 = x_star[y_star == y_star.min()][:, None]
        y_b3 = y_star[y_star == y_star.min()][:, None]
        w_b3 = w_star[y_star == y_star.min()][:, None]

        t_b4 = t_star[y_star == y_star.max()][:, None]
        x_b4 = x_star[y_star == y_star.max()][:, None]
        y_b4 = y_star[y_star == y_star.max()][:, None]
        w_b4 = w_star[y_star == y_star.max()][:, None]

        self.t_b_train = np.concatenate((t_b0, t_b1, t_b2, t_b3, t_b4))
        self.x_b_train = np.concatenate((x_b0, x_b1, x_b2, x_b3, x_b4))
        self.y_b_train = np.concatenate((y_b0, y_b1, y_b2, y_b3, y_b4))
        self.w_b_train = np.concatenate((w_b0, w_b1, w_b2, w_b3, w_b4))

        idx = np.random.choice(t_star.shape[0], self.N_train, replace=True)
        self.t_f_train = t_star[idx, :]
        self.x_f_train = x_star[idx, :]
        self.y_f_train = y_star[idx, :]
        self.u_f_train = u_star[idx, :]
        self.v_f_train = v_star[idx, :]

    def save_dict(self):
        self.data_trans = {}
        # # original data: t(201) x(256) u(256,201)
        # self.data_trans["t_ori"] = self.t_ori
        # self.data_trans["x_ori"] = self.x_ori
        # self.data_trans["u_ori"] = self.u_ori
        # self.data_trans["Exact_ori"] = self.Exact_ori

        # original(flatten) data: t/x/u(201x256) X_star(201x256,2)
        self.data_trans["t_star"] = self.t_star
        self.data_trans["x_star"] = self.x_star
        self.data_trans["y_star"] = self.y_star
        self.data_trans["u_star"] = self.u_star
        self.data_trans["v_star"] = self.v_star
        self.data_trans["w_star"] = self.w_star
        self.data_trans["X_star"] = self.X_data
        self.data_trans["w_data"] = self.w_data

        # train data:  N_train pts random choiced t/x/u(N_train)
        self.data_trans["t_train"] = self.t_train
        self.data_trans["x_train"] = self.x_train
        self.data_trans["y_train"] = self.y_train
        self.data_trans["u_train"] = self.u_train
        self.data_trans["v_train"] = self.v_train
        self.data_trans["w_train"] = self.w_train

        # lb/ub data: shuffled tb/xb(201+201)
        self.data_trans["tb"] = self.t_b_train
        self.data_trans["xb"] = self.x_b_train
        self.data_trans["yb"] = self.y_b_train
        self.data_trans["wb"] = self.w_b_train

        # solver train data: N_f_train pts random generated t_f/x_f(N_f_train)
        self.data_trans["t_f_train"] = self.t_f_train
        self.data_trans["x_f_train"] = self.x_f_train
        self.data_trans["y_f_train"] = self.y_f_train
        self.data_trans["u_f_train"] = self.u_f_train
        self.data_trans["v_f_train"] = self.v_f_train

        # print(self.data_trans)
        scipy.io.savemat(self.save_path, self.data_trans)

    def __call__(self, file_path, save_path, N_train):
        # set config
        self.save_path = save_path
        self.N_train = N_train

        # run
        data_ori = scipy.io.loadmat(file_path)
        self.trans_data(data_ori)
        self.save_dict()


if __name__ == "__main__":
    data = DataLoader()
    data(
        "../../Data/cylinder.mat",
        "../dataset_gen/cylinder.mat",
        50000,
    )
