import numpy as np
from scipy.interpolate import griddata
import sys

sys.path.append("../model/")
from net import Network
from deephpm_2vars import DeepHPM

sys.path.append("../scripts/")
from data_loader_2vars import DataLoader
from plotting import Plotting


class Example(object):
    def __init__(self) -> None:
        pass

    def run(
        self,
        file_idn,
        file_sol,
        figname="test",
        max_grad=2,
        num_var=1,
        lr=0.001,
        n_train_idn=10000,
        n_train_pde=10000,
        n_train_sol=10000,
        mode=["train"],
    ):
        print("########## Starting", figname, "##########")
        # Load Data
        data = DataLoader(num_var)
        data(file_idn, file_sol)

        # Save Path
        net_path = "../saved_nets/" + figname + "_opt"

        # Training
        net_idn_u = Network(data.cfg.layers_idn)
        net_pde_u = Network(data.cfg.layers_pde)
        net_sol_u = Network(data.cfg.layers_sol)
        net_idn = [net_idn_u]
        net_pde = [net_pde_u]
        net_sol = [net_sol_u]
        data_u_train_idn = [data.u_train]
        data_u_train_sol = [data.u0_train]

        if num_var == 2:
            net_idn_v = Network(data.cfg.layers_idn)
            net_pde_v = Network(data.cfg.layers_pde)
            net_sol_v = Network(data.cfg.layers_sol)
            net_idn.append(net_idn_v)
            net_pde.append(net_pde_v)
            net_sol.append(net_sol_v)
            data_u_train_idn.append(data.v_train)
            data_u_train_sol.append(data.v0_train)

        model = DeepHPM()
        model.init_idn(
            net_idn,
            data.t_train,
            data.x_train,
            data_u_train_idn,
            data.lb_idn,
            data.ub_idn,
        )
        model.init_pde(net_pde)
        model.init_sol(
            net_sol,
            data.tb_train,
            data.x0_train,
            data_u_train_sol,
            data.lb_sol,
            data.ub_sol,
            data.X_f_train,
        )

        # train idn and pde
        model.compile(lr=lr, max_grad=max_grad)
        if "load_gen_pde" in mode:
            model.load(net_path, "idn")
            model.load(net_path, "pde")
        if "train_gen_pde" in mode:
            model.train(n_train_idn, "idn")
            model.train(n_train_pde, "pde")
        if "save_gen_pde" in mode:
            model.save(net_path, "idn")
            model.save(net_path, "pde")
        # predict and print error
        pred_identifier = model.predict(data.t_idn_star, data.x_idn_star, "idn")
        error_u_identifier = np.linalg.norm(
            data.u_idn_star - pred_identifier[0], 2
        ) / np.linalg.norm(data.u_idn_star, 2)
        error_f_identifier = np.linalg.norm(pred_identifier[1]) / np.sqrt(
            len(pred_identifier[1])
        )
        print("Error u idn-idn: %e" % (error_u_identifier))
        print("Error f pde-idn: %e" % (error_f_identifier))
        if num_var == 2:
            error_v_identifier = np.linalg.norm(
                data.v_idn_star - pred_identifier[2], 2
            ) / np.linalg.norm(data.v_idn_star, 2)
            error_g_identifier = np.linalg.norm(pred_identifier[3]) / np.sqrt(
                len(pred_identifier[3])
            )
            print("Error v idn-idn: %e" % (error_v_identifier))
            print("Error g pde-idn: %e" % (error_g_identifier))

        # train sol
        if "load_pinns" in mode:
            model.load(net_path, "sol")
        if "train_pinns" in mode:
            model.train(n_train_sol, "sol")
        if "save_pinns" in mode:
            model.save(net_path, "sol")
        # predict and print error
        pred_sol = model.predict(data.t_sol_star, data.x_sol_star, "sol")
        error_u = np.linalg.norm(data.u_sol_star - pred_sol[0], 2) / np.linalg.norm(
            data.u_sol_star, 2
        )
        print("Error u sol-sol: %e" % (error_u))
        if num_var == 2:
            error_v = np.linalg.norm(data.v_sol_star - pred_sol[2], 2) / np.linalg.norm(
                data.v_sol_star, 2
            )
            print("Error v sol-sol: %e" % (error_v))

        # Plotting
        plot = Plotting(
            figname, data.lb_sol, data.ub_sol, data.X_sol_star, (data.T_sol, data.X_sol)
        )
        if num_var == 1:
            plot.draw_n_save(data.Exact_u_sol, pred_sol[0])
            plot.draw_t_2d(data.Exact_u_sol, pred_sol[0])
        elif num_var == 2:
            UV_pred = np.sqrt(pred_sol[0] ** 2 + pred_sol[2] ** 2)
            plot.draw_n_save(data.Exact_uv_sol, UV_pred)
            plot.draw_t_2d(data.Exact_uv_sol, UV_pred)


if __name__ == "__main__":
    lr = 0.00001
    N_train = [100000, 300000, 10000]
    example = Example()
    # Schrodinger
    example.run(
        "../../Data/NLS.mat",
        "../../Data/NLS.mat",
        "NLS",
        2,
        2,
        lr,
        N_train[0],
        N_train[1],
        N_train[2],
        mode=[
            "load_gen_pde",
            "train_gen_pde",
            "save_gen_pde",
            # "train_pinns",
            # "save_pinns",
        ],
    )

    # Burgers_Same
    # example.run(
    #     "../../Data/burgers_sine.mat",
    #     "../../Data/burgers_sine.mat",
    #     "Burgers",
    #     2,
    #     1,
    #     lr,
    #     N_train[0],
    #     N_train[1],
    #     N_train[2],
    #     mode=[
    #         # "load_gen_pde",
    #         "train_gen_pde",
    #         # "save_gen_pde",
    #         "train_pinns",
    #         # "save_pinns",
    #     ],
    # )
