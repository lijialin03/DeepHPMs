import numpy as np
from scipy.interpolate import griddata
import sys

sys.path.append("../model/")
from net import Network
from deephpm import DeepHPM

sys.path.append("../scripts/")
from data_loader import DataLoader
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
        lr=0.001,
        n_train_idn=10000,
        n_train_pde=10000,
        n_train_sol=10000,
        mode=["train"],
    ):
        print("########## Starting", figname, "##########")
        # Load Data
        data = DataLoader()
        data(file_idn, file_sol)

        # Save Path
        net_path = "../saved_nets/" + figname + "_opt"

        # Training
        net_idn = Network(data.cfg.layers_idn)
        net_pde = Network(data.cfg.layers_pde)
        net_sol = Network(data.cfg.layers_sol)

        model = DeepHPM()
        model.init_idn(
            net_idn, data.t_train, data.x_train, data.u_train, data.lb_idn, data.ub_idn
        )
        model.init_pde(net_pde)
        model.init_sol(
            net_sol,
            data.tb_train,
            data.x0_train,
            data.u0_train,
            data.lb_sol,
            data.ub_sol,
            data.X_f_train,
        )

        # train idn and pde
        model.compile(lr=lr, max_grad=max_grad)
        if "train_gen_pde" in mode:
            model.train(n_train_idn, "idn")
            model.train(n_train_pde, "pde")
        elif "load_gen_pde" in mode:
            model.load(net_path, "idn")
            model.load(net_path, "pde")
        if "debug_gen_pde" in mode:
            # model.train(n_train_idn, "idn")
            # plot = Plotting(
            #     figname + "_debug_idn",
            #     data.lb_idn,
            #     data.ub_idn,
            #     data.X_idn_debug,
            #     (data.T_idn_d, data.X_idn_d),
            # )
            # u_pred_identifier, _ = model.predict(
            #     data.t_idn_debug, data.x_idn_debug, "idn"
            # )
            # plot.draw_n_save(data.Exact_idn_debug, u_pred_identifier)
            model.train(n_train_pde, "pde")
            plot = Plotting(
                figname + "_debug_pde",
                data.lb_idn,
                data.ub_idn,
                data.X_idn_debug,
                (data.T_idn_d, data.X_idn_d),
            )
            u_pred_pde, _ = model.predict(data.t_idn_debug, data.x_idn_debug, "pde")
            plot.draw_n_save(data.Exact_idn_debug, u_pred_pde)

            # model.save(net_path, "idn")
            model.save(net_path, "pde")
        # save nets
        if "save_gen_pde" in mode:
            model.save(net_path, "idn")
            model.save(net_path, "pde")
        # predict and print error
        u_pred_identifier, f_pred_identifier = model.predict(
            data.t_idn_star, data.x_idn_star, "idn"
        )
        error_u_identifier = np.linalg.norm(
            data.u_idn_star - u_pred_identifier, 2
        ) / np.linalg.norm(data.u_idn_star, 2)
        error_f_identifier = np.linalg.norm(f_pred_identifier)
        print("Error u idn-idn: %e" % (error_u_identifier))
        print("Error f pde-idn: %e" % (error_f_identifier))

        # train sol
        if "train_pinns" in mode:
            model.train(n_train_sol, "sol")
        elif "load_pinns" in mode:
            model.load(net_path, "sol")
        if "debug_pinns" in mode:
            model.train_debug(
                n_train_sol,
                data.t_sol_star,
                data.x_sol_star,
                data.lb_sol,
                data.ub_sol,
                data.X_sol_star,
                (data.T_sol, data.X_sol),
                "sol",
            )
        # save net_sol to resume training
        if "save_pinns" in mode:
            model.save(net_path, "sol")
        # predict and print error
        u_pred, _ = model.predict(data.t_sol_star, data.x_sol_star, "sol")
        error_u = np.linalg.norm(data.u_sol_star - u_pred, 2) / np.linalg.norm(
            data.u_sol_star, 2
        )
        print("Error u sol-sol: %e" % (error_u))

        # Plotting
        plot = Plotting(
            figname, data.lb_sol, data.ub_sol, data.X_sol_star, (data.T_sol, data.X_sol)
        )
        plot.draw_n_save(data.Exact_sol, u_pred)


if __name__ == "__main__":
    lr = 0.0001
    N_train = [1000, 1000, 10000]
    example = Example()
    # Burgers_Same
    example.run(
        "../../Data/burgers_sine.mat",
        "../../Data/burgers_sine.mat",
        "Burgers",
        2,
        lr,
        N_train[0],
        N_train[1],
        N_train[2],
        mode=[
            # "load_gen_pde",
            "train_gen_pde",
            # "save_gen_pde",
            "train_pinns",
            # "save_pinns",
        ],
    )
