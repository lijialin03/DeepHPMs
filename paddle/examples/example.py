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
        lr=[0.001, 0.001, 0.001],
        n_train_idn=10000,
        n_train_pde=10000,
        n_train_sol=10000,
        mode=["train_gen_pde"],
        opt="adam",
    ):
        print("########## Starting", figname, "##########")
        # Load Data
        data = DataLoader()
        data(file_idn, file_sol)
        data.cfg.layers_pde = [max_grad + 1, 100, 100, 1]
        # data.cfg.layers_sol = [2, 100, 100, 100, 1]

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
            None,  # net_sol,
            data.tb_train,
            data.x0_train,
            data.u0_train,
            data.lb_sol,
            data.ub_sol,
            data.X_f_train,
            # data.t_suv,
            # data.x_suv,
            # data.u_suv,
        )

        # train idn and pde
        model.compile(optimizer=opt, lr=lr, max_grad=max_grad)
        if "load_gen_pde" in mode:
            model.load(net_path, "idn")
            if max_grad >= 3:
                paddle.fluid.core.set_prim_eager_enabled(True)
            model.load(net_path, "pde")
        if "train_gen_pde" in mode:
            model.train(n_train_idn, "idn")
            if max_grad >= 3:
                paddle.fluid.core.set_prim_eager_enabled(True)
            model.train(n_train_pde, "pde")
        if "debug_gen_pde" in mode:
            plot = Plotting(
                figname + "_debug_pde",
                data.lb_idn,
                data.ub_idn,
                data.X_idn_debug,
                (data.T_idn_d, data.X_idn_d),
            )
            u_pred, _ = model.predict(data.t_idn_debug, data.x_idn_debug, "pde")
            error_u_idn = np.linalg.norm(data.u_idn_debug - u_pred, 2) / np.linalg.norm(
                data.u_idn_debug, 2
            )
            print("Error u idn-idn_all: %e" % (error_u_idn))
            plot.draw_n_save(data.Exact_idn_debug, u_pred)
            plot.draw_t_2d(data.Exact_idn_debug, u_pred)
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
        error_f_identifier = np.linalg.norm(f_pred_identifier, 2) / np.sqrt(
            len(f_pred_identifier)
        )
        print("Error u idn-idn: %e" % (error_u_identifier))
        print("Error f pde-idn: %e" % (error_f_identifier))

        # train sol
        if "load_pinns" in mode:
            model.load(net_path, "sol")
        if "train_pinns" in mode:
            model.train(n_train_sol, "sol")
        # save net_sol to resume training
        if "save_pinns" in mode:
            model.save(net_path, "sol")
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
        # predict and print error
        u_pred, f_pred = model.predict(data.t_sol_star, data.x_sol_star, "sol")
        error_u = np.linalg.norm(data.u_sol_star - u_pred, 2) / np.linalg.norm(
            data.u_sol_star, 2
        )
        error_f = np.linalg.norm(f_pred, 2) / np.sqrt(len(f_pred))
        print("Error u sol-sol: %e" % (error_u))
        print("Error f sol-sol: %e" % (error_f))
        u_pred_idn, _ = model.predict(data.t_idn_star, data.x_idn_star, "sol")
        error_u_idn = np.linalg.norm(data.u_idn_star - u_pred_idn, 2) / np.linalg.norm(
            data.u_idn_star, 2
        )
        print("Error u sol-idn: %e" % (error_u_idn))

        # Plotting
        plot = Plotting(
            figname, data.lb_sol, data.ub_sol, data.X_sol_star, (data.T_sol, data.X_sol)
        )
        plot.draw_n_save(data.Exact_sol, u_pred)
        plot.draw_t_2d(data.Exact_sol, u_pred)


if __name__ == "__main__":
    dataset = [
        "../../Data/burgers.mat",
        "../../Data/burgers_sine.mat",
        "../../Data/KdV_sine.mat",
        "../../Data/KdV_cos.mat",
        "../../Data/KS.mat",
        "../../Data/KS_chaotic.mat",
        "../../Data/NLS.mat",
        "../../Data/cylinder.mat",
    ]
    max_grad_dict = {"burgers": 2, "kdv": 3, "ks": 4}
    lr = [1e-3, 1e-3, 1e-3]
    N_train = [50000, 50000, 50000]
    example = Example()
    # test
    # Burgers_Same
    import paddle

    paddle.device.set_device("gpu:2")
    paddle.disable_static()
    # paddle.fluid.core.set_prim_eager_enabled(True)
    # example.run(
    #     dataset[2],
    #     dataset[3],
    #     # "test",
    #     # "Burgers",
    #     # "lbfgs",
    #     # "Burgers_Different",
    #     # "Burgers_Different_Swap",
    #     # "KdV",
    #     "KdV_Different",
    #     max_grad_dict["kdv"],
    #     lr,
    #     N_train[0],
    #     N_train[1],
    #     N_train[2],
    #     mode=[
    #         # "load_gen_pde",
    #         "train_gen_pde",
    #         "debug_gen_pde",
    #         "save_gen_pde",
    #         # "load_pinns",
    #         "train_pinns",
    #         "save_pinns",
    #     ],
    #     opt="adam",
    # )

    # lr = [1, 1, 1]
    # N_train = [50000, 50000, 50000]
    # example.run(
    #     dataset[2],
    #     dataset[3],
    #     # "KdV_lbfgs",
    #     "KdV_Different_lbfgs",
    #     max_grad_dict["kdv"],
    #     lr,
    #     N_train[0],
    #     N_train[1],
    #     N_train[2],
    #     mode=[
    #         # "load_gen_pde",
    #         "train_gen_pde",
    #         "debug_gen_pde",
    #         "save_gen_pde",
    #         # "load_pinns",
    #         "train_pinns",
    #         "save_pinns",
    #     ],
    #     opt="lbfgs",
    # )

    paddle.fluid.core.set_prim_eager_enabled(True)
    lr = [1e-3, 1e-3, 1e-3]
    N_train = [50000, 50000, 50000]
    example.run(
        dataset[5],
        dataset[5],
        # "KdV",
        # "KdV_Different",
        # "KS",
        # "KS_lbfgs",
        # "KS_nasty",
        "KS_nasty_lbfgs",
        max_grad_dict["ks"],
        lr,
        N_train[0],
        N_train[1],
        N_train[2],
        mode=[
            "load_gen_pde",
            # "train_gen_pde",
            # "debug_gen_pde",
            # "save_gen_pde",
            # "load_pinns",
            "train_pinns",
            "save_pinns",
        ],
        opt="lbfgs",
    )
