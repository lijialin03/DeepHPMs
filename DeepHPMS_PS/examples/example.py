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
        net_path = "../saved_nets/" + figname

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
        if "train" in mode:
            model.train(n_train_idn, "idn")
            model.train(n_train_pde, "pde")
            # save nets
            model.save(net_path, "idn")
            model.save(net_path, "pde")
        elif "load" in mode:
            model.load(net_path, "idn")
            model.load(net_path, "pde")
        u_pred_identifier, f_pred_identifier = model.predict(
            data.t_idn_star, data.x_idn_star, "idn"
        )
        error_u_identifier = np.linalg.norm(
            data.u_idn_star - u_pred_identifier, 2
        ) / np.linalg.norm(data.u_idn_star, 2)
        print("Error u: %e" % (error_u_identifier))

        # for plotting
        u_pred_identifier_all, f_pred_identifier_all = model.predict(
            data.t_sol_star, data.x_sol_star, "idn"
        )

        # train sol
        if "train" in mode:
            model.train(n_train_sol, "sol")
            # save net_sol to resume training
            model.save(net_path, "sol")
        elif "resume" in mode:
            model.load(net_path, "sol")
        u_pred, f_pred = model.predict(data.t_sol_star, data.x_sol_star, "sol")
        error_u = np.linalg.norm(data.u_sol_star - u_pred, 2) / np.linalg.norm(
            data.u_sol_star, 2
        )
        print("Error u: %e" % (error_u))
        U_pred = griddata(
            data.X_sol_star, u_pred.flatten(), (data.T_sol, data.X_sol), method="cubic"
        )
        
        # Plotting
        plot = Plotting(figname, data.lb_sol, data.ub_sol)
        plot.draw_n_save(data.Exact_sol, u_pred_identifier_all, U_pred)

