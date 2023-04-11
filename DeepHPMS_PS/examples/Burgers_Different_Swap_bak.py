import numpy as np
from scipy.interpolate import griddata
import sys

sys.path.append("../model/")
from net import Network
from deephpm import DeepHPM

sys.path.append("../scripts/")
from data_loader import DataLoader
from plotting import Plotting


# Load Data
file_idn = "../../Data/burgers.mat"
file_sol = "../../Data/burgers_sine.mat"
data = DataLoader()
data(file_idn, file_sol)

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
model.compile(lr=0.001, max_grad=2)
model.train(100, "idn")
model.train(100, "pde")
u_pred_identifier, f_pred_identifier = model.predict(
    data.t_idn_star, data.x_idn_star, "idn"
)
error_u_identifier = np.linalg.norm(
    data.u_idn_star - u_pred_identifier, 2
) / np.linalg.norm(data.u_idn_star, 2)
print("Error u: %e" % (error_u_identifier))

# train sol
model.train(100, "sol")
u_pred, f_pred = model.predict(data.t_sol_star, data.x_sol_star, "sol")
error_u = np.linalg.norm(data.u_sol_star - u_pred, 2) / np.linalg.norm(
    data.u_sol_star, 2
)
print("Error u: %e" % (error_u))
U_pred = griddata(
    data.X_sol_star, u_pred.flatten(), (data.T_sol, data.X_sol), method="cubic"
)

# Plotting
plot = Plotting("Burgers_fail", data.lb_sol, data.ub_sol)
plot.draw_n_save(data.Exact_sol, U_pred)
