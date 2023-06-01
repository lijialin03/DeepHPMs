import ppsci
from ppsci.utils import logger
from ppsci.autodiff import jacobian, hessian
from ppsci.loss import MSELoss
from ppsci.metric import L2Rel
import paddle

import numpy as np
import sys

sys.path.append("../../")
from paddle_science.model.deephpm import DeepHPMs
from paddle_science.model.loss import LossPde, LossBoundary
from paddle_science.model.matric import MatricPde, MatricSol
from paddle_science.scripts.data_loader import DataLoader

from paddlepaddle.scripts.plotting import Plotting


FLAGS_paddle_num_threads = 1


def test(model, output_key, plot):
    data = DataLoader()
    data(data_t1, data_t2)
    mode = output_key
    if mode in ["u_idn", "f_pde"]:
        t = paddle.to_tensor(data.t_idn_star, dtype="float32")
        x = paddle.to_tensor(data.x_idn_star, dtype="float32")
    elif mode == "u_sol":
        t = paddle.to_tensor(data.t_sol_star, dtype="float32")
        x = paddle.to_tensor(data.x_sol_star, dtype="float32")

    t.stop_gradient = False
    x.stop_gradient = False
    input_test = {"t": t, "x": x}
    pred = model(input_test)[output_key].numpy()

    if mode == "u_idn":
        error_u = np.linalg.norm(data.u_idn_star - pred, 2) / np.linalg.norm(
            data.u_idn_star, 2
        )
        print("Error u idn: %e" % (error_u))
    elif mode == "f_pde":
        label = jacobian(model_idn(input_test)["u_idn"], t).numpy()
        error_f = np.linalg.norm(label - pred, 2) / np.sqrt(len(pred))
        print("Error f pde: %e" % (error_f))
    elif mode == "u_sol":
        error_u = np.linalg.norm(data.u_sol_star - pred, 2) / np.linalg.norm(
            data.u_sol_star, 2
        )
        print("Error u sol: %e" % (error_u))

    if plot:
        t = paddle.to_tensor(data.t_sol_star, dtype="float32")
        x = paddle.to_tensor(data.x_sol_star, dtype="float32")

        t.stop_gradient = False
        x.stop_gradient = False
        input_test = {"t": t, "x": x}
        pred = model(input_test)[output_key].numpy()

        plot = Plotting(
            "Burgers_" + output_key,
            data.lb_sol,
            data.ub_sol,
            data.X_sol_star,
            (data.T_sol, data.X_sol),
        )
        plot.draw_n_save(data.Exact_sol, pred)
        plot.draw_t_2d(data.Exact_sol, pred)


if __name__ == "__main__":
    # set example params
    output_dir = "../outs/"
    data_t1 = "../dataset_gen/burgers_sine.mat"
    data_t2 = "../dataset_gen/burgers_sine.mat"

    t_lb = paddle.to_tensor([0.0], dtype="float32")
    t_ub = paddle.to_tensor([10.0], dtype="float32")
    x_lb = paddle.to_tensor([-8.0], dtype="float32")
    x_ub = paddle.to_tensor([8.0], dtype="float32")

    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    # set training hyper-parameters
    epochs = 3
    iters_per_epoch = 1
    lr = 1e-3

    # init models
    model_idn = ppsci.arch.MLP(("t", "x"), ("u_idn",), 4, 50, "sin")
    model_pde = ppsci.arch.MLP(("u_x", "du_x", "du_xx"), ("f_pde",), 2, 100, "sin", 3)
    model_sol = ppsci.arch.MLP(("t", "x"), ("u_sol",), 4, 50, "sin")

    # transform
    def transform_u(input):
        # input-keys from dataset-keys
        t, x = input["t"], input["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_f(input):
        # input-keys from dataset-keys
        in_idn = {"t": input["t"], "x": input["x"]}
        # print(float(input["t"][0]))
        x = input["x"]
        u = model_idn(in_idn)["u_idn"]
        du_x = jacobian(u, x)
        du_xx = hessian(u, x)
        input_trans = {"u_x": u, "du_x": du_x, "du_xx": du_xx}
        return input_trans
    
    def transform_f_sol(input):
        # input-keys from dataset-keys
        in_sol = {"t": input["t"], "x": input["x"]}
        x = input["x"]
        u = model_sol(in_sol)["u_sol"]
        du_x = jacobian(u, x)
        du_xx = hessian(u, x)
        input_trans = {"u_x": u, "du_x": du_x, "du_xx": du_xx}
        return input_trans

    model_idn.register_input_transform(transform=transform_u)
    model_pde.register_input_transform(transform=transform_f)
    model_sol.register_input_transform(transform=transform_u)
    model_list = ppsci.arch.ModelList((model_idn, model_pde, model_sol))

    # set optimizer
    # lr_scheduler = ppsci.optimizer.lr_scheduler.Constant(lr)
    opt_idn = ppsci.optimizer.Adam(lr)([model_idn])
    opt_pde = ppsci.optimizer.Adam(lr)([model_pde])
    opt_sol = ppsci.optimizer.Adam(lr)([model_sol])

    # init DeepHPMs
    deephpms = DeepHPMs(model_list=model_list, output_dir=output_dir)

    # run idn
    print("———————————————————— run idn ————————————————————")
    # deephpms.checkpoint_path = "../outs/checkpoints/test"
    # deephpms.checkpoint_path = "../outs/checkpoints/epoch_50000"
    deephpms.checkpoint_path = None
    cfg_idn = deephpms.gen_cfg(
        data_t1,
        ("t", "x"),
        ("u_idn",),
        {"t": "t_train", "x": "x_train", "u_idn": "u_train"},
    )
    out_expr_idn = {key: (lambda out, k=key: out[k]) for key in ("u_idn",)}

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        cfg_idn, MSELoss("sum"), out_expr_idn, name="u_mse_sup"
    )
    constraint = {sup_constraint.name: sup_constraint}

    sup_validator = ppsci.validate.SupervisedValidator(
        cfg_idn, MSELoss("sum"), out_expr_idn, {"l2": L2Rel()}, name="u_L2_sup"
    )
    validator = {sup_validator.name: sup_validator}

    deephpms.set_proc_params(opt_idn, epochs, iters_per_epoch, constraint, validator)
    solver = deephpms.gen_solver()
    deephpms.run(solver)

    test(model_idn, "u_idn", True)

    # # run pde
    # print("———————————————————— run pde ————————————————————")
    # # deephpms.checkpoint_path = None
    # cfg_pde = deephpms.gen_cfg(
    #     data_t1, ("t", "x"), alias_dict={"t": "t_train", "x": "x_train"}
    # )
    # out_expr_pde = {
    #     "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
    #     "f_pde": lambda out: out["f_pde"],
    # }

    # sup_constraint = ppsci.constraint.SupervisedConstraint(
    #     cfg_pde, LossPde("sum"), out_expr_pde, name="f_mse_sup"
    # )
    # constraint = {sup_constraint.name: sup_constraint}

    # sup_validator = ppsci.validate.SupervisedValidator(
    #     cfg_pde, LossPde("sum"), out_expr_pde, {"l2": MatricPde()}, name="f_L2_sup"
    # )
    # validator = {sup_validator.name: sup_validator}

    # deephpms.set_proc_params(opt_pde, epochs, iters_per_epoch, constraint, validator)
    # solver.constraint=constraint
    # solver.optimizer=opt_pde
    # solver.validator=validator
    # # solver = deephpms.gen_solver()
    # deephpms.run(solver)

    # test(model_pde, "f_pde", True)

    # # run sol
    # print("———————————————————— run sol ————————————————————")
    # model_pde.register_input_transform(transform=transform_f_sol)
    # # deephpms.checkpoint_path = None
    # cfg_sol_f = deephpms.gen_cfg(
    #     data_t2, ("t", "x"), alias_dict={"t": "t_f_train", "x": "x_f_train"}
    # )
    # cfg_sol_u0 = deephpms.gen_cfg(
    #     data_t2, ("t", "x"), ("u_sol",), {"t": "t0", "x": "x0", "u_sol": "u0"}
    # )
    # cfg_sol_ub = deephpms.gen_cfg(
    #     data_t2, ("t", "x"), alias_dict={"t": "tb", "x": "xb"}
    # )

    # out_expr_sol_f = {
    #     "f_pde": lambda out: out["f_pde"],
    #     "du_t": lambda out: jacobian(out["u_sol"], out["t"]),
    # }
    # out_expr_sol_u0 = {"u_sol": lambda out: out["u_sol"]}
    # out_expr_sol_ub = {
    #     "ub_sol": lambda out: out["u_sol"],
    #     "du_xb_sol": lambda out: jacobian(out["u_sol"], out["x"]),
    # }

    # sup_constraint_f = ppsci.constraint.SupervisedConstraint(
    #     cfg_sol_f, LossPde("sum"), out_expr_sol_f, name="f_mse_sup"
    # )
    # sup_constraint_u0 = ppsci.constraint.SupervisedConstraint(
    #     cfg_sol_u0, MSELoss("sum"), out_expr_sol_u0, name="u0_mse_sup"
    # )
    # sup_constraint_ub = ppsci.constraint.SupervisedConstraint(
    #     cfg_sol_ub, LossBoundary("sum"), out_expr_sol_ub, name="ub_mse_sup"
    # )

    # constraint = {
    #     sup_constraint_f.name: sup_constraint_f,
    #     sup_constraint_u0.name: sup_constraint_u0,
    #     sup_constraint_ub.name: sup_constraint_ub,
    # }

    # cfg_sol_valid = deephpms.gen_cfg(
    #     data_t2,
    #     ("t", "x"),
    #     ("u_sol",),
    #     {"t": "t_star", "x": "x_star", "u_sol": "u_star"},
    # )
    # sup_validator = ppsci.validate.SupervisedValidator(
    #     cfg_sol_valid,
    #     MSELoss("sum"),
    #     out_expr_sol_u0,
    #     {"l2": L2Rel()},
    #     name="u_L2_sup",
    # )
    # validator = {
    #     sup_validator.name: sup_validator,
    # }

    # deephpms.set_proc_params(opt_sol, epochs, iters_per_epoch, constraint, validator)
    # solver.constraint=constraint
    # solver.optimizer=opt_sol
    # solver.validator=validator
    # # solver = deephpms.gen_solver()
    # deephpms.run(solver)

    test(model_sol, "u_sol", True)
