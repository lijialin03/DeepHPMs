# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.loss import MSELoss
from ppsci.metric import L2Rel
from ppsci.utils import logger

import sys

sys.path.append("../../")
from paddle_science.scripts.data_loader_NLS import DataLoader
from paddlepaddle.scripts.plotting import Plotting


def test(model, output_key, plot, plot_uv=False):
    data = DataLoader()
    data(DATASET_PATH, DATASET_PATH_SOL)
    mode = output_key
    if mode in ["u_idn", "v_idn", "f_pde", "g_pde"]:
        t = paddle.to_tensor(data.t_idn_star, dtype="float32")
        x = paddle.to_tensor(data.x_idn_star, dtype="float32")
    elif mode in ["u_sol", "v_sol"]:
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
    elif mode == "v_idn":
        error_v = np.linalg.norm(data.v_idn_star - pred, 2) / np.linalg.norm(
            data.v_idn_star, 2
        )
        print("Error v idn: %e" % (error_v))
    elif mode == "f_pde":
        label = jacobian(model_idn_u(input_test)["u_idn"], t).numpy()
        error_f = np.linalg.norm(label - pred, 2) / np.sqrt(len(pred))
        print("Error f pde: %e" % (error_f))
    elif mode == "g_pde":
        label = jacobian(model_idn_v(input_test)["v_idn"], t).numpy()
        error_g = np.linalg.norm(label - pred, 2) / np.sqrt(len(pred))
        print("Error g pde: %e" % (error_g))
    elif mode == "u_sol":
        error_u = np.linalg.norm(data.u_sol_star - pred, 2) / np.linalg.norm(
            data.u_sol_star, 2
        )
        print("Error u sol: %e" % (error_u))
    elif mode == "v_sol":
        error_v = np.linalg.norm(data.v_sol_star - pred, 2) / np.linalg.norm(
            data.v_sol_star, 2
        )
        print("Error v sol: %e" % (error_v))

    if plot:
        t = paddle.to_tensor(data.t_sol_star, dtype="float32")
        x = paddle.to_tensor(data.x_sol_star, dtype="float32")

        t.stop_gradient = False
        x.stop_gradient = False
        input_test = {"t": t, "x": x}
        pred = model(input_test)[output_key].numpy()

        plot = Plotting(
            "nls_adam_" + output_key,
            [0.0, -5.0],
            [np.pi / 2, 5.0],
            data.X_sol_star,
            (data.T_sol, data.X_sol),
        )
        plot.draw_n_save(data.Exact_u_sol, pred)
        plot.draw_t_2d(data.Exact_u_sol, pred)

    if plot_uv:
        plot.figname = "nls_adam_uv"
        pred_u = model_sol_u(input_test)["u_sol"].numpy()
        pred_v = model_sol_v(input_test)["v_sol"].numpy()
        UV_pred = np.sqrt(pred_u ** 2 + pred_v ** 2)
        plot.draw_n_save(data.Exact_uv_sol, UV_pred)
        plot.draw_t_2d(data.Exact_uv_sol, UV_pred)


class LossPde(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = super().forward({"f_pde": output_dict["f_pde"]},  {"f_pde": output_dict["du_t"]}, weight_dict)
        losses += super().forward({"g_pde": output_dict["g_pde"]},  {"g_pde": output_dict["dv_t"]}, weight_dict)
        return losses


class LossBoundary(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def forward(self, output_dict, label_dict, weight_dict=None):
        u_b, v_b = output_dict["ub_sol"], output_dict["vb_sol"]
        u_lb, u_ub = paddle.split(u_b, 2, axis=0)
        v_lb, v_ub = paddle.split(v_b, 2, axis=0)
        x_b = output_dict["x_b"]

        du_x = jacobian(u_b, x_b)
        du_x_lb, du_x_ub = paddle.split(du_x, 2, axis=0)

        dv_x = jacobian(v_b, x_b)
        dv_x_lb, dv_x_ub = paddle.split(dv_x, 2, axis=0)

        losses = super().forward({"ub": u_lb}, {"ub": u_ub}, weight_dict)
        losses += super().forward({"vb": u_lb}, {"vb": u_ub}, weight_dict)
        losses += super().forward({"du_x_b": du_x_lb}, {"du_x_b": du_x_ub}, weight_dict)
        losses += super().forward({"dv_x_b": du_x_lb}, {"dv_x_b": du_x_ub}, weight_dict)
        return losses


class MatricPde(L2Rel):
    def __init__(self):
        super().__init__()

    def forward(self, output_dict, label_dict):
        out_dict = {"pred_f": output_dict["f_pde"], "pred_g": output_dict["g_pde"]}
        label_dict = {"pred_f": output_dict["du_t"], "pred_g": output_dict["dv_t"]}
        metric_dict = super().forward(out_dict, label_dict)
        return metric_dict

if __name__ == "__main__":
    ppsci.utils.misc.set_random_seed(42)
    EPOCHS = 50000  # 1 for LBFGS
    MAX_ITER = 50000    # for LBFGS
    ITERS_PER_EPOCH = 1
    LEARNING_RATE = 1e-4
    DATASET_PATH = "../dataset_gen/NLS.mat"
    DATASET_PATH_SOL = "../dataset_gen/NLS.mat"
    OUTPUT_DIR = "../outs_nls_adam/"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize boundaries
    t_lb = paddle.to_tensor([0.0], dtype="float32")
    t_ub = paddle.to_tensor([np.pi / 2.0], dtype="float32")
    x_lb = paddle.to_tensor([-5.0], dtype="float32")
    x_ub = paddle.to_tensor([5.0], dtype="float32")

    # initialize models
    model_idn_u = ppsci.arch.MLP(("t", "x"), ("u_idn",), 4, 50, "sin")
    model_idn_v = ppsci.arch.MLP(("t", "x"), ("v_idn",), 4, 50, "sin")
    model_pde_f = ppsci.arch.MLP(("u_x", "v_x", "du_x", "dv_x", "du_xx", "dv_xx"), ("f_pde",), 2, 100, "sin", 6)
    model_pde_g = ppsci.arch.MLP(("u_x", "v_x", "du_x", "dv_x", "du_xx", "dv_xx"), ("g_pde",), 2, 100, "sin", 6)
    model_sol_u = ppsci.arch.MLP(("t", "x"), ("u_sol",), 4, 50, "sin")
    model_sol_v = ppsci.arch.MLP(("t", "x"), ("v_sol",), 4, 50, "sin")
    
    # initialize transform
    def transform_uv(input):
        t, x = input["t"], input["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_fg(input, model_trans_u, model_trans_v, out_key_u, out_key_v):
        in_idn = {"t": input["t"], "x": input["x"]}
        x = input["x"]
        u = model_trans_u(in_idn)[out_key_u]
        v = model_trans_v(in_idn)[out_key_v]

        du_x = jacobian(u, x)
        du_xx = hessian(u, x)

        dv_x = jacobian(v, x)
        dv_xx = hessian(v, x)

        input_trans = {"u_x": u, "v_x": v, "du_x": du_x, "dv_x": dv_x, "du_xx": du_xx, "dv_xx": dv_xx}
        return input_trans

    def transform_fg_idn(input):
        return transform_fg(input, model_idn_u, model_idn_v, "u_idn", "v_idn")

    def transform_fg_sol(input):
        return transform_fg(input, model_sol_u, model_sol_v, "u_sol", "v_sol")

    # register transform
    model_idn_u.register_input_transform(transform=transform_uv)
    model_idn_v.register_input_transform(transform=transform_uv)
    model_pde_f.register_input_transform(transform=transform_fg_idn)
    model_pde_g.register_input_transform(transform=transform_fg_idn)
    model_sol_u.register_input_transform(transform=transform_uv)
    model_sol_v.register_input_transform(transform=transform_uv)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn_u, model_idn_v, model_pde_f, model_pde_g, model_sol_u, model_sol_v))

    # initialize optimizer
    # # Adam
    optimizer_idn = ppsci.optimizer.Adam(LEARNING_RATE)([model_idn_u, model_idn_v])
    optimizer_pde = ppsci.optimizer.Adam(LEARNING_RATE)([model_pde_f, model_pde_g])
    optimizer_sol = ppsci.optimizer.Adam(LEARNING_RATE)([model_sol_u, model_sol_v])

    # LBFGS
    # optimizer_idn = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)([model_idn_u, model_idn_v])
    # optimizer_pde = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)([model_pde_f, model_pde_g])
    # optimizer_sol = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)([model_sol_u, model_sol_v])

    # model 1: identification net
    # maunally build constraint(s)
    train_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn","v_idn"),
            "alias_dict": {"t": "t_train", "x": "x_train", "u_idn": "u_train", "v_idn": "v_train"},
        },
    }

    sup_constraint_idn = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_idn,
        MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        name="uv_mse_sup",
    )
    constraint_idn = {sup_constraint_idn.name: sup_constraint_idn}

    # maunally build validator
    eval_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn","v_idn"),
            "alias_dict": {"t": "t_star", "x": "x_star", "u_idn": "u_star", "v_idn": "v_star"},
        },
    }

    sup_validator_idn = ppsci.validate.SupervisedValidator(
        train_dataloader_cfg_idn,
        MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        {"l2": L2Rel()},
        name="uv_L2_sup",
    )
    validator_idn = {sup_validator_idn.name: sup_validator_idn}

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model_list,
        constraint=constraint_idn,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_idn,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_idn,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    test(model_idn_u, "u_idn", True)
    test(model_idn_v, "v_idn", True)

    # model 2: pde net
    # maunally build constraint(s)
    train_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x"),
            "alias_dict": {"t": "t_train", "x": "x_train"},
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        LossPde("sum"),
        {
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: jacobian(out["v_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
        },
        name="fg_mse_sup",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # maunally build validator
    eval_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x"),
            "alias_dict": {"t": "t_star", "x": "x_star"},
        },
    }

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_pde,
        LossPde("sum"),
        {
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: jacobian(out["v_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
        },
        {"l2": MatricPde()},
        name="fg_L2_sup",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # update solver
    solver = ppsci.solver.Solver(
        model=solver.model,
        constraint=constraint_pde,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_pde,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_pde,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # test(model_pde_f, "f_pde", True)
    # test(model_pde_g, "g_pde", True)

    # model 3: solution net
    # update transform of model 2
    del solver.model.model_list[0]
    del solver.model.model_list[0]
    model_pde_f.register_input_transform(transform=transform_fg_sol)
    model_pde_g.register_input_transform(transform=transform_fg_sol)
    # maunally build constraint(s)
    train_dataloader_cfg_sol_f = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "alias_dict": {"t": "t_f_train", "x": "x_f_train"},
        },
    }
    train_dataloader_cfg_sol_init = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_sol", "v_sol"),
            "alias_dict": {"t": "t0", "x": "x0", "u_sol": "u0", "v_sol": "v0"},
        },
    }
    train_dataloader_cfg_sol_bc = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "alias_dict": {"t": "tb", "x": "xb"},
        },
    }

    sup_constraint_sol_f = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_f,
        LossPde("sum"),
        {
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
            "du_t": lambda out: jacobian(out["u_sol"], out["t"]),
            "dv_t": lambda out: jacobian(out["v_sol"], out["t"]),
        },
        name="fg_mse_sup",
    )
    sup_constraint_sol_init = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_init,
        MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_sol", "v_sol")},
        name="uv0_mse_sup",
    )
    sup_constraint_sol_bc = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_bc,
        LossBoundary("sum"),
        {
            "ub_sol": lambda out: out["u_sol"],
            "vb_sol": lambda out: out["v_sol"],
            "x_b": lambda out: out["x"],
        },
        name="uvb_mse_sup",
    )
    constraint_sol = {
        sup_constraint_sol_f.name: sup_constraint_sol_f,
        sup_constraint_sol_init.name: sup_constraint_sol_init,
        sup_constraint_sol_bc.name: sup_constraint_sol_bc,
    }

    # maunally build validator
    eval_dataloader_cfg_sol = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_sol", "v_sol"),
            "alias_dict": {"t": "t_star", "x": "x_star", "u_sol": "u_star", "v_sol": "v_star"},
        },
    }

    sup_validator_sol = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_sol,
        MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_sol", "v_sol")},
        {"l2": L2Rel()},
        name="uv_L2_sup",
    )
    validator_sol = {
        sup_validator_sol.name: sup_validator_sol,
    }

    # update solver
    solver = ppsci.solver.Solver(
        model=solver.model,#ppsci.arch.ModelList((model_pde_f, model_pde_g, model_sol_u, model_sol_v))
        constraint=constraint_sol,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_sol,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_sol,
        # checkpoint_path = OUTPUT_DIR+"checkpoints/lastest"
    )

    # train model
    solver.train()

    del solver.model.model_list[0]
    del solver.model.model_list[0]
    # evaluate after finished training
    solver.eval()

    test(model_sol_u, "u_sol", True)
    test(model_sol_v, "v_sol", True, True)
