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
from paddle_science.scripts.data_loader_NS import DataLoader
from paddlepaddle.examples.NavierStokes import Plotting


def test(model, output_key, plot):
    data = DataLoader()
    data(DATASET_PATH, DATASET_PATH)
    mode = output_key

    t = paddle.to_tensor(data.t_star, dtype="float32")
    x = paddle.to_tensor(data.x_star, dtype="float32")
    y = paddle.to_tensor(data.y_star, dtype="float32")
    if mode == "w_sol":
        output_key = "w_idn"

    t.stop_gradient = False
    x.stop_gradient = False
    y.stop_gradient = False
    input_test = {"t": t, "x": x, "y": y}
    pred = model(input_test)[output_key].numpy()

    if mode in ["w_idn","w_sol"]:
        error_w = np.linalg.norm(data.w_star - pred, 2) / np.linalg.norm(
            data.w_star, 2
        )
        print("Error w: %e" % (error_w))
    elif mode == "f_pde":
        label = jacobian(model_idn(input_test)["w_idn"], t).numpy()
        error_f = np.linalg.norm(label - pred, 2) / np.sqrt(len(pred))
        print("Error f pde: %e" % (error_f))

    if plot:
        pred = np.reshape(pred, (-1, 151))
        plot = Plotting(
            "ns_lbfgs_" + mode,
            data.X_data,
        )
        plot.draw_t_2d(data.w_data, pred)
        plot.draw_n_save(data.w_data, pred)
        plot.plot_vorticity(data.x_star, data.y_star)


class LossPde(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, output_dict, label_dict, weight_dict=None):
        out_dict = {"f_pde": output_dict["f_pde"]}
        label_dict = {"f_pde": output_dict["dw_t"]}
        losses = super().forward(out_dict, label_dict, weight_dict)
        return losses


class MatricPde(L2Rel):
    def __init__(self):
        super().__init__()

    def forward(self, output_dict, label_dict):
        out_dict = {"pred": output_dict["f_pde"]}
        label_dict = {"pred": output_dict["dw_t"]}
        metric_dict = super().forward(out_dict, label_dict)
        return metric_dict


if __name__ == "__main__":
    paddle.fluid.core.set_prim_eager_enabled(True)
    ppsci.utils.misc.set_random_seed(42)
    EPOCHS = 1
    MAX_ITER = 50000    # for LBFGS
    ITERS_PER_EPOCH = 1
    LEARNING_RATE = 1e-4
    DATASET_PATH = "../dataset_gen/cylinder.mat"
    OUTPUT_DIR = "../outs_ns_lbfgs/"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize burgers boundaries
    # t, x, y
    lb = paddle.to_tensor([0.0, 1, -1.7], dtype="float32")
    ub = paddle.to_tensor([30.0, 7.5, 1.7], dtype="float32")

    # initialize models
    model_idn = ppsci.arch.MLP(("t", "x", "y"), ("w_idn",), 4, 50, "sin")
    model_pde = ppsci.arch.MLP(("u", "v", "w","dw_x", "dw_y", "dw_xx","dw_xy", "dw_yy"), ("f_pde",), 2, 100, "sin", 8)

    # initialize transform
    def transform_w(input):
        t, x, y = input["t"], input["x"], input["y"]
        X = paddle.concat([t, x, y], axis=1)
        H = 2.0 * (X - lb) * paddle.pow((ub - lb), -1) - 1.0
        t, x, y = paddle.split(H, 3, axis=1)
        input_trans = {"t": t, "x": x, "y": y}
        return input_trans

    def transform_f(input):
        in_idn = {"t": input["t"], "x": input["x"], "y": input["y"]}
        x, y = input["x"], input["y"]
        w = model_idn(in_idn)["w_idn"]
        dw_x = jacobian(w, x)
        dw_y = jacobian(w, y)

        dw_xx = hessian(w, x)
        dw_yy = hessian(w, y)
        dw_xy = jacobian(dw_x, y)

        input_trans = {"u": input["u"], "v": input["v"], "w":w, "dw_x":dw_x, "dw_y":dw_y, "dw_xx":dw_xx, "dw_xy":dw_xy, "dw_yy":dw_yy}
        return input_trans

    # register transform
    model_idn.register_input_transform(transform=transform_w)
    model_pde.register_input_transform(transform=transform_f)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn, model_pde))

    # initialize optimizer
    # Adam
    # optimizer_idn = ppsci.optimizer.Adam(LEARNING_RATE)([model_idn])
    # optimizer_pde = ppsci.optimizer.Adam(LEARNING_RATE)([model_pde])

    # LBFGS
    optimizer_idn = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)([model_idn])
    optimizer_pde = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)([model_pde])

    # # model 1: identification net
    # # maunally build constraint(s)
    # train_dataloader_cfg_idn = {
    #     "dataset": {
    #         "name": "IterableMatDataset",
    #         "file_path": DATASET_PATH,
    #         "input_keys": ("t", "x", "y", "u", "v"),
    #         "label_keys": ("w_idn",),
    #         "alias_dict": {"t": "t_train", "x": "x_train", "y": "y_train", "u": "u_train", "v": "v_train", "w_idn": "w_train"},
    #     },
    # }

    # sup_constraint_idn = ppsci.constraint.SupervisedConstraint(
    #     train_dataloader_cfg_idn,
    #     MSELoss("sum"),
    #     {key: (lambda out, k=key: out[k]) for key in ("w_idn",)},
    #     name="w_mse_sup",
    # )
    # constraint_idn = {sup_constraint_idn.name: sup_constraint_idn}

    # # maunally build validator
    # eval_dataloader_cfg_idn = {
    #     "dataset": {
    #         "name": "IterableMatDataset",
    #         "file_path": DATASET_PATH,
    #         "input_keys": ("t", "x", "y", "u", "v"),
    #         "label_keys": ("w_idn",),
    #         "alias_dict": {"t": "t_star", "x": "x_star", "y": "y_star", "u": "u_star", "v": "v_star", "w_idn": "w_star"},
    #     },
    # }

    # sup_validator_idn = ppsci.validate.SupervisedValidator(
    #     train_dataloader_cfg_idn,
    #     MSELoss("sum"),
    #     {key: (lambda out, k=key: out[k]) for key in ("w_idn",)},
    #     {"l2": L2Rel()},
    #     name="w_L2_sup",
    # )
    # validator_idn = {sup_validator_idn.name: sup_validator_idn}

    # # initialize solver
    # solver = ppsci.solver.Solver(
    #     model=model_list,
    #     constraint=constraint_idn,
    #     output_dir=OUTPUT_DIR,
    #     optimizer=optimizer_idn,
    #     epochs=EPOCHS,
    #     iters_per_epoch=ITERS_PER_EPOCH,
    #     eval_during_train=False,
    #     validator=validator_idn,
    #     # checkpoint_path="../outs_1/checkpoints/latest",
    # )

    # # train model
    # solver.train()
    # # evaluate after finished training
    # solver.eval()

    # test(model_idn, "w_idn", True)

    # model 2: pde net
    # maunally build constraint(s)
    train_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "alias_dict": {"t": "t_train", "x": "x_train","y": "y_train","u": "u_train","v": "v_train"},
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        LossPde("sum"),
        {
            "dw_t": lambda out: jacobian(out["w_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
        },
        name="f_mse_sup",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # maunally build validator
    eval_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "alias_dict": {"t": "t_star", "x": "x_star", "y": "y_star", "u": "u_star", "v": "v_star"},
        },
    }

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_pde,
        LossPde("sum"),
        {
            "dw_t": lambda out: jacobian(out["w_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
        },
        {"l2": MatricPde()},
        name="f_L2_sup",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # update solver
    solver = ppsci.solver.Solver(
        model=model_list,#solver.model,
        constraint=constraint_pde,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_pde,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_pde,
        checkpoint_path = OUTPUT_DIR+"checkpoints/lastest",
    )

    # train model
    solver.train()
    # evaluate after finished training
    # solver.eval()

    # test(model_pde, "f_pde", True)

    # model 3: solution net, reuse identification net
    # maunally build constraint(s)
    train_dataloader_cfg_sol_f = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "alias_dict": {"t": "t_f_train", "x": "x_f_train", "y": "y_f_train", "u": "u_f_train", "v": "v_f_train"},
        },
    }
    train_dataloader_cfg_sol_bc = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("wb_sol",),
            "alias_dict": {"t": "tb", "x": "xb", "y": "yb", "wb_sol": "wb", "u": "xb", "v": "yb"},
        },
    }

    sup_constraint_sol_f = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_f,
        LossPde("sum"),
        {
            "f_pde": lambda out: out["f_pde"],
            "dw_t": lambda out: jacobian(out["w_idn"], out["t"]),
        },
        name="f_mse_sup",
    )
    sup_constraint_sol_bc = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_bc,
        MSELoss("sum"),
        {
            "wb_sol": lambda out: out["w_idn"]
        },
        name="ub_mse_sup",
    )
    constraint_sol = {
        sup_constraint_sol_f.name: sup_constraint_sol_f,
        sup_constraint_sol_bc.name: sup_constraint_sol_bc,
    }

    # maunally build validator
    eval_dataloader_cfg_sol = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("w_sol",),
            "alias_dict": {"t": "t_star", "x": "x_star", "y": "y_star", "w_sol": "w_star", "u": "u_star", "v": "v_star"},
        },
    }

    sup_validator_sol = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_sol,
        MSELoss("sum"),
        {"w_sol": lambda out: out["w_idn"]},
        {"l2": L2Rel()},
        name="u_L2_sup",
    )
    validator_sol = {
        sup_validator_sol.name: sup_validator_sol,
    }

    # update solver
    solver = ppsci.solver.Solver(
        model=solver.model,
        constraint=constraint_sol,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_idn,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_sol,
    )

    # train model
    solver.train()
    # evaluate after finished training
    # solver.eval()

    # test(model_pde, "f_pde", True)
    test(model_idn, "w_sol", True)
