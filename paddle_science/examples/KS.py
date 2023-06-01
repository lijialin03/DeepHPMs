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
from paddle_science.scripts.data_loader import DataLoader
from paddlepaddle.scripts.plotting import Plotting


def test(model, output_key, plot):
    data = DataLoader()
    data(DATASET_PATH, DATASET_PATH_SOL)
    mode = output_key
    if mode in ["u_idn", "f_pde"]:
        t = paddle.to_tensor(data.t_idn_star, dtype="float32")
        x = paddle.to_tensor(data.x_idn_star, dtype="float32")
    elif mode == "u_sol":
        t = paddle.to_tensor(data.t_sol_star, dtype="float32")
        x = paddle.to_tensor(data.x_sol_star, dtype="float32")
        output_key = "u_idn"

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
            "ks_reuse_lbfgs_" + mode,
            [0.0, -10.0],
            [50.0, 10.0],
            # [0.0, 0.0],
            # [100.0, 32.0 * np.pi],
            data.X_sol_star,
            (data.T_sol, data.X_sol),
        )
        plot.draw_n_save(data.Exact_sol, pred)
        plot.draw_t_2d(data.Exact_sol, pred)


class LossPde(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, output_dict, label_dict, weight_dict=None):
        out_dict = {"f_pde": output_dict["f_pde"]}
        label_dict = {"f_pde": output_dict["du_t"]}
        losses = super().forward(out_dict, label_dict, weight_dict)
        return losses


class LossBoundary(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def forward(self, output_dict, label_dict, weight_dict=None):
        u_b = output_dict["ub_sol"]
        u_lb, u_ub = paddle.split(u_b, 2, axis=0)

        x_b = output_dict["x_b"]
        du_x = jacobian(u_b, x_b)
        du_xx = hessian(u_b, x_b)
        du_xxx = jacobian(du_xx, x_b)

        # du_x = output_dict["du_x_sol"]
        du_x_lb, du_x_ub = paddle.split(du_x, 2, axis=0)

        # du_xx = output_dict["du_xx_sol"]
        du_xx_lb, du_xx_ub = paddle.split(du_xx, 2, axis=0)

        # du_xxx = output_dict["du_xxx_sol"]
        du_xxx_lb, du_xxx_ub = paddle.split(du_xxx, 2, axis=0)

        losses = super().forward({"ub": u_lb}, {"ub": u_ub}, weight_dict)
        losses += super().forward({"du_x_b": du_x_lb}, {"du_x_b": du_x_ub}, weight_dict)
        losses += super().forward({"du_xx_b": du_xx_lb}, {"du_xx_b": du_xx_ub}, weight_dict)
        losses += super().forward({"du_xxx_b": du_xxx_lb}, {"du_xxx_b": du_xxx_ub}, weight_dict)
        return losses


class MatricPde(L2Rel):
    def __init__(self):
        super().__init__()

    def forward(self, output_dict, label_dict):
        out_dict = {"pred": output_dict["f_pde"]}
        label_dict = {"pred": output_dict["du_t"]}
        metric_dict = super().forward(out_dict, label_dict)
        return metric_dict


if __name__ == "__main__":
    paddle.fluid.core.set_prim_eager_enabled(True)
    ppsci.utils.misc.set_random_seed(42)
    EPOCHS = 1
    MAX_ITER = 50000    # for LBFGS
    ITERS_PER_EPOCH = 1
    LEARNING_RATE = 1e-3
    DATASET_PATH = "../dataset_gen/KS.mat"
    DATASET_PATH_SOL = "../dataset_gen/KS.mat"
    OUTPUT_DIR = "../outs_ks_reuse_lbfgs/"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize burgers boundaries
    t_lb = paddle.to_tensor([0.0], dtype="float32")
    t_ub = paddle.to_tensor([50.0], dtype="float32")
    x_lb = paddle.to_tensor([-10.0], dtype="float32")
    x_ub = paddle.to_tensor([10.0], dtype="float32")

    # t_lb = paddle.to_tensor([0.0], dtype="float32")
    # t_ub = paddle.to_tensor([100.0], dtype="float32")
    # x_lb = paddle.to_tensor([0.0], dtype="float32")
    # x_ub = paddle.to_tensor([32.0 * np.pi], dtype="float32")

    # initialize models
    model_idn = ppsci.arch.MLP(("t", "x"), ("u_idn",), 4, 50, "sin")
    model_pde = ppsci.arch.MLP(("u_x", "du_x", "du_xx", "du_xxx", "du_xxxx"), ("f_pde",), 2, 100, "sin", 5)

    # initialize transform
    def transform_u(input):
        t, x = input["t"], input["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_f(input,model_trans,out_key):
        in_idn = {"t": input["t"], "x": input["x"]}
        x = input["x"]
        u = model_trans(in_idn)[out_key]
        du_x = jacobian(u, x)
        du_xx = hessian(u, x)
        du_xxx = jacobian(du_xx, x)
        du_xxxx = hessian(du_xx, x)
        input_trans = {"u_x": u, "du_x": du_x, "du_xx": du_xx,"du_xxx": du_xxx,"du_xxxx":du_xxxx}
        return input_trans

    def transform_f_idn(input):
        return transform_f(input,model_idn, "u_idn")

    # register transform
    model_idn.register_input_transform(transform=transform_u)
    model_pde.register_input_transform(transform=transform_f_idn)

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
    #         "input_keys": ("t", "x"),
    #         "label_keys": ("u_idn",),
    #         "alias_dict": {"t": "t_train", "x": "x_train", "u_idn": "u_train"},
    #     },
    # }

    # sup_constraint_idn = ppsci.constraint.SupervisedConstraint(
    #     train_dataloader_cfg_idn,
    #     MSELoss("sum"),
    #     {key: (lambda out, k=key: out[k]) for key in ("u_idn",)},
    #     name="u_mse_sup",
    # )
    # constraint_idn = {sup_constraint_idn.name: sup_constraint_idn}

    # # maunally build validator
    # eval_dataloader_cfg_idn = {
    #     "dataset": {
    #         "name": "IterableMatDataset",
    #         "file_path": DATASET_PATH,
    #         "input_keys": ("t", "x"),
    #         "label_keys": ("u_idn",),
    #         "alias_dict": {"t": "t_star", "x": "x_star", "u_idn": "u_star"},
    #     },
    # }

    # sup_validator_idn = ppsci.validate.SupervisedValidator(
    #     train_dataloader_cfg_idn,
    #     MSELoss("sum"),
    #     {key: (lambda out, k=key: out[k]) for key in ("u_idn",)},
    #     {"l2": L2Rel()},
    #     name="u_L2_sup",
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
    # test(model_idn, "u_idn", True)

    # # model 2: pde net
    # # maunally build constraint(s)
    # train_dataloader_cfg_pde = {
    #     "dataset": {
    #         "name": "IterableMatDataset",
    #         "file_path": DATASET_PATH,
    #         "input_keys": ("t", "x"),
    #         "alias_dict": {"t": "t_train", "x": "x_train"},
    #     },
    # }

    # sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
    #     train_dataloader_cfg_pde,
    #     LossPde("sum"),
    #     {
    #         "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
    #         "f_pde": lambda out: out["f_pde"],
    #     },
    #     name="f_mse_sup",
    # )
    # constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # # maunally build validator
    # eval_dataloader_cfg_pde = {
    #     "dataset": {
    #         "name": "IterableMatDataset",
    #         "file_path": DATASET_PATH,
    #         "input_keys": ("t", "x"),
    #         "alias_dict": {"t": "t_star", "x": "x_star"},
    #     },
    # }

    # sup_validator_pde = ppsci.validate.SupervisedValidator(
    #     eval_dataloader_cfg_pde,
    #     LossPde("sum"),
    #     {
    #         "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
    #         "f_pde": lambda out: out["f_pde"],
    #     },
    #     {"l2": MatricPde()},
    #     name="f_L2_sup",
    # )
    # validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # # update solver
    # solver = ppsci.solver.Solver(
    #     model=solver.model,
    #     constraint=constraint_pde,
    #     output_dir=OUTPUT_DIR,
    #     optimizer=optimizer_pde,
    #     epochs=EPOCHS,
    #     iters_per_epoch=ITERS_PER_EPOCH,
    #     eval_during_train=False,
    #     validator=validator_pde,
    #     # checkpoint_path = "../outs_ks/checkpoints/latest",
    # )

    # # train model
    # solver.train()
    # # evaluate after finished training
    # solver.eval()

    # model 3: solution net, reuse identification net
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
            "label_keys": ("u_idn",),
            "alias_dict": {"t": "t0", "x": "x0", "u_idn": "u0"},
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
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
        },
        name="f_mse_sup",
    )
    sup_constraint_sol_init = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_init,
        MSELoss("sum"),
        {"u_idn": lambda out: out["u_idn"]},
        name="u0_mse_sup",
    )
    sup_constraint_sol_bc = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_bc,
        LossBoundary("sum"),
        {
            "ub_sol": lambda out: out["u_idn"],
            "x_b": lambda out: out["x"],
        },
        name="ub_mse_sup",
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
            "label_keys": ("u_idn",),
            "alias_dict": {"t": "t_star", "x": "x_star", "u_idn": "u_star"},
        },
    }

    sup_validator_sol = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_sol,
        MSELoss("sum"),
        {"u_idn": lambda out: out["u_idn"]},
        {"l2": L2Rel()},
        name="u_L2_sup",
    )
    validator_sol = {
        sup_validator_sol.name: sup_validator_sol,
    }

    # update solver
    solver = ppsci.solver.Solver(
        model=model_list,#solver.model,
        constraint=constraint_sol,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_idn,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_sol,
        checkpoint_path = OUTPUT_DIR+"checkpoints/lastest"
    )

    # # train model
    # solver.train()
    # # evaluate after finished training
    # solver.eval()

    test(model_pde, "f_pde", True)
    test(model_idn, "u_sol", True)
