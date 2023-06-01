import paddle
from ppsci.loss import base,MSELoss
from ppsci.autodiff import jacobian, hessian


class LossPde(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, output_dict, label_dict, weight_dict=None):
        # output_dict-keys from constrains-keys
        out_dict = {"f_pde": output_dict["du_t"]}
        label_dict = {"f_pde": output_dict["f_pde"]}
        losses = super().forward(out_dict, label_dict, weight_dict)
        return losses


class LossBoundary(MSELoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, output_dict, label_dict, weight_dict=None):
        # output_dict-keys from constrains-keys
        u_b = output_dict["ub_sol"]
        u_lb, u_ub = paddle.split(u_b, 2, axis=0)

        du_xb = output_dict["du_xb_sol"]
        du_lb, du_ub = paddle.split(du_xb, 2, axis=0)

        out_dict = {"ub": u_lb, "du_b": du_lb}
        label_dict = {"ub": u_ub, "du_b": du_ub}

        losses = super().forward(out_dict, label_dict, weight_dict)
        return losses
