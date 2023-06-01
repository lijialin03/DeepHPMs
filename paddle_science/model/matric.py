from ppsci.metric import L2Rel


class MatricPde(L2Rel):
    def __init__(self):
        super().__init__()

    def forward(self, output_dict, label_dict):
        out_dict = {"pred": output_dict["f_pde"]}
        label_dict = {"pred": output_dict["du_t"]}
        metric_dict = super().forward(out_dict, label_dict)
        return metric_dict


class MatricSol(L2Rel):
    def __init__(self):
        super().__init__()

    def forward(self, output_dict, label_dict):
        out_dict = {"pred": output_dict["du_t_sol"]}
        label_dict = {"pred": output_dict["f_pde"]}
        metric_dict = super().forward(out_dict, label_dict)
        return metric_dict
