import ppsci


class DeepHPMs(object):
    def __init__(self, model_list, output_dir) -> None:
        self.model = model_list
        self.output_dir = output_dir

    def set_proc_params(
        self, opt, epochs, iters_per_epoch, constraint, validator=None, visualizer=None
    ):
        self.opt = opt
        self.epochs = epochs
        self.iters_per_epoch = iters_per_epoch
        self.constraint = constraint
        self.validator = validator
        self.visualizer = visualizer

    def gen_cfg(
        self,
        file_path,
        input_keys,
        label_keys=(),
        alias_dict={},
        dataset_name="IterableMatDataset",
    ):
        # cfg = {
        #     "dataset": {
        #         "name": dataset_name,
        #         "file_path": file_path,
        #         "input_keys": input_keys,
        #         "label_keys": label_keys,
        #         "alias_dict": alias_dict,
        #     },
        # }
        cfg = {
            "dataset": {
                "name": "MatDataset",
                "file_path": file_path,
                "input_keys": input_keys,
                "label_keys": label_keys,
                "alias_dict": alias_dict,
            },
            "sampler": {
                "name": "RandomSampler",
            },
            # "batch_size": 20000,
            # "sampler": {
            #     "name": "BatchSampler",
            #     "drop_last": True,
            #     "shuffle": False,
            # },
        }
        return cfg

    def gen_visualizer(self):
        num_sample = 10
        # set geometry
        geom = {"rect": ppsci.geometry.Rectangle((0.0, -8.0), (10.0, 8.0))}
        # set visualizer(optional)
        vis_points_xy = geom["rect"].sample_interior(num_sample, evenly=True)
        vis_points = {"t": vis_points_xy["x"], "x": vis_points_xy["y"]}
        # print(vis_points)

        self.visualizer = {
            "visulzie_u": ppsci.visualize.VisualizerVtu(
                vis_points,
                {"u": lambda out: out[self.out_expr_key[0]]},
                batch_size=num_sample,
                num_timestamps=1,
                prefix="result_u",
            )
        }

        """self.visualizer = {
            "visulzie_u": ppsci.visualize.Visualizer2DPlot(
                vis_points,
                {"u": lambda d: d[str_out]},
                batch_size=1,
                num_timestamps=1,
                stride=1,
                xticks=np.linspace(0, 10, 200),
                yticks=np.linspace(-8, 8, 256),
                prefix="result_states",
            )
        }"""

    def gen_solver(self):
        # initialize solver
        return ppsci.solver.Solver(
            self.model,
            constraint=self.constraint,
            output_dir=self.output_dir,
            optimizer=self.opt,
            epochs=self.epochs,
            iters_per_epoch=self.iters_per_epoch,
            # save_freq=self.epochs,
            # log_freq=1,
            eval_during_train=False,
            validator=self.validator,
            visualizer=self.visualizer,
            checkpoint_path=self.checkpoint_path,
        )

    def run(self, solver):
        self.solver = solver

        # train model
        self.solver.train()
        if self.validator:
            # evaluate after finished training
            self.solver.eval()
        if self.visualizer:
            # visualize prediction after finished training
            self.solver.visualize()
