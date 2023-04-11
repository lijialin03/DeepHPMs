import paddle


class Network(paddle.nn.Layer):
    """Network"""

    def __init__(self, layer_sizes):
        super().__init__()
        self.initializer_w = paddle.nn.initializer.XavierNormal()
        self.initializer_b = paddle.nn.initializer.Constant(value=0.0)

        self.linears = paddle.nn.LayerList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.initializer_w(self.linears[-1].weight)
            self.initializer_b(self.linears[-1].bias)

        self.activation = paddle.sin

    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        outputs = self.linears[-1](x)
        return outputs
