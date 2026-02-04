from torch import nn


class MLH_l(nn.Module):
    def __init__(self):
        super(MLH_l, self).__init__()
        self.init_layer_size = 1
        self.final_layer_size = 9
        layer_size = 1000
        self.fc1 = nn.Linear(self.init_layer_size, layer_size)
        self.norm = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size, bias=False)
        self.fc_last = nn.Linear(layer_size, self.final_layer_size, bias=False)

    def update_init_layer(self, size: int):
        self.fc1 = nn.Linear(size, 1000)
        self.init_layer_size = size

    def update_final_layer(self, size: int):
        self.fc_last = nn.Linear(1000, size, bias=False)
        self.final_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = nn.functional.softshrink(self.fc1(x), lambd=1e-7)
        x = self.norm(x)
        x = nn.functional.tanhshrink(self.fc2(x))
        x = nn.functional.tanh(self.fc3(x))
        x = nn.functional.tanhshrink(self.fc4(x))
        x = nn.functional.hardshrink(self.fc_last(x), lambd=1e-5)

        return x


class MLH_s(nn.Module):
    def __init__(self):
        super(MLH_s, self).__init__()
        self.init_layer_size = 1
        self.final_layer_size = 9
        layer_size = 700
        self.fc1 = nn.Linear(self.init_layer_size, layer_size)
        self.norm = nn.LayerNorm(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size, bias=False)
        self.fc_last = nn.Linear(layer_size, self.final_layer_size, bias=False)

    def update_init_layer(self, size: int):
        self.fc1 = nn.Linear(size, 700)
        self.init_layer_size = size

    def update_final_layer(self, size: int):
        self.fc_last = nn.Linear(700, size, bias=False)
        self.final_layer_size = size

    def forward(self, x):
        x = x.view(-1, self.init_layer_size)  # Flatten the input
        x = nn.functional.softshrink(self.fc1(x), lambd=1e-7)
        x = self.norm(x)
        x = nn.functional.tanhshrink(self.fc2(x))
        x = nn.functional.tanh(self.fc3(x))
        x = nn.functional.tanhshrink(self.fc4(x))
        x = nn.functional.hardshrink(self.fc_last(x), lambd=1e-5)

        return x