import torch
import torch.nn as nn

class MultiScalePINN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super(MultiScalePINN, self).__init__()
        self.layers = []
        activation_fn = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        self.activation = activation_fn.get(activation, nn.Tanh())

        # Main network
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            if i < len(layers) - 2:
                self.layers.append(self.activation)
        self.model = nn.Sequential(*self.layers)

        # Multi-scale branches
        self.scale_layers = nn.ModuleList()
        for i in range(3):
            scale_layers = []
            for j in range(len(layers) - 1):
                layer = nn.Linear(layers[j], layers[j + 1])
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                scale_layers.append(layer)
                if j < len(layers) - 2:
                    scale_layers.append(self.activation)
            self.scale_layers.append(nn.Sequential(*scale_layers))

    def forward(self, x):
        main_output = self.model(x)
        scale_outputs = [scale_model(x) for scale_model in self.scale_layers]
        scale_weights = [0.5, 0.3, 0.2]
        combined_output = main_output + sum(w * s for w, s in zip(scale_weights, scale_outputs))
        return combined_output