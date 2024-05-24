from torch import nn

N_FEATURES = 10
HIDDEN_LAYER_SIZE = 128
N_OUTPUTS = 1

# Define the model.
class DNN(nn.Module):
    def __init__(self, hidden_layers_num=5):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = [
            nn.Linear(N_FEATURES, HIDDEN_LAYER_SIZE),
            nn.ReLU()
        ]

        for _ in range(hidden_layers_num - 1):
            layers.append(nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(HIDDEN_LAYER_SIZE, N_OUTPUTS))
        self.linear_relu_stack = nn.Sequential(*layers)


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    