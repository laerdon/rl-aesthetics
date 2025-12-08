import torch

MAX_ITEMS = 5

class PlacementModel(torch.nn.Module):
    def __init__(self):
        super(PlacementModel, self).__init__()
        # Define model layers here
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2 + MAX_ITEMS * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, MAX_ITEMS * 2),  # Output x, y coordinates
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)