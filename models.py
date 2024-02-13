import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 11, 64)  # Calculated output size after convolutions and pooling
        self.fc2 = nn.Linear(64, 2)  # Output layer with 2 units for 2 output features

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StateEstimator:
    def __init__(self, nn_model, window_size):
        self.nn_model = nn_model
        self.window_size = window_size
        self.counter = 0
        self.window = []

    def update(self, meas):
        self.window.append(meas)
        if len(self.window) < self.window_size:
            return None
        
        with torch.no_grad():
            input_ = torch.tensor([self.window],  dtype=torch.float32))
            result = self.nn_model(input_.permute(0, 2, 1))

        self.window.pop(0)
        return result.numpy().squeeze()

