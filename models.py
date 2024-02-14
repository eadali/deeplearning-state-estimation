import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 23, 64)  # Calculated output size after convolutions and pooling
        self.fc2 = nn.Linear(64, 2)  # Output layer with 2 units for 2 output features

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StateEstimator:
    def __init__(self, model_path, window_size):
        self.nn_model = torch.load(model_path)
        self.window = np.full(window_size, np.nan)


    def update(self, meas):
        self.window = np.roll(self.window, -1, axis=0)
        self.window[-1] = meas

        if np.any(np.isnan(self.window)):
            return np.nan, np.nan
        
        with torch.no_grad():
            start = self.window[0].copy()
            reshape = self.window - start
            input_ = torch.tensor([reshape],  dtype=torch.float32)
            result = self.nn_model(input_.permute(0, 2, 1))
            res = result.numpy().squeeze()
            res_ = res + start

        return res_

