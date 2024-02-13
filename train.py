import pandas as pd
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import CNN

# Parameters
window_size = 50

# Load data
measurements = pd.read_csv('./data/measurements.csv', index_col=0)
states = pd.read_csv('./data/states.csv', index_col=0)
X = measurements[['X','Y']].to_numpy()
y = states[['X','Y']].to_numpy()

# Preprocess data 
X_window = sliding_window_view(X, (window_size, X.shape[1])).squeeze()
y_window = y[window_size-1:,:]
starts = X_window[:,0,:].copy()
X_train = X_window - starts.reshape(9951, 1, 2)
y_train = y_window - starts



# Instantiate the model
model = CNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# Train the model
epochs = 500
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor.permute(0, 2, 1))
#     loss = criterion(outputs, Y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# torch.save(model, 'saved_model')
model = torch.load('saved_model')
model.eval()
with torch.no_grad():
    test_tensor = torch.tensor([X_train[0]], dtype=torch.float32)
    test = model(test_tensor.permute(0, 2, 1))






out = test.numpy()
print(out)
print()
plt.plot(out[:,0] + starts[:,0])
plt.plot(X[49:,0])
plt.plot(y[49:,0])
# plt.plot(out[:,0] + X_window[:,0,0])
plt.show()


exit()
# plt.show()

# X_window = np.squeeze(X_window)
# y_short = np.squeeze(y_short)
stype(np.float32)

print(X_train.shape)
print(y_train.shape)



# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define input size, hidden size, and output size
input_size = 50 * 2  # Assuming input is flattened
hidden_size = 128    # You can adjust this as needed
output_size = 2  # Output size is (10x6)

# Create an instance of the MLP
model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Assuming you have your data in numpy arrays X_train and y_train
# X_train = np.random.rand(9990, 10 * 9).astype(np.float32)  # Shape: (9990, 10*9)
# y_train = np.random.rand(9990, 10 * 6).astype(np.float32)  # Shape: (9990, 10*6)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)



# Training loop
epochs = 1000
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# torch.save(model, 'saved_model')
model = torch.load('saved_model')
model.eval()
with torch.no_grad():
    test = model(X_train_tensor)






out = test.numpy()


plt.plot(out[:,0] + starts[:,0])
plt.plot(y[:,0])
# plt.plot(out[:,0] + starts)
plt.show()



# plt.show()

# X_window = np.squeeze(X_window)
# y_short = np.squeeze(y_short)
