import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
import matplotlib.gridspec as gridspec

#############################################################################
#################### Importing Train and Test Data ##########################
#############################################################################

# input - v, output u

d = np.load("Data/Exp2_Train.npz", allow_pickle=True)
v_train, x_train, u_train = d["voltage"], d["X"], d["displacement"]

d = np.load("Data/Exp2_Test.npz", allow_pickle=True)
v_test, x_test, u_test = d["voltage"], d["X"], d["displacement"]

print("shape of v_train: ", v_train.shape)
print("shape of x_train: ", x_train.shape)
print("shape of u_train: ", u_train.shape)
print("shape of v_test: ", v_test.shape)
print("shape of x_test: ", x_test.shape)
print("shape of u_test: ", u_test.shape)

############ Error #####################

def relative_error(u_pred, u_exact):
    # Ensure the inputs are torch tensors
    u_pred = torch.tensor(u_pred, dtype=torch.float32)
    u_exact = torch.tensor(u_exact, dtype=torch.float32)

    # Compute the mean squared error
    mse = torch.mean((u_pred - u_exact)**2)

    # Compute the mean of the exact values squared
    mse_exact = torch.mean(u_exact**2)

    # Compute the relative error
    relative_error = torch.sqrt(mse / mse_exact)

    return relative_error

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

# Set random seed for reproducibility
torch.manual_seed(42)

# Toy problem data
input_size = 1000
hidden_size = 128
output_size = 1000
sequence_length = 100
batch_size = 1
num_epochs = 20000

v_train = v_train.T
u_train = u_train.T

# Convert data to tensors
input_tensor = torch.tensor(v_train).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(u_train).view(batch_size, sequence_length, output_size).float()

# Create LSTM instance
lstm = LSTM(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001)

loss_history = []
# Training loop
for epoch in range(num_epochs):
    # Set initial hidden state and cell state
    hidden = torch.zeros(1, batch_size, hidden_size)
    cell = torch.zeros(1, batch_size, hidden_size)

    # Forward pass
    output, (hidden, cell) = lstm(input_tensor, (hidden, cell))
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.8f}')

# Generate predictions on new range of values
input_tensor_pred = torch.tensor(v_test).view(batch_size, -1, input_size).float()
with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    cell_pred = torch.zeros(1, batch_size, hidden_size)
    prediction, _ = lstm(input_tensor, (hidden_pred, cell_pred))

print("pred shape: ",prediction.shape)

prediction = prediction.squeeze(0).transpose(0, 1)

v_train = v_train.T
u_train = u_train.T

# Plot the results
gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[:, :])

# Plot predictions
for i in range(10, 20):
    ax1.plot(v_test[i, :], prediction[i, :], '-k', lw=2.0)

# Plot actual values
for i in range(10, 20):
    ax1.plot(v_test[i, :], u_test[i, :], '--r', lw=2.0)

# Set labels with adequate font size
ax1.set_xlabel('Voltage [V]', fontsize=26)
ax1.set_ylabel('Displacement [$\mu$m]', fontsize=26)

# Adjust tick parameters
ax1.tick_params(axis='both', which='major', labelsize=26)

# Use tight_layout to automatically adjust subplot parameters to give specified padding
plt.tight_layout()

# Save the figure with adjusted bounding box to ensure no cutoff
plt.savefig("Results/LSTM/LSTM_Exp2.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

########## error #######################

error = relative_error(prediction, u_test)
print("Relative error: ", error.detach().numpy())

MAE = mean_absolute_error(prediction, u_test)
print('MAE: ', MAE)

MSE = mean_squared_error(prediction, u_test)
RMSE = np.sqrt(MSE)
print('RMSE: ', RMSE)
