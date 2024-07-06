#############################################################################
######################## Importing Libraries ###############################
############################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

#############################################################################
#################### Importing Train and Test Data ##########################
############################################################################

# input - v, output u

d = np.load("Data/antiderivative_train.npz", allow_pickle=True)
v_train, x_train, u_train = d["X"][0], d["X"][1], d["y"]

d = np.load("Data/antiderivative_test.npz", allow_pickle=True)
v_test, x_test, u_test = d["X"][0], d["X"][1], d["y"]

print("shape of v_train: ", v_train.shape)
print("shape of x_train: ", x_train.shape)
print("shape of u_train: ", u_train.shape)
print("shape of v_test: ", v_test.shape)
print("shape of x_test: ", x_test.shape)
print("shape of u_test: ", u_test.shape)

#############################################################################
###################### Visualizing Imported Data ###########################
############################################################################

# Plotting
fig = plt.figure()
gs0 = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs0[:, :])
ax.plot(x_train, v_train[10,:], '-k', lw=2.0, label="$v(x)$")
ax.plot(x_train, u_train[10,:], '-b', lw=2.0, label="$u(x)$")
ax.plot(x_train, v_test[100,:], '--k', lw=2.0, label="$v(x)$")
ax.plot(x_train, u_test[100,:], '--b', lw=2.0, label="$u(x)$")
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.legend(frameon=False, loc='best')
ax.set_title('$v \sim \mathcal{GP}(0, cov(x,x))$', fontsize=10)
plt.savefig("Results/DON/Data.png", dpi=300)
plt.show()

#========================#
# Training parameters
#========================#
num_epoch = 1000   #500               # number of training epoch
batch_size = 100                  # batch size

# Adam optimizer parameters
lr = 0.001                       # learning rate

modes = 4    #16                   # number of Fourier modes to multiply, at most floor(N/2) + 1
width = 8    #64                   # number of hidden channel

#========================#
# dataset information
#========================#
# load training data


v_train = torch.Tensor(v_train)
u_train = torch.Tensor(u_train)

v_test = torch.Tensor(v_test)
u_test = torch.Tensor(u_test)

# prepare grid information (optional)
grid_all = x_train.reshape(100, 1).astype(np.float64)
grid = torch.tensor(grid_all, dtype=torch.float)

grid_all_test = x_test.reshape(100, 1).astype(np.float64)
grid_test = torch.tensor(grid_all_test, dtype=torch.float)

print(grid.shape)

# concatenate the spatial grid and the spatial solution
v_train = torch.cat([v_train.reshape(v_train.shape[0], -1, 1), grid.repeat(v_train.shape[0], 1, 1)], dim=2)
v_test = torch.cat([v_test.reshape(v_test.shape[0], -1, 1), grid_test.repeat(v_test.shape[0], 1, 1)], dim=2)
print(f'[Dataset] v_train: {v_train.shape}, u_train: {u_train.shape}')
print(f'[Dataset] v_test: {v_test.shape}, u_test: {u_test.shape}')

# create data loader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(v_train, u_train),
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(v_test, u_test),
    batch_size=v_test.shape[0],
    shuffle=False
)

class SpectralConv1d(nn.Module):
    """
    Spectral Convolutional 1D Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of Fourier modes to multiply, at most floor(N/2) + 1.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of Fourier modes to multiply, at most floor(N/2) + 1.
        scale (float): Scaling factor for the weights.
        weights1 (nn.Parameter): Learnable weights for the convolution.

    Methods:
        compl_mul1d(input, weights): Performs complex multiplication between input and weights.
        forward(x): Forward pass of the SpectralConv1d layer.
    """

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes are kept, at most floor(N/2) + 1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        """
        Performs complex multiplication between input and weights.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, in_channel, x).
            weights (torch.Tensor): Weights tensor of shape (in_channel, out_channel, x).

        Returns:
            torch.Tensor: Result of complex multiplication of input and weights, of shape (batch, out_channel, x).
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # in_channel is the common dimension along which the operation is performed.
        # It suggests that every channel (or feature) of the input is being transformed into a new set of channels (or features) in the output.
        # The operation iterates over each batch and each position x, multiplying the input channels
        # by the corresponding weights and summing the results to produce the output channels.

        # For a given position i in the x dimension and a given batch element:
        # Take all values in A at this position and batch (A[batch, :, i]) – a slice of shape (in_channel).
        # Take all corresponding values in B at this position (B[:, :, i]) – a matrix of shape (in_channel, out_channel).
        # Multiply these values element-wise and sum over the in_channel dimension.
        # This produces a vector of shape (out_channel), representing the transformed features at position i for this batch element.
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # Compute Fourier coeffcients
        x_ft = torch.fft.rfft(x)  # [Batch, C_in, Nx] -> [Batch, C_in, Nx//2 + 1], eg. [20, 64, 128] -> [20, 64, 65]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat) # [Batch, Nc, Nx//2 + 1], eg. [20, 64, 65]
        # [Batch, C_in, self.modes1] * [C_in, C_out, self.modes1] -> [Batch, C_out, self.modes1]
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # [Batch, C_out, self.modes1] -> [Batch, C_out, Nx], eg. [20, 64, 65] -> [20, 64, 128]
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        """
        1D Fourier Neural Operator model.

        Args:
            modes (int): Number of spectral modes.
            width (int): Number of hidden channel.
        """
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x) MeshgridTensor + initial condition

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # stage 1: lift the channel from 2 to self.width = 64
        x = self.fc0(x)         # [Batch, Nx, C] -> [Batch, Nx, Width], eg. [20, 128, 2] -> [20, 128, 64]
        x = x.permute(0, 2, 1)  # [Batch, C, Nx], eg. [20, 64, 128]

        # stage 2: integral operators u' = (W + K)(u).
        # W is the linear transformation; K is the spectral convolution kernel.
        x1 = self.conv0(x)      # [Batch, C, Nx], eg. [20, 64, 128]
        x2 = self.w0(x)         # [Batch, C, Nx], eg. [20, 64, 128]
        x = x1 + x2
        x = F.relu(x)           # [Batch, C, Nx], eg. [20, 64, 128]

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)           # [Batch, C, Nx], eg. [20, 64, 128]

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)           # [Batch, C, Nx], eg. [20, 64, 128]

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2             # [Batch, C, Nx], eg. [20, 64, 128]

        # stage 3: put the channel back to 1
        x = x.permute(0, 2, 1)  # [Batch, Nx, C], eg. [20, 128, 64]
        x = self.fc1(x)         # [Batch, Nx, C] -> [Batch, Nx, 128], eg. [20, 128, 64] -> [20, 128, 128]
        x = F.relu(x)
        x = self.fc2(x)         # [Batch, Nx, C] -> [Batch, Nx, 1], eg. [20, 128, 128] -> [20, 128, 1]

        return x

# define a model
model = FNO1d(modes, width)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# loss_func = LpLoss(size_average=False) # used in original FNO paper
loss_func = nn.MSELoss() # MSE loss

loss_history_train, loss_history_test = [], []
# start training
for epoch in range(num_epoch):
    model.train()
    for x, y in train_loader:
        x, y = x, y
        optimizer.zero_grad()
        out = model(x)

        loss_train = loss_func(out.view(batch_size, -1), y.view(batch_size, -1))
        loss_train.backward()

        optimizer.step()
        loss_history_train.append(loss_train.item())

    #scheduler.step()  # change the learning rate
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x, y
            out = model(x)
            loss_test = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    loss_history_test.append(loss_test)

    if epoch % (num_epoch // 10) == 0:
        print(f'[Training] Epoch: {epoch}, loss_train: {loss_history_train[-1]:.2e}, loss_test: {loss_history_test[-1]:.2e}')

print('[Training] Finished.')

# plot loss history for training and test sets
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(loss_history_train)), loss_history_train, label='train', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.yscale('log')
plt.title('Training Loss')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(loss_history_test)), loss_history_test, label='test', color='red')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.yscale('log')
plt.title('Test Loss')
plt.show()

#############################################################################
########################### Testing model ##################################
############################################################################

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x, y
        u_test_pred = model(x)

gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[:, :])

ax1.plot(x_test, u_test_pred[10, :, 0], '-k', lw=2.0, label="Predicted $u(x)$")

for i in range(10, 20):
    ax1.plot(x_test, u_test_pred[i, :, 0], '-k', lw=2.0)

ax1.plot(x_test, u_test[10, :], '--r', lw=2.0, label="Actual $u(x)$")

for i in range(10, 20):
    ax1.plot(x_train, u_test[i, :], '--r', lw=2.0)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$u$')
ax1.legend(frameon=True, loc='best')
plt.savefig("Results/FNO/predicted_and_deeponet.png", dpi=300)
plt.show()

gs2 = gridspec.GridSpec(1, 1)
ax2 = plt.subplot(gs2[:, :])

ax2.plot(x_test, v_test[20, :], '-g', lw=2.0, label="$v(x)$")
for i in range(10, 20):
    ax2.plot(x_test, v_test[i, :], '-g', lw=2.0)

ax2.set_xlabel('$x$')
ax2.set_ylabel('$v(x)$')
ax2.legend(frameon=True, loc='best')
plt.savefig("Results/FNO/forcing.png", dpi=300)