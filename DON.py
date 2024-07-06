#############################################################################
######################## Importing Libraries ###############################
############################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.gridspec as gridspec

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

#############################################################################
########################## DON Model #######################################
############################################################################

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

    # Initialization for DNNs
    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l - 1]
            out_dim = layers[l]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            weight = nn.Parameter(torch.randn(in_dim, out_dim) * std)
            bias = nn.Parameter(torch.randn(1, out_dim) * std)
            W.append(weight)
            b.append(bias)
        return W, b

    def fnn_B(self, X, W, b):
        A = X
        L = len(W)
        for i in range(L - 1):
            A = torch.tanh(torch.matmul(A, W[i]) + b[i])
        Y = torch.matmul(A, W[-1]) + b[-1]
        return Y

    def fnn_T(self, X, W, b):
        A = X
        L = len(W)
        for i in range(L - 1):
            A = torch.tanh(torch.matmul(A, W[i]) + b[i])
        Y = torch.matmul(A, W[-1]) + b[-1]
        return Y

# x (100, 1)--input trunk
# u (1000, 100)--
# v (1000, 100)--input branch

#############################################################################
################# Hyperparameters and Initialize model #####################
############################################################################

# Input dimension for Branch Net
u_dim = 100 # number of points on which input function is sampled

# Output dimension for Branch and Trunk Net
G_dim = 100 # number of basis functions and coefficients

# Branch Net
layers_f = [u_dim] + [40] * 2 + [G_dim]

# Trunk dim
x_dim = 1

# Trunk Net
layers_x = [x_dim] + [40] * 2 + [G_dim]

model = DNN()

def train_step(model, W_branch, b_branch, W_trunk, b_trunk, v, x, u, optimizer):
    train_vars = W_branch + b_branch + W_trunk + b_trunk
    optimizer.zero_grad()
    u_out_branch = model.fnn_B(v, W_branch, b_branch)
    u_out_trunk = model.fnn_T(x, W_trunk, b_trunk)
    # u_pred = torch.einsum('ij,jj->ij', u_out_branch, u_out_trunk)
    u_pred = torch.einsum('ij,kj->ik', u_out_branch, u_out_trunk)

    loss = torch.mean(torch.square(u_pred - u))
    loss.backward()
    optimizer.step()
    return loss.item(), u_pred.detach()


def test_step(model, W_branch, b_branch, W_trunk, b_trunk, v, x, u, optimizer):
    x_test_tensor = torch.from_numpy(x).float()
    v_test_tensor = torch.from_numpy(v).float()
    u_test_tensor = torch.from_numpy(u).float()

    with torch.no_grad():
        u_out_branch = model.fnn_B(v_test_tensor, W_branch, b_branch)
        u_out_trunk = model.fnn_T(x_test_tensor, W_trunk, b_trunk)
        # u_pred = torch.einsum('ij,jj->ij', u_out_branch, u_out_trunk)
        u_pred = torch.einsum('ij,kj->ik', u_out_branch, u_out_trunk)

        loss = torch.mean(torch.square(u_pred - u_test_tensor))

    return loss.item(), u_pred.detach().numpy()

W_branch, b_branch = model.hyper_initial(layers_f)
W_trunk, b_trunk = model.hyper_initial(layers_x)

n = 0
nmax = 1000
lr = 5e-4

optimizer = optim.Adam(list(W_branch) + list(b_branch) + list(W_trunk) + list(b_trunk), lr=lr)

train_err_list = []
test_err_list = []
train_loss_list = []
test_loss_list = []

#############################################################################
########################### Training model #################################
############################################################################

while n <= nmax:
    x_train_tensor = torch.from_numpy(x_train).float()
    v_train_tensor = torch.from_numpy(v_train).float()
    u_train_tensor = torch.from_numpy(u_train).float()

    loss_train, u_train_pred = train_step(model, W_branch, b_branch, W_trunk,
                                          b_trunk, v_train_tensor, x_train_tensor, u_train_tensor,
                                          optimizer)
    #     #err_train = np.mean(np.linalg.norm(u_train - u_train_pred, 2, axis=1) /
    #                         np.linalg.norm(u_train, 2, axis=1))

    u_train_pred_np = u_train_pred.detach().numpy()

    err_train = np.mean(np.linalg.norm(u_train - u_train_pred_np, 2, axis=1) /
                        np.linalg.norm(u_train, 2, axis=1))

    loss_test, u_test_pred = test_step(model, W_branch, b_branch, W_trunk, b_trunk, v_test, x_test, u_test,
                                       optimizer)
    err_test = np.mean(np.linalg.norm(u_test - u_test_pred, 2, axis=1) /
                       np.linalg.norm(u_test, 2, axis=1))

    if n % 100 == 0:
        print(f"Iteration: {n} Train_loss:{loss_train}, Test_loss: {loss_test}")
    train_err_list.append(err_train)
    test_err_list.append(err_test)
    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)
    n = n + 1

#############################################################################
########################### Loss plots #####################################
############################################################################

train_loss = np.array(train_loss_list)
test_loss = np.array(test_loss_list)
plot_int = 100

ep = np.arange(0, nmax + 1, 1)
ep = ep[::plot_int]

train_loss_plot = train_loss[::plot_int]
test_loss_plot = test_loss[::plot_int]

gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[:, :])
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.set_ylim([10**-5, 1])
ax.semilogy(ep, train_loss_plot, '--b', lw=2.0, label="Train")
ax.semilogy(ep, test_loss_plot, '-k', lw=1.0, label="Test")
plt.legend()
plt.savefig("Results/DON/Loss.png", dpi=300)
plt.show()

#############################################################################
########################### Testing model ##################################
############################################################################

gs1 = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs1[:, :])

ax1.plot(x_test, u_test_pred[20, :], '-k', lw=2.0, label="Predicted $u(x)$")

for i in range(10, 20):
    ax1.plot(x_test, u_test_pred[i, :], '-k', lw=2.0)

ax1.plot(x_test, u_test[20, :], '--r', lw=2.0, label="Actual $u(x)$")

for i in range(10, 20):
    ax1.plot(x_train, u_test[i, :], '--r', lw=2.0)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$u$')
ax1.legend(frameon=True, loc='best')
plt.savefig("Results/DON/predicted_and_deeponet.png", dpi=300)
plt.show()

gs2 = gridspec.GridSpec(1, 1)
ax2 = plt.subplot(gs2[:, :])

ax2.plot(x_test, v_test[20, :], '-g', lw=2.0, label="$v(x)$")
for i in range(10, 20):
    ax2.plot(x_test, v_test[i, :], '-g', lw=2.0)

ax2.set_xlabel('$x$')
ax2.set_ylabel('$v(x)$')
ax2.legend(frameon=True, loc='best')
plt.savefig("Results/DON/forcing.png", dpi=300)