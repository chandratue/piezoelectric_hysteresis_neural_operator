import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
from scipy.interpolate import interp1d
import GPy

fd = FiniteDifference(order=2, d=1)

num_curves = 2000
train_test_curves = 1000

# Define input space
X = np.linspace(0, 1, 100)[:, None]  # 100 points evenly spaced between 0 and 10

# Define a complex, non-periodic kernel by combining RBF, Linear, and Matern kernels
rbf_kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=0.2)

# Sample functions from the kernel
functions = rbf_kernel.K(X, X)

# Initialize lists for samples and track rejected samples
samples = []
rejected_samples = 0

# Generate samples with rejection sampling
while len(samples) < num_curves:
    # Generate a sample from the multivariate normal distribution with the kernel matrix K
    sample = np.random.multivariate_normal(np.zeros(X.shape[0]), functions)

    # Check if all values in the sample are within [-1, 1]
    if np.all(sample >= -1) and np.all(sample <= 1):
        samples.append(sample)
    else:
        rejected_samples += 1

# Convert samples to numpy array for easier manipulation
samples = np.array(samples)

# Optional: Print the number of rejected samples for debugging or efficiency check
print(f"Number of rejected samples: {rejected_samples}")

# Convert samples to numpy array
samples = np.array(samples)

plt.plot(X, samples.T)
plt.show()

x_values = X.reshape(100,)

##### Bouc-Wen Model

t = x_values

disp = []

# Define the ODE model
def model(y, t, x_interp, dx_interp):
    x_val = x_interp(t)
    dx_val = dx_interp(t)
    dydt = 0.4*np.abs(dx_val)*x_val - 0.85 * np.abs(dx_val)*y + 0.2*(dx_val)
    #dydt = 5 * dx_val - 0.25 * np.abs(dx_val) * y - 0.5 * (dx_val) * np.abs(y)
    return dydt

for i in range(num_curves):

    x = samples[i,:].T

    dx = fd._differentiate(x, t)

    # Create the interpolation function for x with extrapolation
    x_interp = interp1d(t, x, fill_value="extrapolate")

    dx_interp = interp1d(t, dx, fill_value="extrapolate")

    # Initial condition
    y0 = 0

    # Solve the ODE
    y = odeint(model, y0, t, args=(x_interp, dx_interp))

    # #Plot the hyst curve
    # plt.plot(x, y)
    # plt.xlabel('Voltage')
    # plt.ylabel('Displacement')
    # plt.show()

    disp.append(y)

    print('datasets saved: ', i)

# Convert samples to numpy array
disp = np.array(disp)

disp = np.squeeze(disp, axis=-1)

samples_train = samples[:train_test_curves, :]
disp_train = disp[:train_test_curves, :]

samples_test = samples[train_test_curves:, :]
disp_test = disp[train_test_curves:, :]

x_values = x_values.reshape(-1,1)

# Save X and samples in an .npz file
np.savez_compressed('Exp3_Train.npz', X=x_values, voltage=samples_train, displacement=disp_train)
np.savez_compressed('Exp3_Test.npz', X=x_values, voltage=samples_test, displacement=disp_test)

# # Load the .npz file
# data = np.load('Input_Exp1.npz')
#
# # Access the arrays using their names
# x_values_loaded = data['X']
# samples_loaded = data['voltage']
# disp_loaded = data['displacement']
#
# # Optional: print the shapes to confirm
# print(x_values_loaded.shape)
# print(samples_loaded.shape)
# print(disp_loaded.shape)
#
# plt.plot(samples_loaded[0,:], disp_loaded[0,:])
# plt.show()
