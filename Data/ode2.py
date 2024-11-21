import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
from scipy.interpolate import interp1d

fd = FiniteDifference(order=2, d=1)

num_curves = 2000
train_test_curves = 1000

# Generate random periods and amplitudes
amplitudes = np.random.uniform(0, 1, num_curves)

# Define x values
x_values = np.linspace(0, 1, 100)

samples = []

for i in range(num_curves):
    # Calculate y values for the sine curve
    sample = amplitudes[i] * np.sin(2 * pi * x_values)

    samples.append(sample)

# Convert samples to numpy array
samples = np.array(samples)

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
    y = y**2

    # Plot the hyst curve
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
np.savez_compressed('Exp2_Train.npz', X=x_values, voltage=samples_train, displacement=disp_train)
np.savez_compressed('Exp2_Test.npz', X=x_values, voltage=samples_test, displacement=disp_test)

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
