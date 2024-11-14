import PyTPSA
import numpy as np
import matplotlib.pyplot as plt
import copy

# The number of variables considered (4-dimensional system: x, px, y, py)
num_var = 4

# Maximum order of PyTpsa considered for the calculation
max_order = 4
PyTPSA.initialize(num_var, max_order)

# Butcher tableau for the Runge-Kutta (RK) method (5th order)
RK5 = np.array([[0., 0., 0., 0., 0., 0., 0.],
                [0.25, 0.25, 0., 0., 0., 0., 0.],
                [3 / 8, 3 / 32., 9 / 32, 0., 0., 0., 0.],
                [12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197, 0., 0., 0.],
                [1, 439 / 216, -8, 3680 / 513, -845 / 4104, 0., 0.],
                [1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0.],
                [0., 16 / 135, 0., 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]])

A = RK5[:-1, 1:]
c = RK5[:-1, 0]
b = RK5[-1, 1:]

# Perform one step of the Runge-Kutta integration method
def one_step(f, sn: float, cor_n: list, h: float = 0.1):
    k_derive = np.zeros_like(b).tolist()
    
    # Compute the intermediate stages of the RK method
    for i, A_row in enumerate(A):
        cor_n_temp = [(cor_n[0]).copy(), (cor_n[1]).copy(), (cor_n[2]).copy(), (cor_n[3]).copy()]
        for j in range(i):
            for cor_index in range(len(cor_n_temp)):
                cor_n_temp[cor_index] = cor_n_temp[cor_index] + (A[i, j] * h) * k_derive[j][cor_index]
        k_derive[i] = f(sn + c[i] * h, cor_n_temp.copy())
    
    # Compute the final result for the step using the RK weights
    res = [(cor_n[0]).copy(), (cor_n[1]).copy(), (cor_n[2]).copy(), (cor_n[3]).copy()]
    for index in range(len(k_derive)):
        for cor_index in range(len(cor_n_temp)):
            res[cor_index] = res[cor_index] + (h * b[index]) * k_derive[index][cor_index]
    
    return res

# Perform the full Runge-Kutta integration over a series of steps
def ode(f, sn, cor_n_0):
    cor_n_list = []
    cor_n_list.append([(cor_n_0[0]).copy(), (cor_n_0[1]).copy(), (cor_n_0[2]).copy(), (cor_n_0[3]).copy()])
    
    # Integrate over all steps
    for i in range(len(sn)-1):
        temp = one_step(f, sn[i], cor_n_list[i], h=sn[i + 1] - sn[i])
        cor_n_list.append(temp)
    
    return cor_n_list

# Function to compute the magnetic vector potential\
# (a_x, a_y, a_z) based on harmonic coefficients
# The coefficients are derived from the fourth script\
# and reconstructed data in the third script.
def a_xyz(harm_coeff: np.ndarray):
    harm_shape = np.shape(harm_coeff)
    a_xyz = 0.0
    # Loop over the harmonic coefficients and sum them up
    for x_index in range(harm_shape[0]):
        for y_index in range(harm_shape[1]):
            a_xyz = a_xyz + harm_coeff[x_index, y_index] * PyTPSA.pow(x, x_index) * PyTPSA.pow(y, y_index)
    return a_xyz


# The function representing the system of equations for the Runge-Kutta method
# Here, we use a dipole magnet as an example, but this should be modified for your specific magnetic field
def f(sn: float, cor_n: list):
    # The example here uses a dipole magnet; modify the equations for a specific field.
    # The terms a_x, a_y, a_s are magnetic field components, which should be updated accordingly.
    # you can use a_xyz(harm_coeff: np.ndarray) function and others to redefine a_x/a_y/a_s.
    a_s = -0.50 * h * x * (1.0 + 1.0 / (1.0 + h * x))
    a_x = 0.0
    a_y = 0.0
    # Hamiltonian for the dipole magnetic field
    Hamiltonian = -(1 + h * x) * PyTPSA.sqrt(-PyTPSA.pow(px - a_x, 2) - PyTPSA.pow(py - a_y, 2) + 1.0) - (1 + h * x) * a_s
    # Compute the derivatives of the Hamiltonian
    H_x = Hamiltonian.derivative(2)
    H_px = -Hamiltonian.derivative(1)
    H_y = Hamiltonian.derivative(4)
    H_py = -Hamiltonian.derivative(3)
    
    # Evaluate the Hamiltonian derivatives using the current coordinates
    H_x_composite = H_x.composite(cor_n)
    H_px_composite = H_px.composite(cor_n)
    H_y_composite = H_y.composite(cor_n)
    H_py_composite = H_py.composite(cor_n)
    
    # Return the derivatives
    return [H_x_composite, H_px_composite, H_y_composite, H_py_composite]

# Initial conditions
h = -1.0 # the curvature
x = PyTPSA.tpsa(0.0, 1, dtype = float)
px = PyTPSA.tpsa(0.0, 2, dtype = float)
y = PyTPSA.tpsa(0.0, 3, dtype = float)
py = PyTPSA.tpsa(0.0, 4, dtype = float)

x0 = x.copy()
px0 = px.copy()
y0 = y.copy()
py0 = py.copy()

cor_n_0 = [x0, px0, y0, py0]
sn = np.linspace(0, 1.28, 801)
cor_n_list = ode(f, sn, cor_n_0)

# Store results for plotting
test_x = []
test_y = []
for i in range(len(sn)):
    # the initial coordinates is x_0, px_0, y_0, py_0 = 0.0015, 0.000, 0.0015, 0.000
    tempx = (cor_n_list[i][0]).evaluate([0.0015, 0.000, 0.0015, 0.000])
    tempy = (cor_n_list[i][2]).evaluate([0.0015, 0.000, 0.0015, 0.000])
    test_x.append(tempx)
    test_y.append(tempy)

# Output the final result of x
print(cor_n_list[-1][0])

# Plot the results for x and y over the integration steps
plt.plot(sn, test_x, label="X")
plt.plot(sn, test_y, label="Y")
plt.legend()
plt.show()
