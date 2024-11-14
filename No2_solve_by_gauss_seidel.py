import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr

# This script solves a matrix equation using the Gauss-Seidel iterative method.
# The matrix (B_matrix) relates the magnetic moment intensity to the normal magnetic field (norm_B)
# at points on the surface of the region of interest.

# Load the necessary data:
B_matrix = np.load("B_matrix.npy")  # Matrix defining the relationship between magnetic moment and normal field.
norm_B = np.load("norm_B.npy")  # Normal magnetic field values on the surface.

res = np.zeros(len(norm_B))  # Initial guess for the solution vector.

print("Load Successful")

# Set parameters for the Gauss-Seidel iteration:
max_iter = 1000  # Maximum number of iterations allowed.
tol = 1e-10  # Tolerance level for convergence.

# Define the Gauss-Seidel function:
def gauss_seidel(A, b, x0, max_iter, tol):
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.
    
    Parameters:
    - A: Matrix (numpy array) representing the coefficients in the system.
    - b: Right-hand side vector in the system.
    - x0: Initial guess for the solution.
    - max_iter: Maximum number of iterations.
    - tol: Convergence tolerance.
    
    Returns:
    - x: Approximate solution to the system.
    """
    x = x0.copy()  # Initialize the solution vector.
    for k in range(max_iter):
        x_old = x.copy()  # Store the solution from the previous iteration.
        for i in range(A.shape[0]):
            sum1 = A[i, :i].dot(x[:i])  # Sum of products for terms before diagonal.
            sum2 = A[i, i+1:].dot(x_old[i+1:])  # Sum of products for terms after diagonal.
            x[i] = (b[i] - sum1 - sum2) / A[i, i]  # Update the i-th solution component.
        
        # Compute the relative difference between current and previous solutions.
        diff = np.linalg.norm(x - x_old) / np.linalg.norm(x)
        
        # Check residuals (difference between b and Ax) to monitor convergence.
        diff_b = np.abs(b - np.dot(A, x))
        print(k, diff, max(diff_b))
        
        # Check for convergence:
        if diff < tol:
            print(f"Converged in {k} iterations")
            return x
        
    print("Maximum iterations reached")
    return x

# Run the Gauss-Seidel method:
x = gauss_seidel(B_matrix, norm_B, res, max_iter, tol)

# Save the computed solution:
np.save("res_gauss_seidel.npy", x)  # Save the solution vector.
print("Computation complete.")

