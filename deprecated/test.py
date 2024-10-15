import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the system of ODEs
def system(v, t):
    x, y = v
    dxdt = -x + y
    dydt = -2*x - y
    return [dxdt, dydt]

# Initial conditions: x=10, y=10
v0 = [10, 10]

# Time points
t = np.linspace(0, 10, 500)

# Solve the system of ODEs
v = odeint(system, v0, t)

# Create a grid of points
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)

X, Y = np.meshgrid(x, y)

# Compute the derivative at each point on the grid
u, v = np.zeros(X.shape), np.zeros(Y.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        yprime = system([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

# Normalize the arrows so their size represents their speed
N = np.sqrt(u**2+v**2)  
U = u/N
V = v/N

# Create the quiver plot
plt.quiver(X, Y, U, V, N)
plt.show()