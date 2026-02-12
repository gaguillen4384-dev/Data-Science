# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. & 2. Define the function and its analytical gradient
def f(x):
    x1, x2 = x[0], x[1]
    g1 = np.exp(-(x1 - 1)**2 - (x2 - 1)**2)
    g2 = np.exp(-(x1 + 2)**2 - (x2 + 3)**2)
    return -2 * g1 - g2

def grad_f(x):
    x1, x2 = x[0], x[1]
    g1 = np.exp(-(x1 - 1)**2 - (x2 - 1)**2)
    g2 = np.exp(-(x1 + 2)**2 - (x2 + 3)**2)
    df_dx1 = 4 * (x1 - 1) * g1 + 2 * (x1 + 2) * g2
    df_dx2 = 4 * (x2 - 1) * g1 + 2 * (x2 + 3) * g2
    return np.array([df_dx1, df_dx2])

def gradient_descent(start_x, alpha, iterations):
    path = [start_x]
    values = [f(start_x)]
    curr_x = np.array(start_x, dtype=float)
    
    for _ in range(iterations):
        grad = grad_f(curr_x)
        curr_x = curr_x - alpha * grad
        path.append(curr_x.copy())
        values.append(f(curr_x))
    return np.array(path), np.array(values)

# 3. Parameters
alpha = 0.15  # Tuned step size
iters = 125

path1, vals1 = gradient_descent((0, 0), alpha, iters)
path2, vals2 = gradient_descent((-1, -1), alpha, iters)

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Trajectories in State Space
x_range = np.linspace(-4, 3, 100)
y_range = np.linspace(-5, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = -2 * np.exp(-(X-1)**2 - (Y-1)**2) - np.exp(-(X+2)**2 - (Y+3)**2)

ax1.contour(X, Y, Z, 30, cmap='viridis')
ax1.plot(path1[:,0], path1[:,1], 'r-o', label='Start (0,0)', markersize=4)
ax1.plot(path2[:,0], path2[:,1], 'b-o', label='Start (-1,-1)', markersize=4)
ax1.set_title("Gradient Descent Trajectories")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.legend()

# Plot 2: Function Value vs. Steps
ax2.plot(vals1, 'r', label='Path from (0,0)')
ax2.plot(vals2, 'b', label='Path from (-1,-1)')
ax2.set_title("Function Value Convergence")
ax2.set_xlabel("Step Number")
ax2.set_ylabel("f(x)")
ax2.legend()

plt.tight_layout()
plt.show()