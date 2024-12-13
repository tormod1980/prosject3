import numpy as np
import matplotlib.pyplot as plt

def explicit_scheme(dx, t1, t2):
    """
    Solves the heat equation using the explicit Euler scheme for a given dx, t1, and t2.
    dx: Spatial step size.
    t1: Time point for intermediate solution.
    t2: Time point for near-stationary solution.
    """
    L = 1  # Length of the rod
    nx = int(L / dx) + 1  # Number of spatial points
    x = np.linspace(0, L, nx)  # Spatial domain
    dt = 0.5 * dx**2  # Time step size based on stability condition
    alpha = dt / dx**2  # Alpha (stability parameter)
    
    # Time points for t1 and t2
    t_final = max(t1, t2)  # Total simulation time
    nt = int(t_final / dt)  # Number of time steps

    # Initial condition: u(x, 0) = sin(pi * x)
    u = np.sin(np.pi * x)
    u_new = np.zeros(nx)

    # Boundary conditions: u(0, t) = u(L, t) = 0
    u[0] = u[-1] = 0

    # Store solutions at t1 and t2
    solution_t1 = None
    solution_t2 = None

    # Time evolution using the explicit Euler scheme
    for n in range(nt + 1):
        u_new[0] = u_new[-1] = 0  # Enforcing boundary conditions
        for i in range(1, nx - 1):
            u_new[i] = u[i] + alpha * (u[i + 1] - 2 * u[i] + u[i - 1])
        u[:] = u_new  # Update the temperature field
        
        # Save solutions at t1 and t2
        t_current = n * dt
        if np.isclose(t_current, t1, atol=dt):
            solution_t1 = u.copy()
        if np.isclose(t_current, t2, atol=dt):
            solution_t2 = u.copy()

    return x, solution_t1, solution_t2


# Testing with ∆x = 1/10 and ∆x = 1/100
dx_values = [1/10, 1/100]
t1 = 0.05  # Time where solution is smooth but significantly curved
t2 = 0.5   # Time where solution is close to stationary state

plt.figure(figsize=(12, 8))

for dx in dx_values:
    x, u_t1, u_t2 = explicit_scheme(dx, t1, t2)

    # Plot results for t1
    plt.plot(x, u_t1, label=f"∆x={dx}, t={t1} (Smooth and curved)")
    # Plot results for t2
    plt.plot(x, u_t2, label=f"∆x={dx}, t={t2} (Near stationary)")

# Formatting the plot
plt.xlabel("Position (x)")
plt.ylabel("Temperature (u)")
plt.title("Temperature Profile for Different ∆x at t1 and t2")
plt.legend()
plt.grid(True)
plt.show()
