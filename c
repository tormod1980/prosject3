import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the neural network model for solving the heat equation
class HeatEquationNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the architecture with two hidden layers, each having 64 neurons
        self.hidden1 = tf.keras.layers.Dense(64, activation="tanh")  # First hidden layer
        self.hidden2 = tf.keras.layers.Dense(64, activation="tanh")  # Second hidden layer
        self.output_layer = tf.keras.layers.Dense(1, activation=None)  # Output layer (predict u)

    def call(self, inputs):
        # Split inputs into spatial (x) and temporal (t) components
        x, t = inputs[:, 0:1], inputs[:, 1:2]
        h = tf.concat([x, t], axis=1)  # Combine x and t for input to the network
        h = self.hidden1(h)  # Pass through the first hidden layer
        h = self.hidden2(h)  # Pass through the second hidden layer
        return self.output_layer(h)  # Compute the final output (u(x, t))

# Generate training data: grid points in space (x) and time (t)
def generate_training_data(nx, nt):
    x = np.linspace(0, 1, nx)  # Discretize the spatial domain [0, 1]
    t = np.linspace(0, 0.5, nt)  # Discretize the time domain [0, 0.5]
    X, T = np.meshgrid(x, t)  # Create a grid of (x, t) values
    X_flat = X.flatten()[:, None]  # Flatten the spatial grid
    T_flat = T.flatten()[:, None]  # Flatten the temporal grid
    return np.hstack((X_flat, T_flat))  # Combine x and t into a single input array

# Define the analytical solution for comparison
def analytical_solution(x, t):
    # Exact solution: u(x, t) = exp(-π^2 * t) * sin(π * x)
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

# Define the loss function for training the neural network
def loss_fn(model, inputs):
    x, t = inputs[:, 0:1], inputs[:, 1:2]  # Extract x and t from inputs

    # Use automatic differentiation to compute derivatives
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, t])  # Watch x and t for differentiation
        u = model(inputs)  # Compute the network output u(x, t)
        du_dx = tape1.gradient(u, x)  # Compute ∂u/∂x
        du_dt = tape1.gradient(u, t)  # Compute ∂u/∂t
    d2u_dx2 = tape1.gradient(du_dx, x)  # Compute ∂²u/∂x² (second derivative)
    del tape1  # Release resources used by the tape

    # Compute the PDE residual: ∂u/∂t - ∂²u/∂x²
    pde_residual = du_dt - d2u_dx2

    # Enforce boundary conditions: u(0, t) = u(1, t) = 0
    boundary_u = tf.reduce_mean(tf.square(model(tf.concat([tf.zeros_like(t), t], axis=1)))) + \
                 tf.reduce_mean(tf.square(model(tf.concat([tf.ones_like(t), t], axis=1))))

    # Enforce the initial condition: u(x, 0) = sin(πx)
    initial_u = tf.reduce_mean(tf.square(model(tf.concat([x, tf.zeros_like(t)], axis=1)) - tf.sin(np.pi * x)))

    # Total loss = PDE residual + boundary conditions + initial condition
    return tf.reduce_mean(tf.square(pde_residual)) + boundary_u + initial_u

# Train the neural network to minimize the loss function
def train_model(model, inputs, epochs=5000, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate)  # Optimizer for training
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, inputs)  # Compute the loss
        gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update model parameters
        if epoch % 500 == 0:  # Print progress every 500 epochs
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    return model

# Main program
nx, nt = 50, 50  # Number of spatial (nx) and temporal (nt) points
inputs = generate_training_data(nx, nt).astype(np.float32)  # Generate training data

# Instantiate and train the neural network model
model = HeatEquationNN()
trained_model = train_model(model, inputs)

# Evaluate the neural network solution at two specific time points: t1 and t2
x = np.linspace(0, 1, 100)[:, None]  # Spatial points for evaluation
t1, t2 = 0.05, 0.5  # Time points
t1_inputs = np.hstack((x, t1 * np.ones_like(x))).astype(np.float32)  # Inputs for t1
t2_inputs = np.hstack((x, t2 * np.ones_like(x))).astype(np.float32)  # Inputs for t2

u_t1_nn = trained_model(t1_inputs).numpy().flatten()  # Neural network solution at t1
u_t2_nn = trained_model(t2_inputs).numpy().flatten()  # Neural network solution at t2
u_t1_analytical = analytical_solution(x, t1)  # Analytical solution at t1
u_t2_analytical = analytical_solution(x, t2)  # Analytical solution at t2

# Plot the neural network solution vs the analytical solution
plt.figure(figsize=(12, 6))
plt.plot(x, u_t1_nn, label="NN Solution at t1", linestyle="--", marker="o")
plt.plot(x, u_t1_analytical, label="Analytical Solution at t1", linestyle="-")
plt.plot(x, u_t2_nn, label="NN Solution at t2", linestyle="--", marker="x")
plt.plot(x, u_t2_analytical, label="Analytical Solution at t2", linestyle="-")
plt.xlabel("x")  # Label for x-axis
plt.ylabel("u(x, t)")  # Label for y-axis
plt.title("Neural Network Solution vs Analytical Solution")  # Plot title
plt.legend()  # Add a legend
plt.grid(True)  # Add a grid
plt.show()
