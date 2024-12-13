import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the neural network model with customizable layers and activation functions
class HeatEquationNN(tf.keras.Model):
    def __init__(self, hidden_layers=2, hidden_nodes=64, activation="tanh"):
        super().__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_nodes, activation=activation)
            for _ in range(hidden_layers)
        ]  # Define multiple hidden layers with specified activation
        self.output_layer = tf.keras.layers.Dense(1, activation=None)  # Output layer (no activation)

    def call(self, inputs):
        x, t = inputs[:, 0:1], inputs[:, 1:2]  # Split inputs into spatial (x) and temporal (t)
        h = tf.concat([x, t], axis=1)  # Combine x and t as input to the network
        for layer in self.hidden_layers:
            h = layer(h)  # Pass through each hidden layer
        return self.output_layer(h)  # Final output layer to compute u(x, t)

# Generate training data: grid points in space (x) and time (t)
def generate_training_data(nx, nt):
    x = np.linspace(0, 1, nx)  # Discretize the spatial domain [0, 1]
    t = np.linspace(0, 0.5, nt)  # Discretize the time domain [0, 0.5]
    X, T = np.meshgrid(x, t)  # Create a grid of (x, t) values
    X_flat = X.flatten()[:, None]  # Flatten the spatial grid
    T_flat = T.flatten()[:, None]  # Flatten the temporal grid
    return np.hstack((X_flat, T_flat))  # Combine x and t into a single input array

# Analytical solution for comparison
def analytical_solution(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

# Loss function to enforce the PDE, boundary, and initial conditions
def loss_fn(model, inputs):
    x, t = inputs[:, 0:1], inputs[:, 1:2]  # Extract x and t from inputs

    # Use automatic differentiation to compute derivatives
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, t])  # Watch x and t for differentiation
        u = model(inputs)  # Compute the network output u(x, t)
        du_dx = tape1.gradient(u, x)  # Compute \partial u/\partial x
        du_dt = tape1.gradient(u, t)  # Compute \partial u/\partial t
    d2u_dx2 = tape1.gradient(du_dx, x)  # Compute \partial^2 u/\partial x^2 (second derivative)
    del tape1  # Release resources used by the tape

    # Compute the PDE residual: \partial u/\partial t - \partial^2 u/\partial x^2
    pde_residual = du_dt - d2u_dx2

    # Enforce boundary conditions: u(0, t) = u(1, t) = 0
    boundary_u = tf.reduce_mean(tf.square(model(tf.concat([tf.zeros_like(t), t], axis=1)))) + \
                 tf.reduce_mean(tf.square(model(tf.concat([tf.ones_like(t), t], axis=1))))

    # Enforce the initial condition: u(x, 0) = sin(\pi x)
    initial_u = tf.reduce_mean(tf.square(model(tf.concat([x, tf.zeros_like(t)], axis=1)) - tf.sin(np.pi * x)))

    # Total loss = PDE residual + boundary conditions + initial condition
    return tf.reduce_mean(tf.square(pde_residual)) + boundary_u + initial_u

# Train the neural network model
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

# Main program to test different configurations
nx, nt = 50, 50  # Number of spatial (nx) and temporal (nt) points
inputs = generate_training_data(nx, nt).astype(np.float32)  # Generate training data

# Test different configurations
hidden_layer_configs = [2, 4]  # Number of hidden layers to test
hidden_node_configs = [32, 64, 128]  # Number of hidden nodes per layer to test
activation_functions = ["tanh", "relu"]  # Activation functions to test

# Evaluate each configuration
for hidden_layers in hidden_layer_configs:
    for hidden_nodes in hidden_node_configs:
        for activation in activation_functions:
            print(f"\nTesting configuration: {hidden_layers} layers, {hidden_nodes} nodes, {activation} activation")
            model = HeatEquationNN(hidden_layers=hidden_layers, hidden_nodes=hidden_nodes, activation=activation)
            trained_model = train_model(model, inputs, epochs=3000, learning_rate=1e-3)

            # Evaluate the solution at two specific time points: t1 and t2
            x = np.linspace(0, 1, 100)[:, None]  # Spatial points for evaluation
            t1, t2 = 0.05, 0.5  # Time points
            t1_inputs = np.hstack((x, t1 * np.ones_like(x))).astype(np.float32)  # Inputs for t1
            t2_inputs = np.hstack((x, t2 * np.ones_like(x))).astype(np.float32)  # Inputs for t2

            u_t1_nn = trained_model(t1_inputs).numpy().flatten()  # Neural network solution at t1
            u_t2_nn = trained_model(t2_inputs).numpy().flatten()  # Neural network solution at t2
            u_t1_analytical = analytical_solution(x, t1)  # Analytical solution at t1
            u_t2_analytical = analytical_solution(x, t2)  # Analytical solution at t2

            # Plot the results
            plt.figure(figsize=(12, 6))
            plt.plot(x, u_t1_nn, label=f"NN Solution at t1 ({activation}, {hidden_layers} layers, {hidden_nodes} nodes)", linestyle="--")
            plt.plot(x, u_t1_analytical, label="Analytical Solution at t1", linestyle="-")
            plt.plot(x, u_t2_nn, label=f"NN Solution at t2 ({activation}, {hidden_layers} layers, {hidden_nodes} nodes)", linestyle="--")
            plt.plot(x, u_t2_analytical, label="Analytical Solution at t2", linestyle="-")
            plt.xlabel("x")
            plt.ylabel("u(x, t)")
            plt.title("Neural Network Solution vs Analytical Solution")
            plt.legend()
            plt.grid(True)
            plt.show()
