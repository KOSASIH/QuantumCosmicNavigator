import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression

# Initial conditions
initial_position = [1, 0, 0]
initial_velocity = [0, 1, 0]

# Orbital parameters
semi_major_axis = 1.0
eccentricity = 0.5
inclination = 0.0
longitude_of_ascending_node = 0.0
argument_of_periapsis = 0.0
mean_anomaly = 0.0

# Define the differential equations for motion
def motion_equations(state, t):
    x, y, z, vx, vy, vz = state

    r = np.sqrt(x**2 + y**2 + z**2)

    ax = -G * (M / r**3) * x
    ay = -G * (M / r**3) * y
    az = -G * (M / r**3) * z

    return [vx, vy, vz, ax, ay, az]

# Solve the differential equations
t = np.linspace(0, 10, 100)
initial_state = initial_position + initial_velocity
states = odeint(motion_equations, initial_state, t)

# Predict future positions using linear regression
regression_model = LinearRegression()
regression_model.fit(t.reshape(-1, 1), states[:, :3])

future_t = np.linspace(10, 20, 100)
future_positions = regression_model.predict(future_t.reshape(-1, 1))

# Output the predicted positions
print("Time\tX\tY\tZ")
for t, position in zip(future_t, future_positions):
    print(f"{t:.2f}\t{position[0]:.2f}\t{position[1]:.2f}\t{position[2]:.2f}")
