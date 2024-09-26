"""
Generate all the figures used in the paper. Methods section
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from synthrouting.optimization.dnj import DNJ
from synthrouting.optimization.optimize import Optimizer, compute_thetas_in_cone
from synthrouting.optimization.route import Route
from synthrouting.utils.config import load_config
from synthrouting.vectorfields import FourVortices

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()

"""
Vectorfield and initial conditions
"""

vectorfield = FourVortices()

cfg = load_config("./data/config.toml")
cfg_vf: Dict = cfg["synthetic"]["FourVortices"]
cfg_zivp: Dict = cfg["zivp"]["synthetic"]
cfg_zivp.pop("num_points")
cfg_zivp["num_angles"] = cfg_zivp["num_angles"] // 3
cfg_zivp["angle_amplitude"] = cfg_zivp["angle_amplitude"] / 3
cfg_dnj: Dict = cfg["dnj"]["synthetic"]

x0, y0 = cfg_vf["p0"]
xn, yn = cfg_vf["pn"]
xmin, xmax = -2, 8
ymin, ymax = -4, 6

optimizer = Optimizer(vectorfield, **cfg_zivp)


# Initialize figure with vectorfield
# We encapsulate this code into a function because we are reusing it later
def plot_vectorfield():
    plt.figure(figsize=(5, 5))
    optimizer.vectorfield.plot(
        extent=(xmin, xmax, ymin, ymax), color="grey", alpha=0.8, step=0.25
    )
    plt.gca().set_aspect("equal")
    ticks = np.arange(xmin, xmax, 1)
    plt.xticks(ticks)
    plt.yticks(ticks)


"""
Run Runge-Kutta method and plot its result
"""

plot_vectorfield()

# Plot source point
plt.scatter(x0, y0, c="green", s=20, zorder=10)

# Initial conditions of each segment (only angle varies)
x = np.repeat(x0, 5)
y = np.repeat(y0, 5)
# Compute angles
cone_center = optimizer.geometry.angle_p0_to_p1((x0, y0), (xn, yn))
theta = compute_thetas_in_cone(cone_center, np.pi, 5)

# Run RK method and plot each segment
list_segments = optimizer.solve_ivp(x, y, theta, time_iter=4)
for segment in list_segments:
    x, y = segment.x, segment.y
    plt.plot(x, y, c="black", alpha=0.9, zorder=5)
    plt.scatter(x[1:-1], y[1:-1], c="orange", s=10, zorder=10)
    plt.scatter(x[-1], y[-1], c="red", s=20, zorder=10)

# Store plot
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.tight_layout()
plt.savefig(path_out / "methods_runge_kutta.png")
plt.close()

print("Runge-Kutta - Finished")

"""
Exploration step
"""

run = optimizer.optimize_route((x0, y0), (xn, yn))
list_routes_plot = next(run)

for list_routes in run:
    if optimizer.exploration:
        list_routes_plot = deepcopy(list_routes)
    else:
        break

plot_vectorfield()


# Plot each route segment
# We encapsulate this code into a function because we are reusing it later
def plot_routes(list_routes: List[Route], theta: Optional[float] = None):
    # Plot source point
    plt.scatter(x0, y0, c="green", s=20, zorder=10)
    plt.scatter(xn, yn, c="green", s=20, zorder=10)
    # Plot routes
    for idx, route in enumerate(list_routes):
        x, y = route.x, route.y
        # Highlight the best route of the bunch
        c = "black"
        if (theta is None) and (idx == 0):
            c = "red"
        elif idx == 0:
            c = "blue"
        elif route.theta[0] == theta:
            c = "red"

        plt.plot(x, y, c=c, linewidth=1.5, alpha=0.9, zorder=5)


plot_routes(list_routes_plot)

# Compute angles
cone_center = optimizer.geometry.angle_p0_to_p1((x0, y0), (xn, yn))
arr_theta = compute_thetas_in_cone(
    cone_center, optimizer.angle_amplitude, optimizer.num_angles
)

# Plot original angles
for theta in arr_theta:
    x = x0 + np.cos(theta) * np.array([0, 3])
    y = y0 + np.sin(theta) * np.array([0, 3])
    plt.plot(x, y, linestyle="--", color="orange", alpha=1, zorder=3)

# Store plot
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.tight_layout()
plt.savefig(path_out / "methods_hybrid_exploration.png")
plt.close()

theta0 = list_routes_plot[0].theta[0]
print("Exploration step - Finished")

"""
Refinement step
"""

for list_routes in run:
    if not optimizer.exploration:
        list_routes_plot = deepcopy(list_routes)
    else:
        break

plot_vectorfield()
plot_routes(list_routes_plot, theta=theta0)

# Compute angles
angle_best = list_routes_plot[0].theta[0]
arr_theta = compute_thetas_in_cone(
    angle_best, optimizer.angle_amplitude / 5, optimizer.num_angles
)

# Plot original angles
for theta in arr_theta:
    x = x0 + np.cos(theta) * np.array([0, 3])
    y = y0 + np.sin(theta) * np.array([0, 3])
    plt.plot(x, y, linestyle="--", color="orange", alpha=1, zorder=3)

# Store plot
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.tight_layout()
plt.savefig(path_out / "methods_hybrid_refinement.png")
plt.close()

print("Refinement step - Finished")

"""
Finish optimization
"""

for list_routes in run:
    list_routes_plot = deepcopy(list_routes)
# Append goal
route: Route = list_routes_plot[0]
route.append_point_end(p=(xn, yn), vel=optimizer.vel)

plot_vectorfield()
plot_routes([route])

# Store plot
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.tight_layout()
plt.savefig(path_out / "methods_hybrid_optimized.png")
plt.close()

print("Optimization - Finished")

"""
Discrete Newton-Jacobi
"""

dnj = DNJ(vectorfield, **cfg_dnj)

plot_vectorfield()
plt.scatter(route.x[0], route.y[0], c="green", s=20, zorder=10)
plt.scatter(route.x[-1], route.y[-1], c="green", s=20, zorder=10)
plt.plot(route.x, route.y, c="grey", linewidth=2, alpha=0.9, zorder=5)

# Apply DNJ in loop
for n in range(5):
    dnj.optimize_route(route, num_iter=dnj.num_iter // 5)
    s = 2 if n == 4 else 1
    c = "black" if n == 4 else "grey"
    alpha = 0.9 if n == 4 else 0.6
    # Plot both in normal and zoom
    plt.plot(route.x, route.y, c=c, linewidth=s, alpha=alpha, zorder=5)

# Limit normal axis
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# Store plot
plt.tight_layout()
plt.savefig(path_out / "methods_hybrid_dnj.png")
plt.close()

print("DNJ - Finished")
