"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.utils.plot import plot_textbox
from hybrid_routing.vectorfields import Circular, FourVortices
from hybrid_routing.vectorfields.base import Vectorfield

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()
# Initialize dict of results
dict_results = {}


def pipeline(
    vectorfield: Vectorfield,
    x0: float,
    y0: float,
    xn: float,
    yn: float,
    xmin: float = 0,
    xmax: float = 1,
    ymin: float = None,
    ymax: float = None,
    vel: float = 1,
    rk_time_iter: float = 0.5,
    rk_time_step: float = 0.025,
    dist_min: float = 0.1,
    dnj_time_step: float = 0.01,
    textbox_pos: Optional[Tuple[float]] = None,
    textbox_align: str = "top",
) -> Dict:
    # Initialize the optimizer
    optimizer = Optimizer(
        vectorfield,
        time_iter=rk_time_iter,
        time_step=rk_time_step,
        angle_amplitude=np.pi,
        angle_heading=np.pi / 2,
        num_angles=20,
        vel=vel,
        dist_min=dist_min,
        use_rk=True,
        method="direction",
    )

    # Assign None values
    ymin = xmin if ymin is None else ymin
    ymax = xmax if ymax is None else ymax

    # Run the optimizer until it converges
    for list_routes in optimizer.optimize_route(x0, y0, xn, yn):
        pass

    # Take the best route
    route: RouteJax = list_routes[0]
    route.append_point_end(x=xn, y=yn, vel=optimizer.vel)

    # Initialize figure with vectorfield
    plt.figure(figsize=(5, 5))
    vectorfield.plot(
        x_min=xmin,
        x_max=xmax,
        y_min=ymin,
        y_max=ymax,
        step=0.25,
        color="grey",
        alpha=0.8,
    )
    plt.gca().set_aspect("equal")
    xticks = np.arange(xmin, xmax, 1)
    plt.xticks(xticks)
    yticks = np.arange(ymin, ymax, 1)
    plt.yticks(yticks)

    # Plot source and destination point
    plt.scatter(x0, y0, c="green", s=20, zorder=10)
    plt.scatter(xn, yn, c="green", s=20, zorder=10)
    # Plot route
    plt.plot(route.x, route.y, c="red", linewidth=1, alpha=0.9, zorder=5)

    # Initialize dictionary
    dict_out = {"route": route.asdict()}

    # Recompute times
    route.recompute_times(vel, vectorfield)
    time_opt_rec = float(route.t[-1])
    dict_out["route_recompute"] = route.asdict()

    print("Optimization step done.")

    # Apply DNJ
    dnj = DNJ(vectorfield, time_step=dnj_time_step, optimize_for="fuel")
    # Apply DNJ in loop
    for n in range(5):
        dnj.optimize_route(route, num_iter=200)
        s = 2 if n == 4 else 1
        c = "black" if n == 4 else "grey"
        alpha = 0.9 if n == 4 else 0.6
        plt.plot(route.x, route.y, c=c, linewidth=s, alpha=alpha, zorder=5)
    route.recompute_times(vel, vectorfield)
    time_dnj = float(route.t[-1])
    dict_out["route_dnj"] = route.asdict()

    print("DNJ step done.")

    plot_textbox(
        (x0, y0),
        (xn, yn),
        (time_opt_rec, time_dnj),
        pos=textbox_pos,
        align=textbox_align,
    )

    return dict_out


"""
Vectorfield - Circular
"""

dict_results["Circular"] = pipeline(
    vectorfield=Circular(),
    x0=3,
    y0=2,
    xn=-7,
    yn=2,
    xmin=-8,
    xmax=8,
    textbox_pos=(0.0 - 3.5),
    textbox_align="bottom",
)

# Store plot
plt.xlim(-8, 4)
plt.ylim(-4, 6)
plt.tight_layout()
plt.savefig(path_out / "results-circular.png")
plt.close()

print("Done Circular vectorfield")

"""
Vectorfield - Four Vortices
"""

# We will regenerate the results from Ferraro et al.
dict_results["FourVortices"] = pipeline(
    vectorfield=FourVortices(),
    x0=0,
    y0=0,
    xn=6,
    yn=2,
    xmin=-2,
    xmax=8,
    textbox_pos=(0, 5.5),
)

# Store plot
plt.xlim(-0.5, 6.5)
plt.ylim(-1.5, 6)
plt.tight_layout()
plt.savefig(path_out / "results-fourvortices.png")
plt.close()

print("Done Four Vortices vectorfield")

"""
Store dictionary
"""
with open(path_out / "results.json", "w") as outfile:
    json.dump(dict_results, outfile)
    json.dump(dict_results, outfile)
