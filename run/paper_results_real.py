"""
Generate all the figures used in the paper. Results section
"""

import datetime as dt
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.jax_utils import DNJ, Optimizer, RouteJax
from hybrid_routing.utils.plot import plot_ticks_radians_to_degrees
from hybrid_routing.vectorfields import VectorfieldReal
from hybrid_routing.vectorfields.base import Vectorfield

"""
Create output folder
"""

# Custom folder for this date
today = dt.date.today().strftime("%y-%m-%d")
path_out: Path = Path(f"output/{today}")
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
    vel: float = 10,
    num_angles: int = 20,
    rk_time_iter: float = 360,
    rk_time_step: float = 60,
    dist_min: float = 1000,
    dnj_time_step: float = 60,
    dnj_num_iter: int = 5,
    x_text: float = None,
    y_text: float = None,
    textbox_align: str = "bottom",
) -> Dict:
    # Initialize the output dictionary
    dict_out = {
        "vel": vel,
        "rk_time_iter": rk_time_iter,
        "rk_time_step": rk_time_step,
        "rk_dist_min": dist_min,
        "dnj_time_step": dnj_time_step,
        "dnj_num_iter": dnj_num_iter,
    }

    # Initialize the optimizer
    optimizer = Optimizer(
        vectorfield,
        time_iter=rk_time_iter,
        time_step=rk_time_step,
        angle_amplitude=np.pi,
        angle_heading=np.pi / 2,
        num_angles=num_angles,
        vel=vel,
        dist_min=dist_min,
        use_rk=True,
        method="direction",
    )

    # Assign None values
    ymin = xmin if ymin is None else ymin
    ymax = xmax if ymax is None else ymax
    x_text = xmin if x_text is None else x_text
    y_text = ymin if y_text is None else y_text

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
        step=1 * DEG2RAD,
        color="grey",
        alpha=0.8,
        do_color=True,
    )
    plt.gca().set_aspect("equal")
    plot_ticks_radians_to_degrees(step=5)

    # Plot source and destination point
    plt.scatter(x0, y0, c="green", s=20, zorder=10)
    plt.scatter(xn, yn, c="green", s=20, zorder=10)
    # Plot route
    plt.plot(route.x, route.y, c="red", linewidth=1, alpha=0.9, zorder=5)
    dict_out["route_rk"] = route.asdict()
    print("Optimization step done.")

    # Recompute times
    try:
        route.recompute_times(vel, vectorfield, interp=False)
        dict_out["route_rk2"] = route.asdict()
        print("Recomputation step done.")
    except AssertionError:
        print("Recomputation step failed.")
    time_opt_rec = float(route.t[-1])

    # Apply DNJ
    dnj = DNJ(vectorfield, time_step=dnj_time_step, optimize_for="fuel")
    # Apply DNJ in loop
    num_iter = dnj_num_iter // 5
    for n in range(5):
        dnj.optimize_route(route, num_iter=num_iter)
        s = 2 if n == 4 else 1
        c = "black" if n == 4 else "grey"
        alpha = 0.9 if n == 4 else 0.6
        plt.plot(route.x, route.y, c=c, linewidth=s, alpha=alpha, zorder=5)
    route.recompute_times(vel, vectorfield, interp=False)
    time_dnj = float(route.t[-1])
    dict_out["route_dnj"] = route.asdict()
    print("DNJ step done.")

    # Times
    # Textbox properties
    dict_bbox = dict(boxstyle="round", facecolor="white", alpha=0.95)
    text = (
        r"$\left\langle x_0, y_0 \right\rangle = \left\langle"
        + "{:.1f}".format(x0 / DEG2RAD)
        + ", "
        + "{:.1f}".format(y0 / DEG2RAD)
        + r"\right\rangle$"
        + "\n"
        r"$\left\langle x_T, y_T \right\rangle = \left\langle"
        + "{:.1f}".format(xn / DEG2RAD)
        + ", "
        + "{:.1f}".format(yn / DEG2RAD)
        + r"\right\rangle$"
        + "\nOptimized (red):\n"
        + f"  t = {time_opt_rec:.3f}\n"
        + "Smoothed (black):\n"
        + f"  t = {time_dnj:.3f}"
    )
    plt.text(
        x_text,
        y_text,
        text,
        fontsize=11,
        verticalalignment=textbox_align,
        bbox=dict_bbox,
    )

    # Set plot limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    return dict_out


"""
Vectorfield - Real land
"""

vf = VectorfieldReal.from_folder("./data", "real-land", radians=True)
for vel in [3, 6, 10]:
    try:
        d = pipeline(
            vectorfield=vf,
            x0=43.49 * DEG2RAD,
            y0=-1.66 * DEG2RAD,
            xn=98.14 * DEG2RAD,
            yn=10.21 * DEG2RAD,
            vel=vel,  # m/s
            xmin=vf.arr_x.min(),
            xmax=vf.arr_x.max(),
            ymin=vf.arr_y.min(),
            ymax=vf.arr_y.max(),
        )

        d["data"] = "real-land"

        dict_results[f"Real land {vel}"] = d

        # Store plot
        plt.tight_layout()
        plt.savefig(path_out / f"results-real-land-{vel}.png")
        plt.close()
    except Exception as er:
        print("[ERROR]", er)
    print(f"Done Real vectorfield with land, {vel} m/s\n---")

"""
Vectorfield - Real
"""

vf = VectorfieldReal.from_folder("./data", "real", radians=True)

for vel in [3, 6, 10]:
    try:
        d = pipeline(
            vectorfield=vf,
            x0=-79.7 * DEG2RAD,
            y0=32.7 * DEG2RAD,
            xn=-29.5 * DEG2RAD,
            yn=38.5 * DEG2RAD,
            vel=vel,  # m/s
            xmin=vf.arr_x.min(),
            xmax=vf.arr_x.max(),
            ymin=vf.arr_y.min(),
            ymax=vf.arr_y.max(),
        )

        d["data"] = "real"
        dict_results[f"Real {vel}"] = d

        # Store plot
        plt.tight_layout()
        plt.savefig(path_out / f"results-real-{vel}.png")
        plt.close()
    except Exception as er:
        print("[ERROR]", er)
    print(f"Done Real vectorfield, {vel} m/s\n---")

"""
Store dictionary
"""
with open(path_out / "results-real.json", "w") as outfile:
    json.dump(dict_results, outfile)
