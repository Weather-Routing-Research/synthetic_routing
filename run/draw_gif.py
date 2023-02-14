import os
import shutil
from importlib import import_module
from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import typer

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.jax_utils import DNJ, Optimizer, Route
from hybrid_routing.vectorfields import VectorfieldReal
from hybrid_routing.vectorfields.base import Vectorfield


def main(
    vf: str = "FourVortices",
    discrete: bool = False,
    use_rk: bool = True,
    method: str = "direction",
    time_iter: float = 0.2,
    time_step: float = 0.01,
    angle_amplitude: float = np.pi,
    num_angles: int = 20,
    vel: float = 1,
    dist_min: float = 0.1,
    time_dnj: Optional[float] = None,
    do_color: bool = False,
    path_out: str = "output/",
):
    suptitle = vf
    suptitle += " Discretized" if discrete else ""
    suptitle += " Runge-Kutta" if use_rk else " ODEINT"
    suptitle += " " + method

    color = "orange" if do_color else "green"

    # Build folder with images
    path_img = Path("img_optim")
    if not path_img.exists():
        os.mkdir(path_img)
    images = []

    ####################################################################################
    #  VECTORFIELD
    ####################################################################################

    # Import vectorfield
    if vf.lower() == "real":
        vectorfield = VectorfieldReal.from_folder("./data", "real-land", radians=True)
        q0 = (43.49 * DEG2RAD, -1.66 * DEG2RAD)
        q1 = (98.14 * DEG2RAD, 10.21 * DEG2RAD)
        xlim = (40 * DEG2RAD, 100 * DEG2RAD)
        ylim = (-25 * DEG2RAD, 30 * DEG2RAD)
        step = DEG2RAD
    else:
        module = import_module("hybrid_routing.vectorfields")
        vectorfield: Vectorfield = getattr(module, vf)()
        if discrete:
            vectorfield = vectorfield.discretize(-1, 7, -1, 7, step=1 / 12)
        q0 = (0, 0)
        q1 = (6, 2)
        xlim = (-1, 7)
        ylim = (-1, 7)
        step = 0.2

    ####################################################################################
    #  Helper function to plot the routes
    ####################################################################################

    def plot_routes_and_save(
        list_routes: List[Route],
        fout: str,
        title: str = "",
        color: str = "black",
    ):
        vectorfield.plot(
            x_min=xlim[0],
            x_max=xlim[1],
            y_min=ylim[0],
            y_max=ylim[1],
            step=step,
            do_color=do_color,
        )
        for route in list_routes:
            plt.plot(route.x, route.y, color=color, alpha=0.6)
        plt.scatter([q0[0], q1[0]], [q0[1], q1[1]], c="red")
        plt.suptitle(suptitle)
        plt.title(title)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        # Set ticks on bottom
        plt.gca().tick_params(
            axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
        )
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))

    ####################################################################################
    # OPTIMIZATION ZIVP
    ####################################################################################

    optimizer = Optimizer(
        vectorfield,
        time_iter=time_iter,
        time_step=time_step,
        angle_amplitude=angle_amplitude,
        angle_heading=angle_amplitude / 2,
        num_angles=num_angles,
        vel=vel,
        dist_min=dist_min,
        use_rk=use_rk,
        method=method,
    )

    idx = 0
    # Generate one plot every 10 iterations
    for list_routes in optimizer.optimize_route(q0, q1):
        if idx % 5 == 0:
            n = "exploration" if optimizer.exploration else "exploitation"
            plot_routes_and_save(
                list_routes=list_routes,
                fout=path_img / f"{idx:03d}.png",
                title=f"ZIVP ({n})",
                color=color,
            )
        idx += 1

    # Plot best for some frames
    for _ in range(6):
        plot_routes_and_save(
            list_routes=[list_routes[0]],
            fout=path_img / f"{idx:03d}.png",
            title="ZIVP (result)",
            color=color,
        )
        idx += 1

    ####################################################################################
    # SMOOTHING DNJ
    ####################################################################################

    time_dnj = time_iter if time_dnj is None else time_dnj
    dnj = DNJ(vectorfield, time_step=time_dnj, optimize_for="fuel")

    for n in range(40):
        route = list_routes[0]
        dnj.optimize_route(route, num_iter=200)
        plot_routes_and_save(
            list_routes=[route],
            fout=path_img / f"{idx:03d}.png",
            title=f"DNJ ({int(n * 200)} steps)",
            color=color,
        )
        idx += 1

    # Plot best for some frames
    for _ in range(6):
        plot_routes_and_save(
            list_routes=[list_routes[0]],
            fout=path_img / f"{idx:03d}.png",
            title="DNJ (result)",
            color=color,
        )
        idx += 1

    ####################################################################################
    # GIF
    ####################################################################################

    # Build output folder
    path_out: Path = Path(path_out)
    if not path_out.exists():
        path_out.mkdir()
    # Convert images to gif and delete images
    imageio.mimsave(path_out / f"{vf}_{vel:2.0f}.gif", images)
    shutil.rmtree(path_img)


if __name__ == "__main__":
    typer.run(main)
