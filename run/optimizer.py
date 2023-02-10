import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import typer

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.vectorfields import VectorfieldReal
from hybrid_routing.vectorfields.base import Vectorfield


def main(
    vf: str = "FourVortices",
    discretized: bool = True,
    use_rk: bool = True,
    method: str = "direction",
    time_iter: float = 0.2,
    time_step: float = 0.01,
    angle_amplitude: float = np.pi,
    num_angles: int = 20,
    vel: float = 1,
    dist_min: float = 0.1,
    do_color: bool = False,
    path_out: str = "output/",
):
    # Import vectorfield
    if vf == "Real":
        vectorfield = VectorfieldReal.from_folder("./data", "real-land", radians=True)
        q0 = (43.49 * DEG2RAD, -1.66 * DEG2RAD)
        q1 = (98.14 * DEG2RAD, 10.21 * DEG2RAD)
        xlim = (40 * DEG2RAD, 100 * DEG2RAD)
        ylim = (-25 * DEG2RAD, 30 * DEG2RAD)
        step = DEG2RAD
    else:
        module = __import__("hybrid_routing")
        module = getattr(module, "vectorfields")
        vectorfield: Vectorfield = getattr(module, vf)()
        if discretized:
            vectorfield = vectorfield.discretize(-1, 7, -1, 7)
        q0 = (0, 0)
        q1 = (6, 2)
        xlim = (-1, 7)
        ylim = (-1, 7)
        step = 1

    optimizer = Optimizer(
        vectorfield,
        time_iter=time_iter,
        time_step=time_step,
        angle_amplitude=angle_amplitude,
        num_angles=num_angles,
        vel=vel,
        dist_min=dist_min,
        use_rk=use_rk,
        method=method,
    )

    title = vf
    title += " Discretized" if discretized else ""
    title += " Runge-Kutta" if use_rk else " ODEINT"
    title += " " + optimizer.method

    color = "white" if do_color else "green"

    # Build folder with images
    path_img = Path("img_optim")
    if not path_img.exists():
        os.mkdir(path_img)
    images = []
    idx = 0
    # Generate one plot every 2 iterations
    for list_routes in optimizer.optimize_route(q0, q1):
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
        plt.title(title)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        fout = path_img / f"{idx:03d}.png"
        idx += 1
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))

    # Plot best for some frames
    for i in range(6):
        route = list_routes[0]
        vectorfield.plot(
            x_min=xlim[0],
            x_max=xlim[1],
            y_min=ylim[0],
            y_max=ylim[1],
            step=step,
            do_color=do_color,
        )
        plt.plot(route.x, route.y, color=color, alpha=1)
        plt.scatter([q0[0], q1[0]], [q0[1], q1[1]], c="red")
        plt.title(title)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        fout = path_img / f"{idx:03d}.png"
        idx += 1
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))

    # Build output folder
    path_out: Path = Path(path_out)
    if not path_out.exists():
        path_out.mkdir()
    # Convert images to gif and delete images
    imageio.mimsave(path_out / "optimizer.gif", images)
    shutil.rmtree(path_img)


if __name__ == "__main__":
    typer.run(main)
