import os
import shutil
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import typer

from synthrouting.optimization import DNJ, Optimizer, Route
from synthrouting.pipeline import Pipeline
from synthrouting.utils.config import load_config


def main(
    vf: str = "FourVortices",
    vel: float = 1,
    path_config: str = "./data/config.toml",
    do_color: bool = False,
    path_out: str = "output/",
):
    cfg = load_config(path_config)

    for key in ["synthetic", "real"]:
        if vf in cfg[key].keys():
            break
    else:
        raise ValueError(f"Key not found in config: {vf}")

    color = "orange" if do_color else "green"

    # Build folder with images
    path_img = Path("img_optim")
    if not path_img.exists():
        os.mkdir(path_img)
    images = []

    ####################################################################################
    #  VECTORFIELD
    ####################################################################################

    cfg_vf: Dict = cfg[key][vf]
    cfg_vf["key"] = vf
    pipeline = Pipeline(**cfg_vf)
    vectorfield = pipeline.vectorfield

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
            extent=pipeline._dict_plot.get("extent", None),
            do_color=do_color,
        )
        for route in list_routes:
            plt.plot(route.x, route.y, color=color, alpha=0.6)
        plt.scatter([pipeline.x0, pipeline.xn], [pipeline.y0, pipeline.yn], c="red")
        plt.suptitle(vf)
        plt.title(title)
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

    cfg_zivp: Dict = cfg["zivp"][key]
    cfg_zivp["vel"] = vel
    cfg_zivp.pop("num_points")

    optimizer = Optimizer(vectorfield, **cfg_zivp)

    idx = 0
    # Generate one plot every 10 iterations
    for list_routes in optimizer.optimize_route(
        [pipeline.x0, pipeline.y0], [pipeline.xn, pipeline.yn]
    ):
        if idx % 5 == 0:
            n = "exploration" if optimizer.exploration else "refinement"
            plot_routes_and_save(
                list_routes=list_routes,
                fout=path_img / f"{idx:03d}.png",
                title=f"HS ({n})",
                color=color,
            )
        idx += 1

    # Plot best for some frames
    for _ in range(6):
        plot_routes_and_save(
            list_routes=[list_routes[0]],
            fout=path_img / f"{idx:03d}.png",
            title="HS (result)",
            color=color,
        )
        idx += 1

    ####################################################################################
    # SMOOTHING DNJ
    ####################################################################################

    dnj = DNJ(vectorfield, time_step=cfg_zivp["time_step"], **cfg["dnj"][key])

    for n in range(40):
        route = list_routes[0]
        dnj.optimize_route(route, num_iter=dnj.num_iter // 40)
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
