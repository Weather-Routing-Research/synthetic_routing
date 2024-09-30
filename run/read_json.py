import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import typer

from synthrouting.optimization import Route
from synthrouting.pipeline import Pipeline
from synthrouting.utils.plot import plot_routes


def open_json_and_plot(
    path_json: Path,
    path_out: Path = "output",
    num_iter: Optional[int] = None,
):
    with open(path_json) as file:
        dict_json: dict = json.load(file)

    # Route from ZIVP
    dict_route = dict_json["route"]["zivp"]

    route = Route(
        x=dict_route["x"],
        y=dict_route["y"],
        t=dict_route["t"],
        theta=dict_route["theta"],
        geometry=dict_route["geometry"],
    )

    # Load pipeline
    path = "./data" if bool(dict_json["real"]) else None
    pipe = Pipeline(
        p0=(route.x[0], route.y[0]),
        pn=(route.x[-1], route.y[-1]),
        key=dict_json["key"],
        path=path,
    )
    pipe.add_route(route, vel=dict_json["vel"])

    route.recompute_times(pipe.vel, pipe.vectorfield)
    print("ZIVP")
    print(f"  Time (h): {route.t[-1] / 3600:.1f}")
    print(f"  Distance (km): {route.d.sum() / 1000:.1f}")

    # Route from DNJ
    dict_route = dict_json["route"]["dnj"]

    route_dnj = Route(
        x=dict_route["x"],
        y=dict_route["y"],
        t=dict_route["t"],
        theta=dict_route["theta"],
        geometry=dict_route["geometry"],
    )
    route_dnj.recompute_times(pipe.vel, pipe.vectorfield)
    print("\nDNJ")
    print(f"  Time (h): {route_dnj.t[-1] / 3600:.1f}")
    print(f"  Distance (km): {route_dnj.d.sum() / 1000:.1f}")

    # Route from geodesic
    dict_route = dict_json["route"]["geodesic"]

    route_geod = Route(
        x=dict_route["x"],
        y=dict_route["y"],
        t=dict_route["t"],
        theta=dict_route["theta"],
        geometry=dict_route["geometry"],
    )
    route_geod.recompute_times(pipe.vel, pipe.vectorfield)
    print("\nMinimum distance")
    print(f"  Time (h): {route_geod.t[-1] / 3600:.1f}")
    print(f"  Distance (km): {route_geod.d.sum() / 1000:.1f}")

    plt.figure(dpi=120)
    plt.rcParams.update({"font.size": 10})
    plot_routes(
        [route, route_dnj, route_geod],
        pipe.vectorfield,
        labels=["HS (no smooth)", "HS (smooth)", "Min. dist."],
        vel=pipe.vel,
        alpha=0.6,
        fit_routes=False,
    )
    name = path_json.stem
    plt.tight_layout()
    plt.savefig(path_out / (name.lower().replace(" ", "_") + ".png"))
    plt.close()

    if num_iter:
        print(f"\nDNJ redo, {num_iter} iterations")
        pipe.solve_dnj(num_iter=num_iter)

        route_dnj = pipe.route_dnj
        print(f"  Time (h): {route_dnj.t[-1] / 3600:.1f}")
        print(f"  Distance (km): {route_dnj.d.sum() / 1000:.1f}")
        pipe.plot()

        plt.tight_layout()
        plt.savefig(name.lower().replace(" ", "_") + "_dnj.png")
        plt.close()


def main(
    path_in: str = "output",
    path_out: str = "output",
    num_iter: Optional[int] = None,
):
    path_in: Path = Path(path_in)
    path_out: Path = Path(path_out)

    if path_in.is_dir():
        list_json = [f for f in path_in.iterdir() if f.name.endswith("json")]
    else:
        list_json = [path_in]

    for path_json in list_json:
        print(f"\n-----\n{path_json.name}\n")
        open_json_and_plot(path_json, path_out=path_out, num_iter=num_iter)


if __name__ == "__main__":
    typer.run(main)
