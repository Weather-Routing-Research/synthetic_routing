import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import typer

from hybrid_routing.optimization import Route
from hybrid_routing.pipeline import Pipeline


def main(
    path_json: str,
    path_out: Optional[str] = None,
    num_iter: int = 2000,
):
    path_json: Path = Path(path_json)

    with open(path_json) as file:
        dict_json: dict = json.load(file)

    # Route from ZIVP
    dict_route = dict_json["route_zivp"]

    route = Route(
        x=dict_route["x"],
        y=dict_route["y"],
        t=dict_route["t"],
        theta=dict_route["theta"],
        geometry=dict_route["geometry"],
    )

    print("ZIVP")
    print(f"  Time (h): {route.t[-1] / 3600:.1f}")
    print(f"  Distance (km): {route.d.sum() / 1000:.1f}")

    # Route from DNJ
    dict_route = dict_json["route_dnj"]

    route_dnj = Route(
        x=dict_route["x"],
        y=dict_route["y"],
        t=dict_route["t"],
        theta=dict_route["theta"],
        geometry=dict_route["geometry"],
    )

    print("\nDNJ")
    print(f"  Time (h): {route_dnj.t[-1] / 3600:.1f}")
    print(f"  Distance (km): {route_dnj.d.sum() / 1000:.1f}")

    # Route from geodesic
    if "geodesic" in dict_route.keys():
        dict_route = dict_json["geodesic"]

        route_geod = Route(
            x=dict_route["x"],
            y=dict_route["y"],
            t=dict_route["t"],
            theta=dict_route["theta"],
            geometry=dict_route["geometry"],
        )

        print("\nGeodesic")
        print(f"  Time (h): {route_geod.t[-1] / 3600:.1f}")
        print(f"  Distance (km): {route_geod.d.sum() / 1000:.1f}")

    if path_out:
        path_out: Path = Path(path_out)
        path = "./data" if bool(dict_json["real"]) else None
        pipe = Pipeline(
            p0=(route.x[0], route.y[0]),
            pn=(route.x[-1], route.y[-1]),
            key=dict_json["key"],
            path=path,
        )
        pipe.add_route(route, vel=dict_json["vel"])
        print(f"\nDNJ redo, {num_iter} iterations")
        pipe.solve_dnj(num_iter=num_iter, optimize_for="time")

        route_dnj = pipe.route_dnj
        print(f"  Time (h): {route_dnj.t[-1] / 3600:.1f}")
        print(f"  Distance (km): {route_dnj.d.sum() / 1000:.1f}")

        pipe.plot()
        name = path_json.stem
        plt.savefig(path_out / (name.lower().replace(" ", "_") + ".png"))
        plt.close()


if __name__ == "__main__":
    typer.run(main)
