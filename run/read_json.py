import json
from pathlib import Path

import matplotlib.pyplot as plt
import typer

from hybrid_routing.jax_utils import Route
from hybrid_routing.pipeline import Pipeline


def main(
    path_json: str,
    key: str = "route_zivp",
    path_out: str = "output",
    time_step: float = 3600,
    num_iter: int = 200,
):
    path_out = Path(path_out)

    with open(path_json) as file:
        dict_json: dict = json.load(file)

    # Loop over all runs in dictionary
    for name, dict_run in dict_json.items():
        print("\n---\nBenchmark, run:", name)

        # Try to generate the route
        try:
            route = Route(**dict_run[key], geometry=vf.geometry)
        except KeyError:
            raise KeyError(
                "Key not found in dict. Available are: ", ", ".join(dict_run.keys())
            )

        path = "./data" if bool(dict_run["real"]) else None
        pipe = Pipeline(
            p0=(route.x[0], route.y[0]),
            pn=(route.x[-1], route.y[-1]),
            key=dict_run["key"],
            path=path,
        )
        pipe.add_route(route, vel=dict_run["vel"])
        pipe.solve_dnj(num_iter=num_iter, time_step=time_step, optimize_for="time")
        pipe.plot()
        plt.savefig(path_out / (name.lower().replace(" ", "_") + ".png"))
        plt.close()


if __name__ == "__main__":
    typer.run(main)
