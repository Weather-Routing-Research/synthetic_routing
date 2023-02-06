"""
Generate all the figures used in the paper. Results section
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt

from hybrid_routing.pipeline import Pipeline

list_pipes = [
    Pipeline(
        p0=(-79.7, 32.7),
        pn=(-29.5, 38.5),
        key="real",
        path="./data",
        si_units=True,
    ),
    Pipeline(
        p0=(43.49, -1.66),
        pn=(98.14, 10.21),
        key="real-land",
        path="./data",
        si_units=True,
    ),
]
list_vel = [10, 6, 3]

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

"""
Run pipelines
"""

for pipe in list_pipes:
    for vel in list_vel:
        pipe.solve_zivp(
            vel=vel, num_angles=20, time_iter=3600, time_step=60, dist_min=10000
        )
        pipe.solve_dnj(num_iter=5, time_step=3600)

        # Store in dictionary
        k = pipe.key.lower().replace(" ", "-")
        dict_results[f"{k} {vel}"] = pipe.to_dict()

        # Store plot
        plt.figure(figsize=(5, 5))
        pipe.plot()
        plt.savefig(path_out / f"results-{k}-{vel}.png")
        plt.close()
        print(f"Done {k} vectorfield, {vel} m/s\n---")

"""
Store dictionary
"""
with open(path_out / "results-real.json", "w") as outfile:
    json.dump(dict_results, outfile)
    json.dump(dict_results, outfile)
