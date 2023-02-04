"""
Generate all the figures used in the paper. Results section
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.pipeline import Pipeline
from hybrid_routing.vectorfields import VectorfieldReal

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
Vectorfield - Real
"""

pipe = Pipeline(
    p0=(-79.7 * DEG2RAD, 32.7 * DEG2RAD), pn=(-29.5 * DEG2RAD, 38.5 * DEG2RAD)
)
pipe.add_vectorfield("real", path="./data", radians=True)

for vel in [10, 6, 3]:
    try:
        pipe.solve_zivp(
            vel=vel, num_angles=20, time_iter=360, time_step=60, dist_min=1000
        )
        pipe.solve_dnj(num_iter=20)

        # Store in dictionary
        dict_results[f"Real {vel}"] = pipe.to_dict()

        # Store plot
        pipe.plot()
        plt.tight_layout()
        plt.savefig(path_out / f"results-real-{vel}.png")
        plt.close()

    except Exception as er:
        print("[ERROR]", er)
    print(f"Done Real vectorfield, {vel} m/s\n---")

"""
Vectorfield - Real land
"""

vf = VectorfieldReal.from_folder("./data", "real-land", radians=True)
pipe = Pipeline(
    p0=(43.49 * DEG2RAD, -1.66 * DEG2RAD), pn=(98.14 * DEG2RAD, 10.21 * DEG2RAD)
)
pipe.add_vectorfield("real-land", path="./data", radians=True)

for vel in [10, 6, 3]:
    try:
        pipe.solve_zivp(
            vel=vel, num_angles=20, time_iter=360, time_step=60, dist_min=1000
        )
        pipe.solve_dnj(num_iter=20)

        # Store in dictionary
        dict_results[f"Real land {vel}"] = pipe.to_dict()

        # Store plot
        pipe.plot()
        plt.tight_layout()
        plt.savefig(path_out / f"results-real-{vel}.png")
        plt.close()
    except Exception as er:
        print("[ERROR]", er)
    print(f"Done Real vectorfield with land, {vel} m/s\n---")

"""
Store dictionary
"""
with open(path_out / "results-real.json", "w") as outfile:
    json.dump(dict_results, outfile)
    json.dump(dict_results, outfile)
