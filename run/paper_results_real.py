"""
Generate all the figures used in the paper. Results section
"""

import datetime as dt
import json
from pathlib import Path
from threading import Thread
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.pipeline import Pipeline

max_thread = 6  # Maximum number of threads allowed

list_pipes = [
    dict(
        p0=(-79.7, 32.7),
        pn=(-29.5, 38.5),
        key="real",
        path="./data",
        si_units=True,
    ),
    dict(
        p0=(43.49, -1.66),
        pn=(98.14, 10.21),
        key="real-land",
        path="./data",
        si_units=True,
    ),
]
list_vel = [10, 6, 3]

# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use("Agg")

"""
Create output folder
"""

# Custom folder for this date
today = dt.date.today().strftime("%y-%m-%d")
path_out: Path = Path(f"output/{today}")
if not path_out.exists():
    path_out.mkdir()

"""
Run pipelines
"""


def run_pipeline(dict_pipe: dict, vel: float):
    print(f"Initializing: {dict_pipe['key']}, vel = {vel}")
    pipe = Pipeline(**dict_pipe)

    pipe.solve_zivp(
        vel=vel,
        time_iter=3600,  # 1 hour (3600 s)
        time_step=600,  # Do not go below 10 min (600 s)
        dist_min=10000,
        num_angles=40,
        angle_amplitude=np.pi / 2,
        angle_heading=np.pi / 4,
        interp=200,
    )
    pipe.solve_dnj(num_iter=500, time_step=3600)

    # Store in dictionary
    k = pipe.key.lower().replace(" ", "-")
    dict_results = pipe.to_dict()

    # Decide filename
    file = path_out / f"results_{k}_{vel}"

    # Store plot
    plt.figure(dpi=120)
    pipe.plot()
    plt.savefig(file.with_suffix(".png"))
    plt.close()

    # Store dictionary
    with open(file.with_suffix(".json"), "w") as outfile:
        json.dump(dict_results, outfile)

    print(f"Done {k} vectorfield, {vel} m/s\n---")


# Maximum number of threads cannot be higher than number of processes
max_thread = min(max_thread, len(list_pipes) * len(list_vel))

# Initialize list of threads and index
threads: List[Thread] = [None for i in range(max_thread)]
n_thread = 0
for dict_pipe in list_pipes:
    for vel in list_vel:
        threads[n_thread] = Thread(target=run_pipeline, args=(dict_pipe, vel))
        threads[n_thread].start()
        n_thread += 1
        # If maximum index is reached, wait for all threads to finish
        if n_thread == max_thread:
            [t.join() for t in threads]
            n_thread = 0
            n_thread = 0
