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

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.pipeline import Pipeline

# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use("Agg")

max_thread = 6  # Maximum number of threads allowed

list_benchmark = [
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

# Maximum number of threads cannot be higher than number of processes
max_thread = min(max_thread, len(list_benchmark) * len(list_vel))

# Initialize list of pipelines
list_pipes: List[Pipeline] = [None for n in range(max_thread)]


def run_pipeline(n_thread: int, dict_pipe: dict, vel: float):
    print(f"Initializing: {dict_pipe['key']}, vel = {vel}")
    pipe = Pipeline(**dict_pipe)

    pipe.solve_zivp(
        vel=vel,
        time_iter=14400,  # 4 hour (4 * 3600 s)
        time_step=600,  # Do not go below 10 min (600 s)
        dist_min=10000,
        num_angles=21,
        angle_amplitude=90 * DEG2RAD,
        angle_heading=45 * DEG2RAD,
        num_points=200,
    )
    pipe.compute_geodesic()
    pipe.solve_dnj(num_iter=2000)

    # Append pipeline to list
    list_pipes[n_thread] = pipe

    # Decide filename
    file = path_out / pipe.filename

    # Store in dictionary (json)
    with open(file.with_suffix(".json"), "w") as outfile:
        dict_results = pipe.to_dict()
        json.dump(dict_results, outfile)

    print(f"Done {pipe.filename} vectorfield, {vel} m/s\n---")


# Initialize list of threads and index
threads: List[Thread] = [None for i in range(max_thread)]
n_thread = 0

for dict_pipe in list_benchmark:
    for vel in list_vel:
        threads[n_thread] = Thread(target=run_pipeline, args=(n_thread, dict_pipe, vel))
        threads[n_thread].start()
        n_thread += 1
        # If maximum index is reached, wait for all threads to finish
        if n_thread == max_thread:
            # Plot each thread independently, to avoid overlaps
            for n_thread, t in enumerate(threads):
                t.join()  # Waits for thread to finish before plotting
                pipe = list_pipes[n_thread]  # Get the associated pipeline
                # Decide filename
                file = path_out / pipe.filename
                # Plot results and store
                plt.figure(dpi=120)
                pipe.plot()
                plt.savefig(file.with_suffix(".png"))
                plt.close()
            # Reset thread number
            n_thread = 0
            n_thread = 0
