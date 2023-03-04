"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path
from threading import Thread
from typing import List

import matplotlib
import matplotlib.pyplot as plt

from hybrid_routing.pipeline import Pipeline
from hybrid_routing.utils.config import load_config

# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use("Agg")

max_thread = 6  # Maximum number of threads allowed

config = load_config("data/config.toml", "synthetic")
list_benchmark = config.tolist()

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()


"""
Run pipelines
"""

# Maximum number of threads cannot be higher than number of processes
max_thread = min(max_thread, len(list_benchmark))

# Initialize list of pipelines
list_pipes: List[Pipeline] = [None for n in range(max_thread)]


def run_pipeline(n_thread: int, dict_pipe: dict):
    print(f"Initializing: {dict_pipe['key']}")
    pipe = Pipeline(**dict_pipe)

    pipe.solve_zivp(vel=1, time_iter=0.1, time_step=0.01, num_points=200)
    pipe.compute_geodesic()
    pipe.solve_dnj(num_iter=10000, optimize_for="time")

    # Append pipeline to list
    list_pipes[n_thread] = pipe

    # Define filename
    file = path_out / pipe.filename

    # Store dictionary
    with open(file.with_suffix(".json"), "w") as outfile:
        dict_results = pipe.to_dict()
        json.dump(dict_results, outfile)

    print(f"Done {pipe.filename} vectorfield")


# Initialize list of threads and index
threads: List[Thread] = [None for i in range(max_thread)]
n_thread = 0

for dict_pipe in list_benchmark:
    dict_plot = dict_pipe.pop("plot")
    threads[n_thread] = Thread(target=run_pipeline, args=(n_thread, dict_pipe))
    threads[n_thread].start()
    n_thread += 1
    # If maximum index is reached, wait for all threads to finish
    if n_thread == max_thread:
        # Plot each thread independently, to avoid overlaps
        for n_thread, t in enumerate(threads):
            t.join()  # Waits for thread to finish before plotting
            pipe = list_pipes[n_thread]  # Get the associated pipeline
            # Define filename
            file = path_out / pipe.filename
            # Plot results and store
            plt.figure(dpi=120)
            pipe.plot(**dict_plot)
            plt.savefig(file.with_suffix(".png"))
            plt.close()
        # Reset thread number
        n_thread = 0
        n_thread = 0
