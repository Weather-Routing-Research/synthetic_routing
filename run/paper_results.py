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

max_thread = 6  # Maximum number of threads allowed

list_pipe = [
    dict(p0=(3, 2), pn=(-7, 2), key="Circular"),
    dict(p0=(0, 0), pn=(6, 2), key="FourVortices"),  # Ferraro et al.
]
list_plot = [
    {"extent": (-8, 4, -4, 3), "textbox_pos": (0, -3.5), "textbox_align": "bottom"},
    {"extent": (-1, 7, -1, 6), "textbox_pos": (0, 5.5), "textbox_align": "top"},
]

# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use("Agg")

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()


"""
Run pipelines
"""


def run_pipeline(dict_pipe: dict, dict_plot: dict):
    print(f"Initializing: {dict_pipe['key']}")
    pipe = Pipeline(**dict_pipe)

    pipe.solve_zivp(vel=1, time_iter=0.5, time_step=0.025, num_points=200)
    pipe.solve_dnj(num_iter=10000, optimize_for="time")

    k = pipe.key.lower().replace(" ", "-")
    dict_results = pipe.to_dict()

    # Define filename
    file = path_out / f"results-{k}"

    # Store plot
    plt.figure(dpi=120)
    pipe.plot(**dict_plot)
    plt.savefig(file.with_suffix(".png"))
    plt.close()

    # Store dictionary
    with open(file.with_suffix(".json"), "w") as outfile:
        json.dump(dict_results, outfile)

    print(f"Done {k} vectorfield")


# Maximum number of threads cannot be higher than number of processes
max_thread = min(max_thread, len(list_pipe))

# Initialize list of threads and index
threads: List[Thread] = [None for i in range(max_thread)]
n_thread = 0
for idx, dict_pipe in enumerate(list_pipe):
    threads[n_thread] = Thread(target=run_pipeline, args=(dict_pipe, list_plot[idx]))
    threads[n_thread].start()
    n_thread += 1
    # If maximum index is reached, wait for all threads to finish
    if n_thread == max_thread:
        [t.join() for t in threads]
        n_thread = 0
