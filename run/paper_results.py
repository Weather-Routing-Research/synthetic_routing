"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from hybrid_routing.pipeline import Pipeline

list_pipe = [
    Pipeline(p0=(3, 2), pn=(-7, 2), key="Circular"),
    Pipeline(p0=(0, 0), pn=(6, 2), key="FourVortices"),  # Ferraro et al.
]
list_plot = [
    {"extent": (-8, 8, 0, -3.5), "textbox_pos": (0, -3.5), "textbox_align": "bottom"},
    {"extent": (-2, 8, 0, 5.5), "textbox_pos": (0, 5.5), "textbox_align": "top"},
]

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()


"""
Run pipelines
"""

for idx, pipe in enumerate(list_pipe):
    pipe.solve_zivp(vel=1, time_iter=0.5, time_step=0.025)
    pipe.solve_dnj(num_iter=1000, time_step=0.01)

    k = pipe.key.lower().replace(" ", "-")
    dict_results = pipe.to_dict()

    # Define filename
    file = path_out / f"results-{k}"

    # Store plot
    plt.figure(figsize=(5, 5))
    # Define plot extent
    xmin, xmax = pipe.route_zivp.x.min(), pipe.route_zivp.x.max()
    ymin, ymax = pipe.route_zivp.y.min(), pipe.route_zivp.y.max()
    extent = (xmin - 1, xmax + 1, ymin - 1, ymax + 1)
    pipe.plot(**list_plot[idx])
    plt.savefig(file.with_suffix(".png"))
    plt.close()

    # Store dictionary
    with open(file.with_suffix(".json"), "w") as outfile:
        json.dump(dict_results, outfile)

    print(f"Done {k} vectorfield")
