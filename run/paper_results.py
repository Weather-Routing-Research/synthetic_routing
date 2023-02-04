"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from hybrid_routing.pipeline import Pipeline

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()
# Initialize dict of results
dict_results = {}


"""
Vectorfield - Circular
"""

pipe = Pipeline(p0=(3, 2), pn=(-7, 2), key="Circular")
pipe.solve_zivp(vel=1, time_iter=0.5, time_step=0.025)
pipe.solve_dnj(num_iter=1000, time_step=0.01)

dict_results["Circular"] = pipe.to_dict()

# Store plot
pipe.plot(extent=(-8, 4, -4, 6), textbox_pos=(0, -3.5), textbox_align="bottom")
plt.savefig(path_out / "results-circular.png")
plt.close()

print("Done Circular vectorfield")

"""
Vectorfield - Four Vortices
"""

# We will regenerate the results from Ferraro et al.
pipe = Pipeline(p0=(0, 0), pn=(6, 2), key="FourVortices")
pipe.solve_zivp(vel=1, time_iter=0.5, time_step=0.025)
pipe.solve_dnj(num_iter=1000, time_step=0.01)

# Store plot
pipe.plot(extent=(-0.5, 6.5, -1.5, 6), textbox_pos=(0, 5.5), textbox_align="top")
plt.savefig(path_out / "results-fourvortices.png")
plt.close()

print("Done Four Vortices vectorfield")

"""
Store dictionary
"""
with open(path_out / "results.json", "w") as outfile:
    json.dump(dict_results, outfile)
    json.dump(dict_results, outfile)
