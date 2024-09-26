"""
Generate all the figures used in the paper. Results section
"""

from synthrouting.pipeline import run_pipelines

run_pipelines(
    "synthetic",
    path_config="data/config.toml",
    path_out="output",
    max_thread=6,
)
