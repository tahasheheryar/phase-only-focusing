"""Regenerate the main and supplementary figures from saved simulation results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_rendering import (style, figure1, figure2, figure3, figure4, figure5,
                              supplement1, supplement2, supplement3,
                              supplement4, supplement5)

ROOT = Path(__file__).resolve().parent
SOURCE_DATA = ROOT / "source_data"
OUTPUT = ROOT / "results" / "reproduced_figures"


def load_results():
    with (SOURCE_DATA / "simulation_results.json").open(encoding="utf-8") as f:
        data = json.load(f)
    with (SOURCE_DATA / "receiver_gains.json").open(encoding="utf-8") as f:
        data["optical"] = json.load(f)
    return data


def main():
    data = load_results()
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    makers = [figure1, figure2, figure3, figure4, figure5]
    makers += [supplement1, supplement2, supplement3, supplement4, supplement5]
    for i, make_figure in enumerate(makers):
        name = (f"Figure_{i+1}" if i < 5 else f"Supplementary_Figure_{i-4}")
        fig = make_figure(data)
        fig.savefig(OUTPUT / f"{name}.pdf")
        fig.savefig(OUTPUT / f"{name}.png", dpi=300)
        plt.close(fig)
        print(name)


if __name__ == "__main__":
    main()
