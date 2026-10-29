from __future__ import annotations

import argparse

import numpy as np

from edfa_whitebox import load_chain_config
from edfa_whitebox.chain import _ase_grid, _spectroscopy, channel_plan
from edfa_whitebox.solver import FiberSpec, PumpSpec, solve_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve one EDF stage for debugging")
    parser.add_argument("--config", default="configs/whitebox_three_stage_demo.json")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--input-dbm", type=float, default=-24.0)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    config = load_chain_config(args.config)
    element = next(item for item in config["elements"] if item["id"] == args.stage)
    channels = config["channels"]
    wavelengths, _ = channel_plan(channels)
    grid = _ase_grid(channels)

    result = solve_stage(
        wavelengths,
        np.full(wavelengths.size, 10 ** (args.input_dbm / 10)),
        [PumpSpec(**item) for item in element["pumps"]],
        FiberSpec(**element["fiber"]),
        _spectroscopy(config["spectroscopy"]),
        ase_grid_nm=grid if grid.size else None,
        steps=args.steps,
    )
    print("converged:", result.converged, "iterations:", result.iterations)
    print("gain_db:", np.round(result.gain_db, 2))
    print("nf_db:", np.round(result.nf_db, 2))
    print("n2 (z=0, mid, L):", np.round(result.n2[[0, len(result.n2) // 2, -1]], 3))
    print("pump_out_mw:", result.pump_out_mw)


if __name__ == "__main__":
    main()
