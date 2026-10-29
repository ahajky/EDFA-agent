from __future__ import annotations

import argparse
import json
import sys

from .chain import load_chain_config, simulate_chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edfa_whitebox")
    parser.add_argument("config")
    parser.add_argument("--full", action="store_true", help="print per-channel spectra")
    args = parser.parse_args(argv)
    result = simulate_chain(load_chain_config(args.config))
    if not args.full:
        result = {key: value for key, value in result.items() if key != "spectra"}
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
