if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from edfa_sim.cli import main
else:
    from .cli import main

raise SystemExit(main())
