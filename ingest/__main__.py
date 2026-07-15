
#   python -m ingest          run the full ingestion pipeline
#   python -m ingest patch    fix null damages in existing pi_cases.json
#   python -m ingest stats    print dataset health report


import sys

from .dataset import patch_null_damages, print_stats
from .pipeline import run_pipeline

_USAGE = """
Usage:
  python -m ingest          Run the full ingestion pipeline
  python -m ingest patch    Fix null damages in existing pi_cases.json
  python -m ingest stats    Print dataset health report
"""

_COMMANDS = {"patch", "stats"}


def main() -> None:
    args = sys.argv[1:]

    if not args:
        run_pipeline()

    elif args[0] == "patch":
        patch_null_damages()
        print_stats()

    elif args[0] == "stats":
        print_stats()

    else:
        print(f"Unknown command: '{args[0]}'")
        print(_USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()