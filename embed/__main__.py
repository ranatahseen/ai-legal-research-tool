
# python -m embed              → embed all cases from pi_cases.json
# python -m embed path/to/file → embed cases from a specific JSON file


import sys

from .pipeline import run_pipeline
from .config import PI_CASES_FILE


def main() -> None:
    cases_file = sys.argv[1] if len(sys.argv) > 1 else PI_CASES_FILE
    run_pipeline(cases_file)


if __name__ == "__main__":
    main()