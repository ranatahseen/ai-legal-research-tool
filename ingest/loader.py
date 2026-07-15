# loaders.py

import glob
import re
from dataclasses import dataclass

import fitz

from .config import INPUT_FOLDER, MIN_CASE_LENGTH


@dataclass
class RawCase:
    """A loaded PDF — filepath plus its extracted plain text."""
    filepath: str
    text: str


def clean_text(text: str) -> str:
    """
    Normalise raw PDF text.

    PyMuPDF sometimes produces null bytes and runs of whitespace/newlines.
    This collapses everything to single spaces and strips the edges.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(filepath: str) -> str | None:
    """
    Extract plain text from a single PDF.

    Returns None if the file cannot be opened or produces text shorter than
    MIN_CASE_LENGTH (too short to be a real judgment).
    """
    try:
        doc = fitz.open(filepath)
        text = "".join(page.get_text() for page in doc)
        doc.close()

        text = clean_text(text)

        if len(text) < MIN_CASE_LENGTH:
            print(f"  skip  {filepath}  (too short: {len(text)} chars)")
            return None

        return text

    except Exception as e:
        print(f"  error {filepath}: {e}")
        return None


def load_local_cases(folder: str = INPUT_FOLDER) -> list[RawCase]:
    """
    Load every PDF in `folder` and return a list of RawCase objects.

    Skips files that are too short or unreadable.
    Prints a summary line for each file so overnight runs are easy to monitor.
    """
    files = glob.glob(f"{folder}/*.pdf")

    if not files:
        print(f"No PDFs found in '{folder}/'")
        return []

    print(f"Found {len(files)} PDFs in '{folder}/'\n")

    cases = []
    for filepath in files:
        text = load_pdf(filepath)
        if text is None:
            continue
        cases.append(RawCase(filepath=filepath, text=text))
        print(f"  loaded  {filepath}  ({len(text):,} chars)")

    print(f"\n{len(cases)} of {len(files)} PDFs loaded successfully\n")
    return cases