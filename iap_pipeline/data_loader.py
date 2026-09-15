"""
Data loader for the MentalManip dataset.

Uses the csv module (as recommended by the dataset authors) instead of pandas,
because pandas.read_csv does not parse the columns correctly for this dataset.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dialogue:
    id: str
    text: str
    label: int  # 1 = manipulative, 0 = non-manipulative
    techniques: list[str]  # e.g. ["Rationalization", "Denial"]
    vulnerabilities: list[str]


def _split_field(value: str) -> list[str]:
    """Split a comma-separated annotation field, stripping whitespace."""
    if not value or value.strip() == "":
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def load_dataset(path: str | Path) -> list[Dialogue]:
    """
    Load a MentalManip CSV (mentalmanip_maj.csv or mentalmanip_con.csv).

    Expected columns: ID, Dialogue, Manipulative, Technique, Vulnerability
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from:\n"
            "  https://github.com/audreycs/MentalManip/tree/main/mentalmanip_dataset"
        )

    dialogues: list[Dialogue] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        # Some exports include spaces after commas before quoted fields.
        # skipinitialspace=True keeps quoted dialogue/annotation columns aligned.
        reader = csv.DictReader(
            f,
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True,
        )
        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or missing a header row.")

        # Build column map: lowercase -> original field name
        col = {name.strip().lower(): name for name in reader.fieldnames if name is not None}

        required = {"id", "dialogue", "manipulative"}
        missing = required - set(col.keys())
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            raw_id = (row.get(col["id"], "") or "").strip()
            raw_text = (row.get(col["dialogue"], "") or "").strip()
            raw_label = (row.get(col["manipulative"], "") or "").strip()

            if not raw_id or not raw_text or raw_label == "":
                continue

            try:
                label = int(raw_label)
            except ValueError:
                continue

            dialogues.append(
                Dialogue(
                    id=raw_id,
                    text=raw_text,
                    label=label,
                    techniques=_split_field((row.get(col["technique"], "") or "")) if "technique" in col else [],
                    vulnerabilities=_split_field((row.get(col["vulnerability"], "") or "")) if "vulnerability" in col else [],
                )
            )

    return dialogues
