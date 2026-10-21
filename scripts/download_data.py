"""Obtain or verify the public Olist dataset.

The script never invents a download URL. It points to the canonical Kaggle
dataset page and can optionally call the Kaggle CLI if it is installed and
authenticated on your machine.

Canonical source:
    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Licence: CC BY-NC-SA 4.0 (see data/README.md).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OLIST_FILES, RAW_DATA_DIR  # noqa: E402
from src.data_loader import olist_files_present  # noqa: E402

KAGGLE_SLUG = "olistbr/brazilian-ecommerce"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"


def check(raw_dir: Path) -> bool:
    missing = [name for name in OLIST_FILES.values() if not (raw_dir / name).exists()]
    print(f"Raw data directory: {raw_dir}")
    if missing:
        print("Missing files:")
        for name in missing:
            print(f"  - {name}")
        print(f"\nDownload from {KAGGLE_URL} and extract the CSVs into {raw_dir}.")
        return False
    print("All required Olist files are present.")
    return True


def download(raw_dir: Path) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        print(
            "Kaggle CLI is not installed. Either install it with\n"
            "  pip install kaggle\n"
            "and configure ~/.kaggle/kaggle.json, or download the archive manually from\n"
            f"  {KAGGLE_URL}\n"
            f"and extract the CSV files into {raw_dir}.",
            file=sys.stderr,
        )
        return 2
    command = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(raw_dir), "--unzip"]
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            "Kaggle download failed. Check your Kaggle credentials and dataset terms.",
            file=sys.stderr,
        )
        return result.returncode
    return 0 if check(raw_dir) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="Attempt download with the Kaggle CLI")
    parser.add_argument("--check", action="store_true", help="Only verify that the CSV files exist")
    args = parser.parse_args()

    if args.download:
        return download(args.raw_dir)
    ok = check(args.raw_dir)
    return 0 if ok or args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
