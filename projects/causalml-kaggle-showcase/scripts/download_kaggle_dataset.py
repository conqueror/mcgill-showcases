from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

DEFAULT_DATASET_SLUG = "faviovaz/marketing-ab-testing"
DEFAULT_RAW_DATA_DIR = Path("data/raw")

app = typer.Typer(help="Download the marketing A/B test dataset from Kaggle.")


@app.command()
def main(
    dataset: Annotated[
        str,
        typer.Option(help="Kaggle dataset slug in the format owner/dataset-name."),
    ] = DEFAULT_DATASET_SLUG,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to store the downloaded CSV."),
    ] = DEFAULT_RAW_DATA_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_path = shutil.which("kaggle")
    if kaggle_path is None:
        raise typer.BadParameter(
            "`kaggle` CLI is not installed. "
            "Run `uv add kaggle` and configure ~/.kaggle/kaggle.json."
        )

    command = [
        kaggle_path,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    typer.echo(f"Downloading dataset `{dataset}` into `{output_dir}`...")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    if completed.returncode != 0:
        typer.echo(completed.stdout)
        typer.echo(completed.stderr)
        raise RuntimeError(
            "Kaggle download failed. Verify API credentials and dataset slug."
        )

    csv_candidates = sorted(output_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in {output_dir} after download.")

    chosen_file = csv_candidates[0]
    canonical_target = output_dir / "marketing_ab.csv"
    if chosen_file != canonical_target:
        chosen_file.replace(canonical_target)

    typer.echo(f"Dataset ready at `{canonical_target}`")


if __name__ == "__main__":
    app()
