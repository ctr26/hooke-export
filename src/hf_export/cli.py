"""CLI for exporting VCB results to Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(help="Export VCB benchmark metrics to Hugging Face model cards.")


@app.command()
def push(
    results_path: Annotated[Path, typer.Argument(help="Path to VCB results.parquet")],
    repo_id: Annotated[str, typer.Argument(help="HF repo ID (e.g. user/model-name)")],
    dataset_name: Annotated[str, typer.Option("--dataset-name", "-d")] = "vcb",
    dataset_type: Annotated[str, typer.Option("--dataset-type")] = "valence-labs/vcb",
    task_type: Annotated[str, typer.Option("--task-type", "-t")] = "other",
    dataset_split: Annotated[Optional[str], typer.Option("--dataset-split", "-s")] = "test",
    model_name: Annotated[Optional[str], typer.Option("--model-name", "-m")] = None,
    description: Annotated[str, typer.Option("--description")] = "",
    tags: Annotated[Optional[list[str]], typer.Option("--tag")] = None,
    create_pr: Annotated[bool, typer.Option("--create-pr/--no-pr")] = False,
    create_repo: Annotated[bool, typer.Option("--create-repo/--no-create-repo")] = False,
) -> None:
    """Push VCB benchmark results to a Hugging Face model card."""
    from huggingface_hub import create_repo as hf_create_repo

    from hf_export.converter import load_and_convert, push_model_card

    if create_repo:
        hf_create_repo(repo_id, exist_ok=True)

    eval_results = load_and_convert(
        results_path,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        task_type=task_type,
        dataset_split=dataset_split,
    )

    url = push_model_card(
        repo_id=repo_id,
        eval_results=eval_results,
        model_name=model_name,
        tags=tags or [],
        description=description,
        create_pr=create_pr,
    )

    typer.echo(f"Pushed {len(eval_results)} metrics to {url}")


@app.command()
def preview(
    results_path: Annotated[Path, typer.Argument(help="Path to VCB results.parquet")],
    dataset_name: Annotated[str, typer.Option("--dataset-name", "-d")] = "vcb",
    dataset_type: Annotated[str, typer.Option("--dataset-type")] = "valence-labs/vcb",
    task_type: Annotated[str, typer.Option("--task-type", "-t")] = "other",
    dataset_split: Annotated[Optional[str], typer.Option("--dataset-split", "-s")] = "test",
) -> None:
    """Preview the HF EvalResults without pushing."""
    from hf_export.converter import load_and_convert

    eval_results = load_and_convert(
        results_path,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        task_type=task_type,
        dataset_split=dataset_split,
    )

    for r in eval_results:
        typer.echo(f"  {r.metric_type}: {r.metric_value}")
