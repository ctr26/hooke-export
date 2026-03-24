"""Convert VCB benchmark results to HF-native model card metrics."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from huggingface_hub import EvalResult, ModelCard, ModelCardData


# VCB metric names that map cleanly to HF metric types
METRIC_TYPE_MAP: dict[str, str] = {
    "mse": "mse",
    "pearson": "pearson_r",
    "cosine": "cosine_similarity",
    "pearson_delta": "pearson_delta",
    "cosine_delta": "cosine_delta",
    "edistance": "energy_distance",
    "retrieval_mae": "retrieval_mae",
    "retrieval_edistance": "retrieval_edistance",
    "hit_score_error_mse": "hit_score_mse",
    "hit_score_error_mae": "hit_score_mae",
    "hit_ranking_spearman": "spearman_r",
    "hit_classification_accuracy": "accuracy",
    "hit_classification_precision": "precision",
    "hit_classification_recall": "recall",
    "hit_classification_f1_score": "f1",
    "hit_classification_auprc": "auprc",
    "hit_classification_auroc": "auroc",
    "hit_classification_enrichment_factor_5per": "enrichment_factor_5pct",
    "map_error_mse": "map_mse",
    "map_error_mae": "map_mae",
    "map_ranking_spearman": "map_spearman_r",
}


def vcb_results_to_eval_results(
    results: pl.DataFrame,
    dataset_name: str = "vcb",
    dataset_type: str = "valence-labs/vcb",
    task_type: str = "other",
    dataset_split: str | None = "test",
) -> list[EvalResult]:
    """Convert a VCB results DataFrame to a list of HF EvalResult objects.

    Aggregates per-perturbation scores to mean values per metric.

    Args:
        results: VCB results DataFrame with 'metric' and 'score' columns.
        dataset_name: Human-readable dataset name for the model card.
        dataset_type: Dataset identifier (HF dataset ID or custom).
        task_type: HF task type. VCB tasks don't map to standard HF tasks,
            so defaults to "other".
        dataset_split: Which data split was evaluated.

    Returns:
        List of EvalResult objects ready for a ModelCard.
    """
    summary = results.group_by("metric").agg(pl.col("score").mean().alias("mean_score"))

    eval_results = []
    for row in summary.iter_rows(named=True):
        metric_name = row["metric"]
        hf_metric_type = METRIC_TYPE_MAP.get(metric_name, metric_name)

        eval_results.append(
            EvalResult(
                task_type=task_type,
                dataset_type=dataset_type,
                dataset_name=dataset_name,
                metric_type=hf_metric_type,
                metric_value=round(row["mean_score"], 6),
                dataset_split=dataset_split,
            )
        )

    return eval_results


def push_model_card(
    repo_id: str,
    eval_results: list[EvalResult],
    model_name: str | None = None,
    spdx_license: str = "mit",
    tags: list[str] | None = None,
    description: str = "",
    create_pr: bool = False,
) -> str:
    """Create or update a HF model card with VCB evaluation results.

    Args:
        repo_id: HF repo ID (e.g. "user/model-name").
        eval_results: List of EvalResult from vcb_results_to_eval_results.
        model_name: Display name. Defaults to repo_id.
        license: SPDX license identifier.
        tags: Additional tags for the model card.
        description: Model description text.
        create_pr: If True, open a PR instead of pushing directly.

    Returns:
        URL of the pushed model card or PR.
    """
    model_name = model_name or repo_id.split("/")[-1]
    all_tags = ["vcb", "cellular-perturbation"]
    if tags:
        all_tags.extend(tags)

    card_data = ModelCardData(
        license=spdx_license,
        library_name="vcb",
        model_name=model_name,
        tags=all_tags,
        datasets=list({r.dataset_type for r in eval_results}),
        eval_results=eval_results,
    )

    card = ModelCard.from_template(
        card_data,
        model_id=model_name,
        model_description=description or f"Model evaluated with VCB (Virtual Cell Benchmark).",
    )

    card.push_to_hub(repo_id, create_pr=create_pr)
    return f"https://huggingface.co/{repo_id}"


def load_and_convert(
    results_path: str | Path,
    dataset_name: str = "vcb",
    dataset_type: str = "valence-labs/vcb",
    task_type: str = "other",
    dataset_split: str | None = "test",
) -> list[EvalResult]:
    """Load a VCB results parquet and convert to EvalResults."""
    df = pl.read_parquet(results_path)
    return vcb_results_to_eval_results(
        df,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        task_type=task_type,
        dataset_split=dataset_split,
    )
