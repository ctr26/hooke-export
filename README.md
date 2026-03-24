# hooke-export

Export [VCB](https://github.com/valence-labs/vcb) (Virtual Cell Benchmark) metrics to Hugging Face model cards.

Reads VCB `results.parquet` files, aggregates per-perturbation scores, and pushes them as native HF `EvalResult` metadata in model cards.

## Install

```bash
pip install git+https://github.com/ctr26/hooke-export.git
```

## Usage

### CLI

```bash
# Preview metrics before pushing
hooke-export preview results.parquet

# Push to a HF model card
hooke-export push results.parquet user/my-model --create-repo

# With options
hooke-export push results.parquet user/my-model \
  --dataset-name "VCB Drug Screen" \
  --task-type other \
  --tag drug-discovery \
  --create-pr
```

### Python

```python
import polars as pl
from hf_export import vcb_results_to_eval_results, push_model_card

results = pl.read_parquet("results.parquet")
eval_results = vcb_results_to_eval_results(results, dataset_name="vcb")
push_model_card("user/my-model", eval_results)
```
