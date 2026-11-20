# PrimeVul · UniXcoder Colab baseline

The [Colab notebook](primevul_unixcoder_baseline_colab.ipynb) contains the code used for the completed 10-epoch classifier run on the original PrimeVul release. It reproduces the setup of the [published UniXcoder PrimeVul → PrimeVul baseline](https://arxiv.org/html/2403.18624v2#S5.T5) with a Hugging Face training loop, not the original implementation byte for byte. No Qwen, adapter-routing or RL results are included.

## Completed run

- NVIDIA L4, BF16, 7 h 32 min in the Trainer loop; best validation F1 **26.58%** at optimizer step **13,000**.
- On the 25,911-function test split: vulnerable-class F1 **22.84%**, VD-S **89.35%** (false-negative rate at a test ROC point with false-positive rate ≤0.5%).
- On the 564 matched pairs: **8 both correct**, **76 both vulnerable**, **476 both benign**, **4 reversed**, at the separate 0.5 probability cutoff.
- The saved scores rank the vulnerable member higher in **143** pairs, tied in **202**, and lower in **219**. This is a diagnostic, not the paper’s pair-correct metric.

## Reuse

Open the notebook in Colab and use a CUDA runtime with BF16 support. For a fresh run set `RESUME_FROM_LAST_CHECKPOINT = False` **and** set `PROJECT_DIR` to a new Drive directory; this protects the previous run’s outputs. For interrupted training, use the same run configuration and directory with `RESUME_FROM_LAST_CHECKPOINT = True`; resume from a full `checkpoint-*`, not `best_model/`.

The repository’s Git history contains only code and aggregate snapshots:

- [`colab/midpoint_baseline.json`](colab/midpoint_baseline.json): configuration, split counts, saved results, and the published comparator.
- [`colab/validation_history.csv`](colab/validation_history.csv): all 28 scheduled validation measurements at notebook display precision.
- [`colab/prediction_audit.json`](colab/prediction_audit.json): independent aggregate replay of the saved prediction files, including score ordering.

## Restore the complete Colab run

The [full-run GitHub Release](https://github.com/sujitu0806/primevul-peft-moe-grpo/releases/tag/primevul-unixcoder-midpoint-full) contains all 44 saved files, including the dataset copies, raw predictions, tokenizer, best model, and three resumable checkpoints. The three ZIPs are about 4.52 GB to download and restore to the same folder. They are Release assets, so a regular clone or pull does not download them.

Download all three ZIPs and `SHA256SUMS` from the release, or use the GitHub CLI, then unpack the complete snapshot:

```sh
mkdir -p run-downloads
gh release download primevul-unixcoder-midpoint-full \
  --repo sujitu0806/primevul-peft-moe-grpo --dir run-downloads
(cd run-downloads && shasum -a 256 -c SHA256SUMS) || exit 1
mkdir restored-run || exit 1
for archive in run-downloads/primevul_unixcoder_baseline-*.zip; do
  unzip -n "$archive" -d restored-run
done
```

Load the selected model and tokenizer without training:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "restored-run/primevul_unixcoder_baseline/best_model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
```

For training resume, point the notebook’s `PROJECT_DIR` to the restored `primevul_unixcoder_baseline` folder and use the latest full checkpoint, `checkpoint-28820`. This checkpoint is at the end of the original 10-epoch schedule; training beyond it requires a higher stopping limit. See the [upstream PrimeVul repository](https://github.com/DLVulDet/PrimeVul) for the original dataset and citation.

## Audit the reported metrics

With your own copies of the completed notebook and exports, regenerate the summaries locally without a GPU:

```sh
python3 colab/extract_midpoint.py /path/to/completed_notebook.ipynb --output-dir /path/to/output
python3 colab/audit_predictions.py /path/to/test_predictions.csv /path/to/paired_test_predictions.csv
python3 -B -m unittest discover -s colab -v
```
