# Lab 2 — CNN Bee Pollen

This repository trains and packages the four experiments required by the lab.

| ZIP output | Split | Model |
| --- | --- | --- |
| `baseline_original.zip` | original labels, 70/10/20 | CNN baseline |
| `improved_original.zip` | original labels, 70/10/20 | augmentation + regularisation |
| `traindev_original.zip` | original labels, 63/7/10/20 | improved model; train-dev diagnosis |
| `improved_corrected.zip` | corrected labels, 70/10/20 | improved model |

Each ZIP contains the best `.keras` checkpoint, train-history CSV, metrics JSON,
classification report, predictions, confusion matrix, training curves, the split
manifest and experiment configuration. Test labels are never used for model selection.

## Kaggle workflow

1. Create a Kaggle Notebook and add the Bee Pollen dataset as an **Input**.
2. In Notebook settings select **Accelerator → GPU T4 x2** and turn **Internet on**, because the notebook clones this GitHub repository.
3. Upload `notebooks/train_on_kaggle.ipynb`, set `DATA_DIR`, keep `RUN_MODE = 'original'` and `CORRECTIONS_CSV = None`, then run all cells. This creates the first three ZIPs and a review-candidate CSV.
4. Review candidate images manually and create a small Kaggle Dataset containing `corrections.csv`. Its columns are `relative_path,old_label,new_label,reason`.
5. Add that correction dataset as Input, set `RUN_MODE = 'corrected'` and set `CORRECTIONS_CSV` to its path. Run the notebook again to create the fourth ZIP.

Kaggle mounts inputs at `/kaggle/input/<dataset-slug>/`. For the original Ivan Feliciano dataset, set `DATA_DIR` to the **dataset root**; the script searches below it and derives labels from the original filenames: `P*.jpg` = `pollen`, `NP*.jpg` = `no_pollen`. It also supports the alternative layout with one folder per class. The exact slug is shown in the Input panel, for example:

```python
DATA_DIR = "/kaggle/input/datasets/ivanfel/honey-bee-pollen"
CORRECTIONS_CSV = "/kaggle/input/bee-pollen-corrections/corrections.csv"
```

Never write input data to `/kaggle/input`: it is read-only. The program writes its artefacts to `/kaggle/working`.

The Kaggle distribution may contain duplicate nested copies of the same source images. The training script removes byte-identical images with SHA-256 **before** splitting, preventing duplicate-image leakage between train, validation, and test.

## Label correction is manual

There is no valid automatic way to correct labels from a model's predictions. The first Kaggle run creates `correction_review_candidates.csv` from the incorrect predictions of the improved model. These are **review candidates**, not automatically wrong labels: manually inspect them first. To produce a full file listing, run:

```bash
python src/train_lab.py --data-dir /path/to/bee-pollen --export-manifest-only
```

It writes `label_audit_manifest.csv`. Fill `new_label` only for labels you have verified are wrong. The corrected run requires a non-empty corrections file, so the fourth experiment genuinely uses corrected data.

## Local usage

```bash
python src/train_lab.py \
  --data-dir /path/to/bee-pollen \
  --run-mode original \
  --output-dir outputs
```

After creating and reviewing `corrections.csv`, run `--run-mode corrected --corrections-csv /path/to/corrections.csv`.

Useful controls: `--epochs 20`, `--image-size 128`, `--batch-per-replica 32`. A fixed seed is used for all splits.
