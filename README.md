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
2. Create a small Kaggle Dataset containing your reviewed `corrections.csv`, then add it as an Input. Required columns are `relative_path,old_label,new_label,reason`.
3. In Notebook settings select **Accelerator → GPU T4 x2** and turn **Internet on**, because the notebook clones this GitHub repository.
4. Upload `notebooks/train_on_kaggle.ipynb`, edit only `DATA_DIR` and `CORRECTIONS_CSV`, then run all cells.
5. Click **Save Version** and download the four ZIPs from `/kaggle/working/DPL-Lab2_CNN-BeePollen/outputs/packages/`.

Kaggle mounts inputs at `/kaggle/input/<dataset-slug>/`. The exact slug is shown in the Input panel. `DATA_DIR` must be the directory that contains the class folders, for example:

```python
DATA_DIR = "/kaggle/input/honey-bee-pollen/Bee Pollen Images"
CORRECTIONS_CSV = "/kaggle/input/bee-pollen-corrections/corrections.csv"
```

Never write input data to `/kaggle/input`: it is read-only. The program writes its artefacts to `/kaggle/working`.

## Label correction is manual

There is no valid automatic way to correct labels from a model's predictions. Before the full run, prepare `corrections.csv` by manually reviewing images. To produce a full file listing, run:

```bash
python src/train_lab.py --data-dir /path/to/bee-pollen --export-manifest-only
```

It writes `label_audit_manifest.csv`. Fill `new_label` only for labels you have verified are wrong. The normal four-run command requires a non-empty corrections file, so the fourth experiment genuinely uses corrected data.

## Local usage

```bash
python src/train_lab.py \
  --data-dir /path/to/bee-pollen \
  --corrections-csv /path/to/corrections.csv \
  --output-dir outputs
```

Useful controls: `--epochs 20`, `--image-size 128`, `--batch-per-replica 32`. A fixed seed is used for all splits.
