#!/usr/bin/env python3
"""Train and package the four Bee Pollen CNN experiments for Kaggle or local use."""
from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTO = tf.data.AUTOTUNE


@dataclass(frozen=True)
class RunSpec:
    name: str
    improved: bool
    split_kind: str


RUNS = (
    RunSpec("baseline_original", False, "standard"),
    RunSpec("improved_original", True, "standard"),
    RunSpec("traindev_original", True, "traindev"),
    RunSpec("improved_corrected", True, "standard"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing one subfolder per class.")
    parser.add_argument("--corrections-csv", type=Path,
                        help="CSV with relative_path,new_label columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-per-replica", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-manifest-only", action="store_true")
    return parser.parse_args()


def setup(seed: int) -> tf.distribute.Strategy:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    strategy = tf.distribute.MirroredStrategy()
    print(f"Visible GPUs: {len(gpus)}; replicas in sync: {strategy.num_replicas_in_sync}")
    return strategy


def samples(data_dir: Path) -> pd.DataFrame:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")
    rows = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            rows.append({"path": str(path.resolve()),
                         "relative_path": path.relative_to(data_dir).as_posix(),
                         "label": path.parent.name})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.label.nunique() < 2:
        raise ValueError("No class-organised images found. Point DATA_DIR at the class-folder parent.")
    return frame


def audit_manifest(frame: pd.DataFrame, output_dir: Path) -> Path:
    audit = frame[["relative_path", "label"]].rename(columns={"label": "old_label"})
    audit["new_label"] = ""
    audit["reason"] = ""
    result = output_dir / "label_audit_manifest.csv"
    audit.to_csv(result, index=False)
    return result


def corrected_frame(frame: pd.DataFrame, csv_path: Path | None) -> pd.DataFrame:
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError("Run 4 requires --corrections-csv; see README.md.")
    edits = pd.read_csv(csv_path, comment="#").fillna("")
    if not {"relative_path", "new_label"}.issubset(edits.columns):
        raise ValueError("corrections.csv requires relative_path and new_label columns.")
    edits = edits[edits.new_label.astype(str).str.strip() != ""].copy()
    if edits.empty:
        raise ValueError("corrections.csv has no corrections. Do not present unchanged labels as corrected data.")
    if edits.relative_path.duplicated().any():
        raise ValueError("corrections.csv has duplicated relative_path values.")
    changes = dict(zip(edits.relative_path, edits.new_label))
    unknown = set(changes) - set(frame.relative_path)
    if unknown:
        raise ValueError(f"Unknown correction path(s), first 5: {sorted(unknown)[:5]}")
    result = frame.copy()
    result["label"] = result.apply(lambda row: changes.get(row.relative_path, row.label), axis=1)
    print(f"Applied {len(changes)} verified label corrections.")
    return result


def split_frame(frame: pd.DataFrame, kind: str, seed: int) -> dict[str, pd.DataFrame]:
    counts = frame.label.value_counts()
    if (counts < 10).any():
        raise ValueError(f"Each class needs at least 10 images; got {counts[counts < 10].to_dict()}")
    train_val, test = train_test_split(frame, test_size=.20, stratify=frame.label, random_state=seed)
    train, val = train_test_split(train_val, test_size=.125, stratify=train_val.label, random_state=seed)
    result = {"train": train.copy(), "val": val.copy(), "test": test.copy()}
    if kind == "traindev":
        train, train_dev = train_test_split(train, test_size=.10, stratify=train.label, random_state=seed)
        result["train"] = train.copy()
        result["train_dev"] = train_dev.copy()
    return result


def dataset(frame: pd.DataFrame, label_ids: dict[str, int], image_size: int,
            batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
    paths = frame.path.astype(str).to_numpy()
    labels = frame.label.map(label_ids).astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(frame), 5000), seed=seed, reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return tf.cast(tf.image.resize(image, (image_size, image_size)), tf.float32) / 255., label
    return ds.map(decode, num_parallel_calls=AUTO).batch(batch_size).prefetch(AUTO)


def model(strategy: tf.distribute.Strategy, image_size: int, classes: int, improved: bool) -> tf.keras.Model:
    with strategy.scope():
        inputs = tf.keras.Input((image_size, image_size, 3))
        x = inputs
        if improved:
            x = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(.05),
                tf.keras.layers.RandomZoom(.10), tf.keras.layers.RandomContrast(.10),
            ], name="augmentation")(x)
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            if improved:
                x = tf.keras.layers.Dropout(.20)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if improved:
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(.35)(x)
        outputs = tf.keras.layers.Dense(classes, activation="softmax", dtype="float32")(x)
        result = tf.keras.Model(inputs, outputs)
        result.compile(optimizer=tf.keras.optimizers.Adam(3e-4 if improved else 1e-3),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return result


def plots(history: tf.keras.callbacks.History, matrix: np.ndarray, labels: list[str], folder: Path) -> None:
    hist = pd.DataFrame(history.history)
    hist.to_csv(folder / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.accuracy, label="train")
    axes[0].plot(hist.val_accuracy, label="validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(hist.loss, label="train")
    axes[1].plot(hist.val_loss, label="validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    picture = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(picture, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train_one(spec: RunSpec, frame: pd.DataFrame, strategy: tf.distribute.Strategy,
              args: argparse.Namespace) -> dict[str, object]:
    folder = args.output_dir / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    partitions = split_frame(frame, spec.split_kind, args.seed)
    for split_name, split in partitions.items():
        split[["relative_path", "label"]].assign(split=split_name).to_csv(
            folder / f"{split_name}_manifest.csv", index=False)

    labels = sorted(frame.label.unique())
    label_ids = {label: number for number, label in enumerate(labels)}
    batch_size = args.batch_per_replica * strategy.num_replicas_in_sync
    train_ds = dataset(partitions["train"], label_ids, args.image_size, batch_size, True, args.seed)
    val_ds = dataset(partitions["val"], label_ids, args.image_size, batch_size, False, args.seed)
    test_ds = dataset(partitions["test"], label_ids, args.image_size, batch_size, False, args.seed)

    tf.keras.backend.clear_session()
    gc.collect()
    cnn = model(strategy, args.image_size, len(labels), spec.improved)
    checkpoint = folder / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=.5, min_lr=1e-6),
    ]
    started = time.time()
    history = cnn.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    elapsed = time.time() - started

    # Test set is evaluated only after selecting the best validation checkpoint.
    best = tf.keras.models.load_model(checkpoint)
    probabilities = best.predict(test_ds, verbose=0)
    truth = partitions["test"].label.map(label_ids).to_numpy()
    predicted = probabilities.argmax(axis=1)
    test_loss, test_accuracy = best.evaluate(test_ds, verbose=0)
    matrix = confusion_matrix(truth, predicted, labels=range(len(labels)))
    macro_f1 = f1_score(truth, predicted, average="macro", zero_division=0)
    report = classification_report(truth, predicted, target_names=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(folder / "classification_report.csv")
    output = partitions["test"][["relative_path", "label"]].copy()
    output["predicted_label"] = [labels[index] for index in predicted]
    output["confidence"] = probabilities.max(axis=1)
    output.to_csv(folder / "test_predictions.csv", index=False)
    plots(history, matrix, labels, folder)

    metrics: dict[str, object] = {
        "run": spec.name, "improved": spec.improved, "split_kind": spec.split_kind,
        "test_accuracy": float(test_accuracy), "test_loss": float(test_loss),
        "test_macro_f1": float(macro_f1), "elapsed_seconds": round(elapsed, 2),
        "classes": labels, "batch_size": batch_size,
    }
    if "train_dev" in partitions:
        train_dev_ds = dataset(partitions["train_dev"], label_ids, args.image_size, batch_size, False, args.seed)
        train_loss, train_accuracy = best.evaluate(train_ds, verbose=0)
        td_loss, td_accuracy = best.evaluate(train_dev_ds, verbose=0)
        metrics.update({"train_accuracy": float(train_accuracy), "train_loss": float(train_loss),
                        "train_dev_accuracy": float(td_accuracy), "train_dev_loss": float(td_loss)})
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (folder / "run_config.json").write_text(json.dumps(
        {"arguments": vars(args), "spec": asdict(spec), "label_to_id": label_ids},
        indent=2, default=str), encoding="utf-8")
    return metrics


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = samples(args.data_dir)
    manifest = audit_manifest(frame, args.output_dir)
    print(f"Found {len(frame)} images / {frame.label.nunique()} classes. Audit manifest: {manifest}")
    if args.export_manifest_only:
        return
    strategy = setup(args.seed)
    corrected = corrected_frame(frame, args.corrections_csv)
    packages = args.output_dir / "packages"
    packages.mkdir(exist_ok=True)
    summary = []
    for spec in RUNS:
        print(f"\n{'=' * 18} {spec.name} {'=' * 18}")
        summary.append(train_one(spec, corrected if spec.name == "improved_corrected" else frame, strategy, args))
        archive = shutil.make_archive(str(packages / spec.name), "zip", root_dir=args.output_dir / spec.name)
        print(f"Saved {archive}")
    pd.DataFrame(summary).to_csv(args.output_dir / "metrics_summary.csv", index=False)
    print(f"Done. Four ZIP files: {packages}")


if __name__ == "__main__":
    main()
