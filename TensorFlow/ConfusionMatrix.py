"""
Brady Barlow
Oklahoma State University
11/28/2025

TensorFlow Confusion Matrix Visualization
This script loads confusion matrix statistics from a JSON file, computes per-class metrics,
and generates visualizations including bar charts and small confusion matrix heatmaps.

Usage Example:
python -u ConfusionMatrix.py

"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPORT_DIR = "./export"
CONFUSION_JSON = os.path.join(EXPORT_DIR, "confusion_stats.json")
PLOT_PATH = os.path.join(EXPORT_DIR, "confusion_matrix_metrics.png")


def load_confusion(path: str = CONFUSION_JSON) -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(f"confusion_stats.json not found at {path}")
	with open(path, "r") as f:
		data = json.load(f)

	# Expecting {class: {TP,FP,TN,FN}}
	rows = []
	for cls, stats in data.items():
		tp = stats.get("TP", 0)
		fp = stats.get("FP", 0)
		tn = stats.get("TN", 0)
		fn = stats.get("FN", 0)

		precision = tp / (tp + fp + 1e-9)
		recall = tp / (tp + fn + 1e-9)
		f1 = 2 * precision * recall / (precision + recall + 1e-9)
		acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

		rows.append(
			{
				"class": cls,
				"TP": tp,
				"FP": fp,
				"TN": tn,
				"FN": fn,
				"precision": precision,
				"recall": recall,
				"f1": f1,
				"accuracy": acc,
			}
		)

	return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame, out_path: str = PLOT_PATH) -> None:
	classes = df["class"].tolist()
	x = np.arange(len(classes))

	width = 0.22
	fig, ax = plt.subplots(figsize=(10, 4))

	ax.bar(x - 1.5 * width, df["precision"], width, label="Precision")
	ax.bar(x - 0.5 * width, df["recall"], width, label="Recall")
	ax.bar(x + 0.5 * width, df["f1"], width, label="F1")
	ax.bar(x + 1.5 * width, df["accuracy"], width, label="Accuracy")

	ax.set_xticks(x)
	ax.set_xticklabels(classes, rotation=20)
	ax.set_ylim(0, 1.0)
	ax.set_ylabel("Score")
	ax.set_title("Per-class metrics from confusion stats")
	ax.grid(axis="y", alpha=0.3)
	ax.legend(fontsize=8)

	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"Saved metrics plot: {out_path}")


def plot_confusion_grid(df: pd.DataFrame, out_path: str | None = None) -> None:
	"""Plot a 2x2 confusion matrix per class as small heatmaps."""

	n_classes = len(df)
	cols = min(4, n_classes)
	rows = int(np.ceil(n_classes / cols))

	fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
	if not isinstance(axes, np.ndarray):
		axes = np.array([[axes]])
	axes = axes.flatten()

	for i, (_, row) in enumerate(df.iterrows()):
		tp, fp, tn, fn = row["TP"], row["FP"], row["TN"], row["FN"]
		mat = np.array([[tp, fn], [fp, tn]], dtype=float)
		total = mat.sum()
		if total > 0:
			mat = mat / total

		ax = axes[i]
		im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0)
		ax.set_title(row["class"])
		ax.set_xticks([0, 1])
		ax.set_yticks([0, 1])
		ax.set_xticklabels(["Pred +", "Pred -"], rotation=20, fontsize=8)
		ax.set_yticklabels(["True +", "True -"], fontsize=8)

		# annotate values
		for r in range(2):
			for c in range(2):
				ax.text(
					c,
					r,
					f"{mat[r, c]:.2f}",
					ha="center",
					va="center",
					color="black" if mat[r, c] < 0.6 else "white",
					fontsize=7,
				)

	# hide any unused axes
	for j in range(i + 1, len(axes)):
		axes[j].axis("off")

	fig.suptitle("Normalized 2x2 confusion per class", y=1.02)
	plt.tight_layout()
	if out_path is None:
		out_path = os.path.join(EXPORT_DIR, "confusion_matrices.png")
	plt.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved confusion grid: {out_path}")


def main():
	df = load_confusion()
	print(df)
	plot_metrics(df)
	plot_confusion_grid(df)


if __name__ == "__main__":
	main()

