import argparse
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_val_recall5(csv_path: str, output_path: Optional[str] = None) -> str:
	"""
	Read a metrics.csv exported by training and plot val/recall@5 over steps.

	If output_path is None, a PNG named 'val_recall5.png' will be saved next to the CSV.
	Returns the path to the saved image.
	"""
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	# Read CSV. Some rows contain training-only metrics; we keep rows where val/recall@5 is present
	df = pd.read_csv(csv_path)

	# Ensure required columns exist
	required_columns = {"step", "val/recall@5"}
	missing = [c for c in required_columns if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in CSV: {missing}")

	# Filter out rows where val/recall@5 is NaN
	val_df = df[df["val/recall@5"].notna()].copy()

	# Sort by step to ensure monotonic x-axis
	val_df.sort_values("step", inplace=True)

	# Determine output path
	if output_path is None:
		base_dir = os.path.dirname(os.path.abspath(csv_path))
		output_path = os.path.join(base_dir, "val_recall5.png")

	# Plot
	plt.figure(figsize=(8, 4.5))
	plt.plot(val_df["step"], val_df["val/recall@5"], marker="o", linewidth=1.5, markersize=3, label="val/recall@5")
	plt.xlabel("step")
	plt.ylabel("val/recall@5")
	plt.title("Validation Recall@5 over Steps")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.tight_layout()
	plt.legend()
	plt.savefig(output_path, dpi=150)
	plt.close()

	return output_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot val/recall@5 from metrics.csv")
	parser.add_argument("csv_path", type=str, help="Path to metrics.csv")
	parser.add_argument("--output", type=str, default=None, help="Output PNG path")
	args = parser.parse_args()

	out_path = plot_val_recall5(args.csv_path, args.output)
	print(out_path)


if __name__ == "__main__":
	main()


