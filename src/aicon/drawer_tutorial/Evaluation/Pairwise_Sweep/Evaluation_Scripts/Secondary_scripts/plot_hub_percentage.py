from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CSV_PATHS = {
    "Normal": Path(__file__).resolve().parents[2] / "data" / "normal" / "metrics" / "success" / "hub_parameters_interaction_summary.csv",
    "Disturbance": Path(__file__).resolve().parents[2] / "data" / "disturbance" / "metrics" / "success" / "hub_parameters_interaction_summary.csv",
    "Bad Prior": Path(__file__).resolve().parents[2] / "data" / "bad_prior" / "metrics" / "success" / "hub_parameters_interaction_summary.csv",
    "Noisy": Path(__file__).resolve().parents[2] / "data" / "noise" / "metrics" / "success" / "hub_parameters_interaction_summary.csv",
}


def read_percentage(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == 'percentage':
                try:
                    return float(row[1])
                except Exception:
                    return float('nan')
    raise ValueError(f"percentage not found in {csv_path}")


def main(output_dir: Path = None):
    labels = list(CSV_PATHS.keys())
    values = [read_percentage(CSV_PATHS[l]) for l in labels]

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot styling: clean, thesis-suitable
    try:
        plt.style.use('seaborn-white')
    except Exception:
        plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'], edgecolor='black', linewidth=0.6)

    # Y axis formatting
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Share of total interaction strength (%)', fontsize=11)
    ax.set_title('Interaction Strength Associated with the Two Hub Parameters',
             fontsize=12, pad=10)

    # Add numeric labels above bars
    for rect, val in zip(bars, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height + 1.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=9)

    # Minimal clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    fig.tight_layout()

    pdf_path = output_dir / 'hub_percentage.pdf'
    png_path = output_dir / 'hub_percentage.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
