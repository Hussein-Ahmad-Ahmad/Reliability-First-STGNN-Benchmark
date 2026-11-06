"""Render the existing IG overlap data with descriptive labels and no inference."""

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGURE = 'figures-final/figures/xai/xai5_ig_sensor_consensus.png'
MODELS = ['D2STGNN', 'MegaCRN', 'MTGNN', 'STNorm', 'STGCN-Cheb', 'STID', 'STAEformer']
ORIGINAL_DISPLAY = [67, 166, 90, 203, 96, 123, 35, 16, 196, 57, 93, 149, 105, 107, 176, 163, 0, 2, 28, 162]
ORIGINAL_COUNTS = [4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--records-path', type=Path, required=True)
    parser.add_argument('--previous-figure', type=Path)
    args = parser.parse_args()
    inputs, sets = {}, []
    archive_path = args.supplement
    with zipfile.ZipFile(archive_path) as archive:
        for model in MODELS:
            name = model.replace('STGCN-Cheb', 'STGCNChebGraphConv')
            path = f'results/task3_explainability/integrated_gradients/{name}_METR-LA_ig_results.json'
            data = archive.read(path)
            inputs[path] = hashlib.sha256(data).hexdigest()
            sensors = list(map(int, json.loads(data)['top_10_sensors']))
            assert len(sensors) == len(set(sensors)) == 10
            sets.append(sensors)
    counts = Counter(s for values in sets for s in values)
    display = [s for s, _ in counts.most_common(20)]
    frequencies = [counts[s] for s in display]
    assert display == ORIGINAL_DISPLAY and frequencies == ORIGINAL_COUNTS
    matrix = np.array([[int(s in values) for s in display] for values in sets])
    assert matrix.sum(axis=0).tolist() == ORIGINAL_COUNTS
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'axes.titlesize': 15, 'axes.labelsize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 12,
                         'legend.fontsize': 10.5, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.825, 5.105), dpi=200)
    fig.subplots_adjust(left=.045, right=.993, top=.82, bottom=.20, wspace=.26)
    fig.suptitle('Integrated Gradients: Cross-model top-10 sensor overlap (METR-LA)', fontsize=17, fontweight='bold', y=.98)
    colors = ['#EF5350' if v >= 3 else '#FFA726' if v == 2 else '#9FB0B8' for v in frequencies]
    axes[0].barh(np.arange(20), frequencies, color=colors, height=.72)
    axes[0].set_yticks(np.arange(20), [f'#{s}' for s in display])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 7.5)
    axes[0].set_xticks(range(8))
    axes[0].set_title('Sensor inclusion frequency')
    axes[0].set_xlabel('Number of models including sensor in top-10')
    axes[0].axvline(3, color='#EF5350', linestyle='--', linewidth=1.2, label='Included in 3+ models')
    axes[0].axvline(2, color='#FFA726', linestyle=':', linewidth=1.2, label='Included in 2 models')
    axes[0].legend(loc='upper right', frameon=True)
    axes[1].imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    axes[1].set_title('Sensor presence in each model\'s top-10')
    axes[1].set_yticks(range(7), MODELS)
    axes[1].set_xticks(range(20), [f'#{s}' for s in display], rotation=55, ha='right', fontsize=10)
    axes[1].tick_params(axis='both', length=0)
    for row, column in np.argwhere(matrix):
        axes[1].text(column, row, '+', color='white', ha='center', va='center', fontsize=11)
    dest = args.output_dir / Path(FIGURE).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor='white', metadata={'Software': None, 'Author': 'Hussein Ahmad Ahmad, Seyyed Kasra Mortazavi, Taha Benarbia, Fadi Al Machot, Kyandoghere Kyamakya'})
    plt.close(fig)
    payload = {'figure_path': FIGURE, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
               'previous_sha256': hashlib.sha256(args.previous_figure.read_bytes()).hexdigest() if args.previous_figure and args.previous_figure.is_file() else None,
               'source_archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
               'input_records': inputs, 'model_order': MODELS, 'top10_sets': sets,
               'displayed_sensor_order': display, 'bar_counts': frequencies, 'membership_matrix': matrix.tolist(),
               'bar_count_checks': 20, 'membership_cells': 140,
               'original_sensor_order_preserved': display == ORIGINAL_DISPLAY,
               'numeric_data_unchanged': frequencies == ORIGINAL_COUNTS and matrix.sum(axis=0).tolist() == ORIGINAL_COUNTS,
               'scope': 'Plot presentation only. Source sensor sets and the original plotted ordering are preserved.',
               'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__,
               'training_performed': False, 'inference_performed': False}
    args.records_path.parent.mkdir(parents=True, exist_ok=True)
    args.records_path.write_text(json.dumps(payload, indent=2) + '\n')
    print('Rendered original 20 frequencies and 140 membership cells with overlap labels.')


if __name__ == '__main__':
    main()
