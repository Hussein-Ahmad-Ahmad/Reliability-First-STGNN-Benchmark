"""Render three benchmark figures from frozen records without model inference."""

from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = None
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
DATASETS = ['METR-LA', 'PEMS-BAY', 'PEMS04']
MODELS = ['D2STGNN', 'MegaCRN', 'MTGNN', 'STNorm', 'STGCNChebGraphConv', 'STID', 'STAEformer']
COLORS = [BLUE, ORANGE, GREEN, '#6B7280', '#7A5195', '#56B4E9', '#CC79A7']
INPUTS, FIGURES = {}, {}
Z = None


def read(name):
    data = Z.read(name)
    INPUTS[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def save(fig, name, values):
    for ax in fig.axes:
        for direction in ['x', 'y']:
            lower, upper = getattr(ax, f'get_{direction}lim')()
            ticks = getattr(ax, f'get_{direction}ticks')()
            if direction == 'x' and name != 'uq/uq5_coverage_width_tradeoff':
                continue
            getattr(ax, f'set_{direction}ticks')([t for t in ticks if lower <= t <= upper])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (name, text.get_text())
        assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (name, text.get_text())
    outputs = []
    for suffix in ['.pdf', '.png']:
        path = OUTPUT / (name + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {'Creator': '', 'Producer': '', 'CreationDate': None, 'ModDate': None} if suffix == '.pdf' else {}
        fig.savefig(path, dpi=400, metadata=metadata)
        outputs.append({'path': (Path('figures-final/figures') / path.relative_to(OUTPUT)).as_posix(),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    FIGURES[name] = {'values': values, 'outputs': outputs, 'text_within_canvas': True}
    plt.close(fig)


def main():
    global Z, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--records-path', type=Path, required=True)
    args = parser.parse_args()
    Z = zipfile.ZipFile(args.supplement)
    OUTPUT = args.output_dir
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5, 'legend.fontsize': 9.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'grid.color': '#D1D5DB', 'grid.alpha': .6, 'grid.linewidth': .6})
    seeds = read('results/task1_point_forecasting/seed_metrics.json')
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9))
    fig.subplots_adjust(left=.105, right=.99, bottom=.38, top=.90, wspace=.53)
    values = {}
    for ax, dataset in zip(axes, DATASETS):
        values[dataset] = {}
        for index, (model, color) in enumerate(zip(MODELS, COLORS)):
            runs = seeds[dataset][model]
            assert set(runs) == {'43', '44', '45'}
            y = [float(np.mean([runs[str(s)][f'horizon_{h}']['MAE'] for s in [43, 44, 45]])) for h in [3, 6, 12]]
            values[dataset][model] = y
            ax.plot([3, 6, 12], y, color=color, marker=['o', 's', '^', 'D', 'v', 'P', 'X'][index],
                    linestyle='--' if index % 2 else '-', markersize=4, linewidth=1.15,
                    label='STGCN-Cheb' if model == 'STGCNChebGraphConv' else model)
        ax.set_title(dataset)
        ax.set_xticks([3, 6, 12])
        ax.set_xlabel('Forecast horizon')
        ax.set_ylabel('MAE')
        ax.grid(axis='y')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4,
               bbox_to_anchor=(.5, .01), frameon=False, columnspacing=1.2, handlelength=1.7)
    save(fig, 'point_forecast/pf2_horizon_profiles', values)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    fig.subplots_adjust(left=.12, right=.985, bottom=.35, top=.89, wspace=.78)
    values = {}
    for ax, dataset in zip(axes, DATASETS):
        values[dataset] = {}
        for variant, color, marker, label in [('fixed', ORANGE, 'o', 'Pooled'), ('per_horizon', BLUE, 's', 'Per horizon')]:
            row = read(f'results/task2_uncertainty/conformal/{dataset}_conformal_{variant}_metrics.json')['evaluation_set']
            values[dataset][variant] = {'MPIW': row['MPIW'], 'PICP': row['PICP']}
            ax.scatter(row['MPIW'], row['PICP'], color=color, marker=marker, s=38, label=label, zorder=3)
        ax.set_title(dataset)
        ax.set_xlabel('Width (dataset units)')
        ax.set_ylabel('Coverage')
        ax.set_ylim(.899, .908)
        ax.set_yticks([.900, .904, .908])
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')
        ax.locator_params(axis='x', nbins=3)
        ax.axhline(.9, color='#222222', linestyle=':', linewidth=1)
        ax.grid()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .005), frameon=False)
    save(fig, 'uq/uq5_coverage_width_tradeoff', values)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.65))
    fig.subplots_adjust(left=.09, right=.99, bottom=.37, top=.88, wspace=.72)
    values = {}
    budgets = [5, 10, 20, 50]
    for ax, model in zip(axes, ['D2STGNN', 'MegaCRN', 'STID', 'STAEformer']):
        row = read(f'results/task3_explainability/gnnexplainer/{model}/METR-LA_fidelity_metrics.json')
        selected = [row[f'k={k}']['mean_delta_important'] for k in budgets]
        random = [row[f'k={k}']['mean_delta_random'] for k in budgets]
        values[model] = {'budgets': budgets, 'selected': selected, 'random': random}
        x = np.arange(4)
        ax.bar(x - .18, selected, .36, color=BLUE, label='Selected')
        ax.bar(x + .18, random, .36, color='#6B7280', label='Random')
        ax.set_xticks(x, budgets)
        ax.set_xlabel('Sensors zeroed\n(k)', fontsize=9.5)
        ax.set_ylabel('Absolute forecast change')
        ax.set_title(model)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .005), frameon=False)
    save(fig, 'xai/xai2_gnnexplainer_delta_comparison', values)

    record = {'inputs': INPUTS, 'figures': FIGURES, 'training_performed': False,
              'inference_performed': False, 'font': 'DejaVu Sans'}
    args.records_path.parent.mkdir(parents=True, exist_ok=True)
    args.records_path.write_text(json.dumps(record, indent=2) + '\n')
    Z.close()
    print('Three figures generated from frozen numerical records.')


if __name__ == '__main__':
    main()
