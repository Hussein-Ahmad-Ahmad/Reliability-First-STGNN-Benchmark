"""Recompute retained-array coverage after an evaluation-only boundary exclusion."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import zipfile

import numpy as np


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_source(archive, name, module_name):
    module = type(sys)(module_name)
    module.__file__ = name
    sys.modules[module_name] = module
    code = archive.read(name)
    exec(compile(code, name, 'exec'), module.__dict__)
    return module, hashlib.sha256(code).hexdigest()


def normalized_metrics(gci, y, mean, std, threshold):
    value = gci.interval_metrics(y, mean, std, threshold if np.isscalar(threshold) else threshold.reshape(1, -1, 1))
    value['per_horizon'] = gci.per_horizon_metrics(y, mean, std, threshold)
    return value


def plain_metrics(y, mean, half_width):
    lower, upper = mean - half_width, mean + half_width
    valid = np.isfinite(y) & y.astype(bool)
    covered = (y >= lower) & (y <= upper)
    result = {'PICP': float(covered[valid].mean()), 'MPIW': float(2 * half_width),
              'valid_target_count': int(valid.sum()), 'per_horizon': {}}
    for h in range(y.shape[1]):
        mask = valid[:, h, :]
        result['per_horizon'][f'horizon_{h + 1}'] = {
            'PICP': float(covered[:, h, :][mask].mean()), 'MPIW': float(2 * half_width),
            'valid_target_count': int(mask.sum())}
    return result


def verify_reproduction(actual, stored, name):
    records = []
    for label, a, b in [('aggregate', actual, stored)] + [
        (f'horizon_{h}', actual['per_horizon'][f'horizon_{h}'], stored['per_horizon'][f'horizon_{h}']) for h in range(1, 13)]:
        for key, tolerance in [('PICP', 1e-12), ('MPIW', 1e-5), ('valid_target_count', 0)]:
            passed = abs(a[key] - b[key]) <= tolerance
            records.append({'method': name, 'scope': label, 'metric': key,
                            'recomputed': a[key], 'published_record': b[key], 'tolerance': tolerance, 'passes': passed})
            assert passed, records[-1]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays-dir', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--raw-data', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(args.supplement)
    manifest = json.loads(archive.read('results/task2_uncertainty/conformal/METR-LA_ensemble_manifest.json'))
    inputs = {}
    for filename, key in [('ensemble_predictions.npy', 'ensemble_predictions_sha256'), ('targets.npy', 'targets_sha256')]:
        path = args.arrays_dir / filename
        digest = file_hash(path)
        assert digest == manifest[key], f'Use the current aligned arrays, not historical exports: {filename}'
        inputs[filename] = {'sha256': digest, 'bytes': path.stat().st_size}
    gci, code_hash = load_source(archive, 'scripts/generate_conformal_intervals.py', 'frozen_boundary_conformal')
    predictions = np.load(args.arrays_dir / 'ensemble_predictions.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(args.arrays_dir / 'targets.npy', mmap_mode='r', allow_pickle=False)
    assert predictions.shape == (6, 6831, 12, 207) and targets.shape == (6831, 12, 207)
    mean = np.asarray(predictions.mean(axis=0), dtype=np.float32)
    std = np.asarray(predictions.std(axis=0, ddof=0), dtype=np.float32)
    split, excluded = targets.shape[0] // 2, 11
    normalized = {key: json.loads(archive.read(f'results/task2_uncertainty/conformal/METR-LA_conformal_{key}_metrics.json'))
                  for key in ['fixed', 'per_horizon']}
    control = json.loads(archive.read('results/task2_uncertainty/conformal_sigma_control/METR-LA_conformal_sigma_control.json'))
    q = normalized['fixed']['calibration']['threshold']
    qh = np.asarray(normalized['per_horizon']['calibration']['thresholds'], dtype=np.float64)
    plain_q = control['plain_sigma1_control']['calibration']['half_width']
    # Reproduce the original calibration calculation as an input-consistency check.
    cal_y, cal_mean, cal_std = targets[:split], mean[:split], std[:split]
    valid = np.isfinite(cal_y) & cal_y.astype(bool)
    scores = np.full(cal_y.shape, np.nan, dtype=np.float32)
    scores[valid] = np.abs(cal_y[valid] - cal_mean[valid]) / (cal_std[valid] + gci.EPSILON)
    quantile_checks = []
    for label, score_pool, stored_q in [('pooled', scores, q)] + [(f'horizon_{h + 1}', scores[:, h, :], qh[h]) for h in range(12)]:
        computed_q = gci.higher_quantile(score_pool, .1)[0]
        tolerance = 0.0
        quantile_checks.append({'pool': label, 'computed': computed_q, 'stored': float(stored_q),
                                'tolerance': tolerance, 'passes': bool(abs(computed_q - stored_q) <= tolerance)})
        assert quantile_checks[-1]['passes'], quantile_checks[-1]
    plain_scores = np.full(cal_y.shape, np.nan, dtype=np.float32)
    plain_scores[valid] = np.abs(cal_y[valid] - cal_mean[valid])
    assert gci.higher_quantile(plain_scores, .1)[0] == plain_q
    methods, reproduction, rows = {}, [], []
    for method in ['normalized_pooled', 'normalized_per_horizon', 'plain_pooled']:
        evaluations = {}
        for label, begin in [('original', split), ('boundary_excluded', split + excluded)]:
            y, mu, sd = targets[begin:], mean[begin:], std[begin:]
            evaluations[label] = plain_metrics(y, mu, plain_q) if method == 'plain_pooled' else normalized_metrics(gci, y, mu, sd, q if method == 'normalized_pooled' else qh)
        original, sensitivity = evaluations['original'], evaluations['boundary_excluded']
        stored = (control['plain_sigma1_control'] if method == 'plain_pooled' else normalized['fixed' if method == 'normalized_pooled' else 'per_horizon'])['evaluation_set']
        reproduction.extend(verify_reproduction(original, stored, method))
        for scope in ['aggregate'] + [f'horizon_{h}' for h in range(1, 13)]:
            before = original if scope == 'aggregate' else original['per_horizon'][scope]
            after = sensitivity if scope == 'aggregate' else sensitivity['per_horizon'][scope]
            rows.append({'method': method, 'scope': scope,
                         'original_PICP': before['PICP'], 'boundary_excluded_PICP': after['PICP'],
                         'coverage_change_percentage_points': 100 * (after['PICP'] - before['PICP']),
                         'original_MPIW': before['MPIW'], 'boundary_excluded_MPIW': after['MPIW'],
                         'width_change': after['MPIW'] - before['MPIW'],
                         'original_valid_elements': before['valid_target_count'],
                         'boundary_excluded_valid_elements': after['valid_target_count']})
        methods[method] = evaluations
    boundary_y = targets[split:split + excluded]
    boundary_valid_count = int((np.isfinite(boundary_y) & boundary_y.astype(bool)).sum())
    raw_data_path = args.raw_data
    raw_data = np.memmap(raw_data_path, dtype='float32', mode='r', shape=(34272, 207, 3))
    test_start = len(raw_data) - int(len(raw_data) * .2)
    raw_boundary_matches = all(np.array_equal(targets[o], raw_data[test_start + o + 12:test_start + o + 24, :, 0])
                               for o in range(split, split + excluded))
    assert raw_boundary_matches
    output = {'dataset': 'METR-LA', 'calibration_origins': split, 'original_evaluation_origins': 3416,
              'excluded_evaluation_origins': excluded, 'remaining_evaluation_origins': 3405,
              'evaluation_window_start_indices_excluded_zero_based': list(range(split, split + excluded)),
              'thresholds_unchanged': True, 'calibration_quantile_consistency': quantile_checks,
              'normalized_pooled_threshold': q, 'normalized_horizon_thresholds': qh.tolist(), 'plain_half_width': plain_q,
              'validity_mask': 'finite targets not equal to zero', 'std_ddof': 0, 'epsilon': gci.EPSILON,
              'excluded_boundary_valid_target_count': boundary_valid_count,
              'excluded_boundary_targets_all_zero': bool((boundary_y == 0).all()),
              'boundary_targets_match_supplied_raw_dataset': raw_boundary_matches,
              'supplied_dataset_sha256': file_hash(raw_data_path),
              'boundary_raw_target_time_indices_inclusive': [test_start + split + 12, test_start + split + excluded - 1 + 23],
              'input_arrays': inputs, 'frozen_interval_code_sha256': code_hash,
              'numpy_version': np.__version__, 'original_metric_reproduction': reproduction,
              'methods': methods, 'comparisons': rows, 'training_performed': False, 'inference_performed': False,
              'scope': 'Saved-array sensitivity to evaluation-boundary exclusion, not an exchangeability or conditional-coverage guarantee.'}
    (args.output_dir / 'calibration_boundary_sensitivity.json').write_text(json.dumps(output, indent=2) + '\n')
    with (args.output_dir / 'calibration_boundary_sensitivity.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['scope'] == 'aggregate':
            print(json.dumps(row))
    print('Maximum absolute horizon coverage change (percentage points):', max(abs(r['coverage_change_percentage_points']) for r in rows if r['scope'] != 'aggregate'))
    print('Original metric checks:', len(reproduction), 'passed.')


if __name__ == '__main__':
    main()
