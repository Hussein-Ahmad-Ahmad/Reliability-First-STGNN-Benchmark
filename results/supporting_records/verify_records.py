"""Verify public supporting records without training, inference, or private arrays."""

from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import statistics


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def csv_rows(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-12):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=HERE.parents[1])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
    manifest = {}
    for line in (HERE / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        assert name not in manifest and Path(name).name == name
        manifest[name] = expected
    assert set(manifest) == {p.name for p in files}
    assert all(sha(HERE / name) == expected for name, expected in manifest.items())

    s1 = csv_rows('Table S1.csv')
    assert len(s1) == 10
    for row in s1:
        path = (root / row['source_file_in_v1.4.4']).resolve()
        path.relative_to(root)
        assert sha(path) == row['source_sha256']
        source = json.loads(path.read_text(encoding='utf-8'))
        close(row['clean_MAE'], source['baseline_mae'])
        close(row['selected_delta_MAE'], source['important_delta'])
        for prefix, key in [('uniform', 'uniform_random'), ('degree', 'degree_matched_random')]:
            control = source[key]
            draws = [x['delta'] for x in control['draws']]
            assert len(draws) == 30
            close(row[prefix + '_mean_delta'], statistics.mean(draws))
            close(row[prefix + '_sample_SD'], statistics.stdev(draws))
            rank = 1 + sum(x < source['important_delta'] for x in draws)
            assert rank == control['empirical_rank_among_31']
            if prefix == 'degree':
                assert rank == int(row['degree_rank_among_31'])

    boundary = load('calibration_boundary_sensitivity.json')
    assert (boundary['calibration_origins'], boundary['original_evaluation_origins'],
            boundary['excluded_evaluation_origins'], boundary['remaining_evaluation_origins']) == (3415, 3416, 11, 3405)
    assert boundary['excluded_boundary_valid_target_count'] == 0
    assert boundary['thresholds_unchanged']
    assert len(boundary['original_metric_reproduction']) == 117
    for row in boundary['original_metric_reproduction']:
        close(row['recomputed'], row['published_record'], row['tolerance'])
        assert row['passes']
    for method in boundary['methods'].values():
        for key in ['PICP', 'MPIW', 'valid_target_count']:
            close(method['original'][key], method['boundary_excluded'][key], 0)
        assert method['original']['valid_target_count'] == 7118436
    comparisons = boundary['comparisons']
    s2 = csv_rows('Table S2 all horizons.csv')
    assert len(s2) == len(comparisons)
    for public, source in zip(s2, comparisons):
        assert set(public) == set(source)
        for key, value in source.items():
            if isinstance(value, (int, float)):
                close(public[key], value)
            else:
                assert public[key] == str(value)

    checkpoints = load('checkpoint_states.json')['cases']
    assert len(checkpoints) == 27
    assert all(row['auxiliary_vs_epoch100']['equal'] for row in checkpoints)
    assert sum(not row['auxiliary_vs_best']['equal'] for row in checkpoints) == 25
    for row in checkpoints:
        assert row['auxiliary']['tensor_sha256'] == row['archived_epoch100']['tensor_sha256']
        assert row['auxiliary']['epoch'] == row['archived_epoch100']['epoch'] == 100

    codes = csv_rows('RER evidence map.csv')
    assert len(codes) == 117 and len({row['study_label'] for row in codes}) == 13
    assert len({(row['study_label'], row['RER_id']) for row in codes}) == 117
    assert Counter(row['status'] for row in codes if row['RER_id'] == 'RER-4') == Counter(P=6, NR=7)
    assert Counter(row['status'] for row in codes if row['RER_id'] == 'RER-8') == Counter(M=6, P=7)
    synthesis = csv_rows('Synthesis sources.csv')
    assert len(synthesis) == len({row['study_id'] for row in synthesis}) == 63
    assert Counter(row['publication_type'] for row in synthesis) == Counter(journal=19, conference=39, preprint=5)
    assert Counter(row['reference_group'] for row in synthesis) == Counter(historical_synthesis=59, purposive_addition=4)
    screening = csv_rows('Screening accounting.csv')
    assert len(screening) == 7

    figures = load('figure_generation.json')
    assert len(figures['figures']) == 3 and not any('entropy' in key for key in figures['figures'])
    for path, expected in figures['inputs'].items():
        assert sha(root / path) == expected
    ig = load('ig_overlap_figure.json')
    for path, expected in ig['input_records'].items():
        assert sha(root / path) == expected
    matrix = ig['membership_matrix']
    assert len(matrix) == 7 and all(len(row) == 20 for row in matrix)
    assert [sum(column) for column in zip(*matrix)] == ig['bar_counts']

    print(json.dumps({'status': 'passed', 'supporting_files': len(files),
                      'S1_cases': 10, 'S2_metric_reproduction_records': 117,
                      'checkpoint_cases': 27, 'different_from_best': 25,
                      'literature_codes': 117, 'synthesis_sources': 63,
                      'training_performed': False, 'inference_performed': False}, indent=2))


if __name__ == '__main__':
    main()
