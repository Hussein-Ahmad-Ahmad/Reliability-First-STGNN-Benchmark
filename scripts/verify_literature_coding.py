"""Calculate unweighted agreement from two separate evidence-coding matrices."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / 'results/literature_coding'
DIMENSIONS = [f'RER-{index}' for index in range(1, 10)]
CODES = ('M', 'P', 'NR', 'N/A')


def load_matrix(path):
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['study', *DIMENSIONS]:
            raise ValueError(f'Unexpected coding columns: {path}')
        rows = list(reader)
    if not rows or len({row['study'] for row in rows}) != len(rows):
        raise ValueError(f'Empty matrix or duplicate studies: {path}')
    if any(row[dim] not in CODES for row in rows for dim in DIMENSIONS):
        raise ValueError(f'Unknown or missing coding status: {path}')
    return rows


def calculate(directory):
    primary = load_matrix(directory / 'primary_assignments.csv')
    second = load_matrix(directory / 'second_assignments.csv')
    if [row['study'] for row in primary] != [row['study'] for row in second]:
        raise ValueError('The coding matrices must contain the same ordered studies.')
    left_counts = Counter(row[dim] for row in primary for dim in DIMENSIONS)
    right_counts = Counter(row[dim] for row in second for dim in DIMENSIONS)
    agreements = [sum(left[dim] == right[dim] for left, right in zip(primary, second))
                  for dim in DIMENSIONS]
    cells = len(primary) * len(DIMENSIONS)
    observed = sum(agreements) / cells
    expected = sum(left_counts[code] * right_counts[code] for code in CODES) / cells**2
    result = {
        'cells': cells,
        'exact_agreements': sum(agreements),
        'agreement_percent': 100 * observed,
        'primary_category_totals': dict(left_counts),
        'computed_category_totals': dict(right_counts),
        'agreements_by_requirement': agreements,
        'expected_agreement': expected,
        'cohens_kappa_unweighted': (observed - expected) / (1 - expected) if expected < 1 else None,
        'agreement_stage': 'Before rule-based reconciliation',
        'primary_records': 'primary_assignments.csv',
        'second_records': 'second_assignments.csv',
        'final_map_used_in_agreement': False,
    }
    paired = [{'study': left['study'], 'requirement': dim,
               'primary': left[dim], 'second': right[dim],
               'exact_agreement': str(int(left[dim] == right[dim]))}
              for left, right in zip(primary, second) for dim in DIMENSIONS]
    return result, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--write', action='store_true', help='Generate numerical summary and paired decisions')
    args = parser.parse_args()
    result, paired = calculate(args.directory)
    summary_path = args.directory / 'agreement.json'
    pairs_path = args.directory / 'paired_decisions.csv'
    if args.write:
        summary_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        with pairs_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(paired)
    else:
        if json.loads(summary_path.read_text(encoding='utf-8')) != result:
            raise ValueError('Saved agreement summary does not match the two coding matrices.')
        with pairs_path.open(encoding='utf-8', newline='') as handle:
            if list(csv.DictReader(handle)) != paired:
                raise ValueError('Saved paired decisions do not match the coding matrices.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
