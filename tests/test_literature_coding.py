import csv
import json
import unittest

from scripts.verify_literature_coding import DEFAULT_DIRECTORY, calculate


class LiteratureCodingTests(unittest.TestCase):
    def test_original_pass_agreement(self):
        result, paired = calculate(DEFAULT_DIRECTORY)
        self.assertEqual(result, json.loads((DEFAULT_DIRECTORY / 'agreement.json').read_text(encoding='utf-8')))
        self.assertEqual(result['cells'], 90)
        self.assertEqual(result['exact_agreements'], 66)
        self.assertEqual(result['agreements_by_requirement'], [0, 10, 8, 7, 10, 10, 10, 5, 6])
        self.assertAlmostEqual(result['cohens_kappa_unweighted'], .6104598737601442)
        self.assertFalse(result['final_map_used_in_agreement'])
        with (DEFAULT_DIRECTORY / 'paired_decisions.csv').open(encoding='utf-8', newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)), paired)

    def test_coder_identity_and_status_totals(self):
        metadata = json.loads((DEFAULT_DIRECTORY / 'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['second_coder'], 'Taha Benarbia')
        self.assertIn('shared RER specifications', metadata['instruction_basis'])
        self.assertIsNone(metadata['instruction_version_identifier'])
        result, _ = calculate(DEFAULT_DIRECTORY)
        self.assertEqual(result['primary_category_totals'], {'M': 25, 'P': 35, 'NR': 30})
        self.assertEqual(result['computed_category_totals'], {'M': 43, 'P': 14, 'NR': 33})


if __name__ == '__main__':
    unittest.main()
