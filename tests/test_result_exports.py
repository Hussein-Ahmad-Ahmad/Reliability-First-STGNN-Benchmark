"""Exercise the actual export methods without importing training dependencies."""

import ast
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ('base_tsf_runner.py', 'base_tsc_runner.py', 'base_utsf_runner.py')


def load_writer(filename):
    path = ROOT / 'framework/basicts/runners' / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == '_save_test_results')
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {'np': np, 'os': os}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['_save_test_results']


def batch(values):
    inputs = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
    return {'inputs': inputs, 'prediction': inputs + .25, 'target': inputs.copy()}


class ResultExportTests(unittest.TestCase):
    def exercise(self, sizes, repeats=1):
        for filename in RUNNERS:
            with self.subTest(runner=filename, sizes=sizes), tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
                runner = types.SimpleNamespace(
                    ckpt_save_dir=directory,
                    test_data_loader=types.SimpleNamespace(dataset=range(sum(sizes))))
                writer = load_writer(filename)
                for repeat in range(repeats):
                    offset = 0
                    values = np.arange(sum(sizes), dtype=np.float32) + 1 + repeat * 100
                    with patch.object(np.memmap, 'flush', autospec=True) as flush:
                        for index, size in enumerate(sizes):
                            writer(runner, index, batch(values[offset:offset + size]))
                            offset += size
                            self.assertEqual(flush.call_count, 3 if offset == sum(sizes) else 0)
                    for name, delta in [('inputs', 0), ('predictions', .25), ('targets', 0)]:
                        exported = np.fromfile(Path(directory) / 'test_results' / f'{name}.npy', dtype=np.float32)
                        np.testing.assert_array_equal(exported, values + delta)
                    self.assertEqual(runner._test_results_offset, sum(sizes))
                    self.assertEqual(np.mean(np.abs(runner._prediction_memmap - runner._target_memmap)), .25)
                    for name in ('_inputs_memmap', '_prediction_memmap', '_target_memmap'):
                        getattr(runner, name)._mmap.close()

    def test_partial_final_batch(self):
        self.exercise([8, 8, 3])

    def test_divisible_batches(self):
        self.exercise([8, 8])

    def test_single_batch(self):
        self.exercise([3])

    def test_repeated_exports(self):
        self.exercise([8, 8, 3], repeats=2)

    def test_variable_batch_lengths(self):
        self.exercise([5, 3, 8, 3])

    def test_invalid_lengths_and_order(self):
        for filename in RUNNERS:
            with self.subTest(runner=filename), tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
                runner = types.SimpleNamespace(ckpt_save_dir=directory,
                    test_data_loader=types.SimpleNamespace(dataset=range(19)))
                writer = load_writer(filename)
                with self.assertRaises(ValueError):
                    writer(runner, 1, batch([1]))
                bad = batch([1, 2])
                bad['target'] = bad['target'][:1]
                with self.assertRaises(ValueError):
                    writer(runner, 0, bad)
                with self.assertRaises(ValueError):
                    writer(runner, 0, batch([]))
                with self.assertRaises(ValueError):
                    writer(runner, 0, batch(range(20)))
                writer(runner, 0, batch(range(8)))
                with self.assertRaises(ValueError):
                    writer(runner, 2, batch(range(8)))
                for name in ('_inputs_memmap', '_prediction_memmap', '_target_memmap'):
                    getattr(runner, name)._mmap.close()


if __name__ == '__main__':
    unittest.main()
