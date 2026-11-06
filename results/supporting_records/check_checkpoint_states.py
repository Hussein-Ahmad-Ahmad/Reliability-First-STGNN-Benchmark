"""Compare retained checkpoint tensors without constructing or running models."""

from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    state = obj.get('model_state_dict', obj)
    fingerprint = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        fingerprint.update(key.encode())
        fingerprint.update(str(value.dtype).encode())
        fingerprint.update(str(tuple(value.shape)).encode())
        fingerprint.update(value.numpy().tobytes())
    return {'file_sha256': digest(path), 'tensor_sha256': fingerprint.hexdigest(),
            'epoch': obj.get('epoch'), 'tensor_count': len(state)}, state


def compare(left, right):
    if left.keys() != right.keys():
        return {'equal': False, 'same_keys': False}
    differences = [key for key in left if not torch.equal(left[key], right[key])]
    return {'equal': not differences, 'same_keys': True,
            'different_tensor_count': len(differences), 'examples': differences[:5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--repository-root', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    supplement = args.supplement
    with zipfile.ZipFile(supplement) as archive:
        manifest = json.loads(archive.read('results/release/export_verification.json'))
    indexed = {(c['model'], c['dataset'], c['seed']): c for c in manifest['cases']}
    required = {(m, d, s) for d, m in [
        ('METR-LA', 'D2STGNN'), ('METR-LA', 'MegaCRN'), ('METR-LA', 'STID'),
        ('PEMS-BAY', 'D2STGNN'), ('PEMS-BAY', 'STID'),
        ('PEMS04', 'D2STGNN'), ('PEMS04', 'STID')] for s in [43, 44, 45]}
    required.update((m, 'METR-LA', 43) for m in ['MTGNN', 'STGCNChebGraphConv', 'STAEformer', 'STNorm'])
    required.update(('MTGNN', d, 43) for d in ['PEMS-BAY', 'PEMS04'])
    rows = []
    for model, dataset, seed in sorted(required):
        local = args.repository_root / f'checkpoints/{model}/{dataset}_seed{seed}/best_model.pt'
        row = {'model': model, 'dataset': dataset, 'seed': seed, 'auxiliary_path': str(local)}
        if not local.is_file():
            row['status'] = 'auxiliary file unavailable'
            rows.append(row)
            continue
        meta, local_state = inspect(local)
        row['auxiliary'] = meta
        record = indexed.get((model, dataset, seed))
        if record:
            original = args.archive_root / record['checkpoint']
        else:
            candidates = list((args.archive_root / f'checkpoints/{model}/{dataset}_100_12_12_seed{seed}').glob(f'*/{model}_best_val_MAE.pt'))
            original = candidates[0] if len(candidates) == 1 else None
        if original and original.is_file():
            best_meta, best_state = inspect(original)
            row['archived_best_path'] = str(original)
            row['archived_best'] = best_meta
            row['best_matches_export_hash'] = (best_meta['file_sha256'] == record['checkpoint_sha256']) if record else None
            row['auxiliary_vs_best'] = compare(local_state, best_state)
            last_path = original.with_name(f'{model}_100.pt')
            if last_path.is_file():
                last_meta, last_state = inspect(last_path)
                row['archived_epoch100_path'] = str(last_path)
                row['archived_epoch100'] = last_meta
                row['auxiliary_vs_epoch100'] = compare(local_state, last_state)
            row['status'] = 'compared original tensor states'
        else:
            row['status'] = 'original archived file unavailable'
        rows.append(row)
        print(f"{model}/{dataset}/{seed}: best={row.get('auxiliary_vs_best', {}).get('equal')}, epoch100={row.get('auxiliary_vs_epoch100', {}).get('equal')}", flush=True)
    payload = {'training_performed': False, 'inference_performed': False,
               'archive_root': str(args.archive_root), 'cases': rows}
    (output / 'checkpoint_state_comparison.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
