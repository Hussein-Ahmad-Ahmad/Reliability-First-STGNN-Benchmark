import json
from pathlib import Path
import types
import importlib
import sys
import torch
from thop import profile

ROOT = Path(r'D:/Hussein-Files/STGNN-Reliability-Benchmark')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIGS = ROOT / 'configs'
OUT_JSON = ROOT / 'results' / 'flops_profile_results.json'
OUT_MD = ROOT / 'results' / 'flops_profile_table.md'

TARGET_MODELS = ['D2STGNN','MegaCRN','MTGNN','STAEformer','STGCNChebGraphConv','STID','STNorm']
TARGET_DATASETS = ['METR-LA','PEMS-BAY','PEMS04']

ARCH_IMPORTS = {
    'D2STGNN': ('models.D2STGNN.arch.d2stgnn_arch', 'D2STGNN'),
    'MegaCRN': ('models.MegaCRN.arch.megacrn_arch', 'MegaCRN'),
    'MTGNN': ('models.MTGNN.arch.mtgnn_arch', 'MTGNN'),
    'STAEformer': ('models.STAEformer.arch.staeformer_arch', 'STAEformer'),
    'STGCNChebGraphConv': ('models.STGCNChebGraphConv.arch.stgcn_arch', 'STGCNChebGraphConv'),
    'STID': ('models.STID.arch.stid_arch', 'STID'),
    'STNorm': ('models.STNorm.arch.stnorm_arch', 'STNorm'),
}

def prepare_fake_config_package(model_name: str):
    if 'configs' not in sys.modules:
        pkg = types.ModuleType('configs')
        pkg.__path__ = []
        sys.modules['configs'] = pkg
    model_pkg_name = f'configs.{model_name}'
    if model_pkg_name not in sys.modules:
        mp = types.ModuleType(model_pkg_name)
        mp.__path__ = []
        sys.modules[model_pkg_name] = mp

    arch_mod_name = f'{model_pkg_name}.arch'
    mod_path, cls_name = ARCH_IMPORTS[model_name]
    real_mod = importlib.import_module(mod_path)
    cls = getattr(real_mod, cls_name)
    fake_arch = types.ModuleType(arch_mod_name)
    setattr(fake_arch, cls_name, cls)
    # STGCN configs import STGCN alias
    if model_name == 'STGCNChebGraphConv':
        setattr(fake_arch, 'STGCN', cls)
    sys.modules[arch_mod_name] = fake_arch

def pick_num_nodes(param):
    for k in ['num_nodes','num_of_vertices','node_num','nodes','num_node','n_vertex']:
        if k in param:
            try:
                return int(param[k])
            except Exception:
                pass
    return 207

def pick_channels(model, param):
    if model == 'D2STGNN':
        return 3
    if 'in_dim' in param:
        try:
            return max(1, int(param['in_dim']))
        except Exception:
            pass
    if 'input_dim' in param:
        try:
            return max(2, int(param['input_dim']) + 1)
        except Exception:
            pass
    if 'num_feat' in param:
        try:
            return max(1, int(param['num_feat']))
        except Exception:
            pass
    if model == 'STGCNChebGraphConv':
        return 1
    return 2

def load_hot_params(cfg_file: Path, model: str):
    prepare_fake_config_package(model)
    package = f'configs.{model}'
    txt = cfg_file.read_text(encoding='utf-8-sig')
    marker = '############################## General Configuration ##############################'
    if marker in txt:
        txt = txt.split(marker)[0]
    # strip relative imports we do not need for hot params
    keep = []
    for ln in txt.splitlines():
        if ln.strip().startswith('from .') and ('.arch import' not in ln):
            continue
        keep.append(ln)
    code = '\n'.join(keep) + '\n'

    g = {'__file__': str(cfg_file), '__name__': f'{package}.cfg_hot', '__package__': package}
    exec(compile(code, str(cfg_file), 'exec'), g, g)
    return g['MODEL_ARCH'], dict(g['MODEL_PARAM']), int(g.get('INPUT_LEN', 12)), int(g.get('OUTPUT_LEN', 12))

rows = []
errors = []

for model in TARGET_MODELS:
    mdir = CONFIGS / model
    if not mdir.exists():
        continue
    for ds in TARGET_DATASETS:
        cfg_file = mdir / f'{ds}_seed43.py'
        if not cfg_file.exists():
            errors.append({'model':model,'dataset':ds,'error':'missing config seed43'})
            continue
        try:
            model_arch, model_param, input_len, output_len = load_hot_params(cfg_file, model)
            num_nodes = pick_num_nodes(model_param)
            channels = pick_channels(model, model_param)

            history = torch.zeros((1, input_len, num_nodes, channels), dtype=torch.float32)
            future = torch.zeros((1, output_len, num_nodes, channels), dtype=torch.float32)
            if channels > 1:
                history[:, :, :, 1] = torch.rand((1, input_len, num_nodes))
                future[:, :, :, 1] = torch.rand((1, output_len, num_nodes))
            if channels > 2:
                history[:, :, :, 2] = torch.rand((1, input_len, num_nodes))
                future[:, :, :, 2] = torch.rand((1, output_len, num_nodes))

            model_inst = model_arch(**model_param)
            model_inst.eval()
            if model == 'MegaCRN':
                prof_inputs = (history, future, 0, 0)
            elif model == 'MTGNN':
                prof_inputs = (history,)
            else:
                prof_inputs = (history, future, 0, 0, False)
            with torch.no_grad():
                macs, _ = profile(model_inst, inputs=prof_inputs, verbose=False)

            params = int(sum(p.numel() for p in model_inst.parameters()))
            rows.append({
                'dataset': ds,
                'model': model,
                'macs': float(macs),
                'flops': float(macs) * 2.0,
                'params': params,
                'config': str(cfg_file).replace('\\\\', '/'),
                'input_shape': [1, input_len, num_nodes, channels],
            })
        except Exception as e:
            errors.append({'model':model,'dataset':ds,'error':str(e)})

rows.sort(key=lambda r: (TARGET_DATASETS.index(r['dataset']), TARGET_MODELS.index(r['model'])))

OUT_JSON.write_text(json.dumps({'rows':rows,'errors':errors}, indent=2), encoding='utf-8')

md = []
md.append('| Dataset | Model | Params | MACs | FLOPs | Input shape used | Source config |')
md.append('|---|---|---:|---:|---:|---|---|')
for r in rows:
    md.append(f"| {r['dataset']} | {r['model']} | {r['params']} | {r['macs']:.2f} | {r['flops']:.2f} | {r['input_shape']} | {r['config']} |")
OUT_MD.write_text('\\n'.join(md)+'\\n', encoding='utf-8')

print('rows', len(rows))
print('errors', len(errors))
for e in errors:
    print('ERR', e['dataset'], e['model'], '-', e['error'][:220])
print('json', OUT_JSON)
print('md', OUT_MD)
