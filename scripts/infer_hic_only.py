from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from puget.hic_only_data import (
    HiCFoundationRawDataset, collate_hicfoundation, init_hicfoundation_worker,
    load_biosample_table,
)
from scripts._puget_config import load_config as load_inference_config
from scripts.train_hic_only import load_config, make_model
from puget.hic_only_model import HiCFoundationLit


def group_inputs(config: dict, group: str):
    names = list(config['split']['train_biosamples' if group == 'train14' else 'test_biosamples'])
    test_label = Path(config['data']['test_label'])
    labels = test_label.with_name('traincell_testgene.npy') if group == 'train14' else test_label
    return names, labels


def output_paths(directory: Path, run: str, group: str):
    stem = f'{run}_testgenes_train14' if group == 'train14' else run
    suffix = '_pred' if group == 'train14' else '_testpred'
    names_suffix = '_biosamples' if group == 'train14' else '_test_biosamples'
    return (directory / f'{stem}{suffix}.npy',
            directory / f'{stem}{names_suffix}.txt',
            directory / f'{stem}_metadata.json')


def build_loader(config: dict, group: str, batch_size: int, workers: int):
    data = config['data']
    names, labels = group_inputs(config, group)
    table = {name: accession for _row, accession, name in load_biosample_table(data['biosamples_csv'])}
    bedpe = Path(data['test_bedpe'])
    n_genes = sum(bool(line.strip()) for line in bedpe.open())
    values = np.load(labels, mmap_mode='r')
    if values.shape != (len(names), n_genes) or values.dtype != np.float32:
        raise ValueError(f'{labels}: expected {(len(names), n_genes)} float32 labels')
    paths = [Path(data['hic_root']) / 'test' / f'{table[name]}.pkl' for name in names]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} test-gene raw Hi-C files; first: {missing[0]}')
    dataset = HiCFoundationRawDataset(
        bedpe_path=str(bedpe), pkl_paths=[str(path) for path in paths],
        label_rows=list(range(len(names))), biosample_names=names,
        label_npy=str(labels), size=int(data['size']),
        skip_missing=bool(data.get('skip_missing_hic', True)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0,
                        prefetch_factor=2 if workers > 0 else None,
                        collate_fn=collate_hicfoundation,
                        worker_init_fn=init_hicfoundation_worker if workers > 0 else None)
    return loader, names, n_genes


def atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='inference YAML in configs/inference')
    parser.add_argument('--gpu', type=int, default=None,
                        help='physical GPU index for this job (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.gpu is not None:
        if args.gpu < 0:
            raise SystemExit('--gpu must be a non-negative GPU index')
        # CUDA is not initialized before this point, so this selects the device.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    inference = load_inference_config(args.config)
    config = load_config(str(ROOT / inference['training_config']))
    checkpoint = ROOT / inference['checkpoint']
    output_dir = ROOT / inference['output_dir']
    batch_size, workers = int(inference['batch_size']), int(inference['num_workers'])
    if inference['group'] not in ('train14', 'test2', 'both'):
        raise ValueError('group must be train14, test2, or both')
    groups = ('train14', 'test2') if inference['group'] == 'both' else (inference['group'],)
    run = str(config['run_name'])
    if config['data']['input_semantics'] != 'raw_none':
        raise ValueError('HiCFoundation inference requires raw/NONE windows')
    if config['decoder']['output_activation'] != 'softplus':
        raise ValueError('The raw Hi-C baseline requires Softplus output')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for group in groups:
        names, labels = group_inputs(config, group)
        table = {name: accession for _row, accession, name in load_biosample_table(config['data']['biosamples_csv'])}
        n_genes = sum(bool(line.strip()) for line in Path(config['data']['test_bedpe']).open())
        values = np.load(labels, mmap_mode='r')
        if values.shape != (len(names), n_genes):
            raise ValueError(f'{labels}: expected {(len(names), n_genes)} labels')
        missing = [name for name in names if not (Path(config['data']['hic_root']) / 'test' / f'{table[name]}.pkl').is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} {group} test-gene raw windows')
    if not args.preflight_only and not args.overwrite:
        existing = [path for group in groups for path in output_paths(output_dir, run, group)
                    if path.exists()]
        if existing:
            raise FileExistsError(f'Output exists: {existing}')
    print(f'Preflight passed: {run}; groups={groups}; checkpoint={checkpoint}')
    if args.preflight_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; inference requires a GPU')
    device = torch.device('cuda')
    model = HiCFoundationLit.load_from_checkpoint(str(checkpoint), model=make_model(config['encoder'], config['decoder']), map_location='cpu').eval().to(device)
    precision = str(config['training']['precision'])
    use_amp = 'mixed' in precision
    amp_dtype = torch.bfloat16 if 'bf16' in precision else torch.float16
    for group in groups:
        paths = output_paths(output_dir, run, group)
        loader, names, n_genes = build_loader(config, group, batch_size, workers)
        predictions = np.full((len(names), n_genes), np.nan, dtype=np.float32)
        with torch.inference_mode():
            for image, count, _target, metadata in loader:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = model(image.to(device), count.to(device))
                for value, meta in zip(output.float().cpu().numpy().reshape(-1), metadata):
                    predictions[int(meta[1]), int(meta[2])] = float(value)
        finite = np.isfinite(predictions)
        if not finite.any(axis=1).all() or np.isinf(predictions).any() or (predictions[finite] < 0).any():
            raise ValueError(f'{group}: empty, infinite, or negative predictions; inspect raw windows')
        atomic_write(paths[0], lambda handle: np.save(handle, predictions))
        atomic_write(paths[1], lambda handle: handle.write(('\n'.join(names) + '\n').encode()))
        metadata = {'run_name': run, 'group': group, 'shape': list(predictions.shape),
                    'input_semantics': 'raw_none', 'config': str(Path(args.config).resolve()),
                    'checkpoint': str(checkpoint.resolve()),
                    'test_bedpe': config['data']['test_bedpe'],
                    'biosamples': names, 'prediction_layout': 'biosample_x_test_gene_in_bedpe_order',
                    'n_finite': int(finite.sum()), 'n_nan': int(np.isnan(predictions).sum()),
                    'dropped_hic_windows': loader.dataset.dropped}
        atomic_write(paths[2], lambda handle: handle.write((json.dumps(metadata, indent=2) + '\n').encode()))
        print(f'Saved {paths[0]}: {predictions.shape}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
