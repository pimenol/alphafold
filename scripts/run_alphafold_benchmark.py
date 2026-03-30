#!/usr/bin/env python3
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run full AlphaFold2 (run_alphafold.py) on proteins listed in a benchmark CSV.

Path layout for --data_dir matches docker/run_docker.py (official download tree).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _database_paths(data_dir: Path, db_preset: str, model_preset: str) -> List[Tuple[str, Path]]:
    """Return (flag_name, path) pairs for run_alphafold.py."""
    paths: List[Tuple[str, Path]] = [
        ('uniref90_database_path', data_dir / 'uniref90' / 'uniref90.fasta'),
        (
            'mgnify_database_path',
            data_dir / 'mgnify' / 'mgy_clusters_2022_05.fa',
        ),
        ('template_mmcif_dir', data_dir / 'pdb_mmcif' / 'mmcif_files'),
        ('obsolete_pdbs_path', data_dir / 'pdb_mmcif' / 'obsolete.dat'),
    ]
    if 'multimer' in model_preset:
        paths.append(('uniprot_database_path', data_dir / 'uniprot' / 'uniprot.fasta'))
        paths.append(
            ('pdb_seqres_database_path', data_dir / 'pdb_seqres' / 'pdb_seqres.txt')
        )
    else:
        paths.append(('pdb70_database_path', data_dir / 'pdb70' / 'pdb70'))

    if db_preset == 'reduced_dbs':
        paths.append(
            (
                'small_bfd_database_path',
                data_dir / 'small_bfd' / 'bfd-first_non_consensus_sequences.fasta',
            )
        )
    else:
        paths.append(
            (
                'bfd_database_path',
                data_dir / 'bfd' / 'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
            )
        )
        paths.append(
            ('uniref30_database_path', data_dir / 'uniref30' / 'UniRef30_2021_03')
        )
    return paths


def _parse_benchmark_csv(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_fasta(path: Path, protein_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(f'>{protein_id}\n{sequence.strip()}\n')


def _build_command(
    run_alphafold: Path,
    fasta_path: Path,
    output_base: Path,
    data_dir: Path,
    db_preset: str,
    model_preset: str,
    max_template_date: str,
    use_precomputed_msas: bool,
    use_gpu_relax: bool,
    random_seed: Optional[int],
    passthrough: Sequence[str],
) -> List[str]:
    cmd: List[str] = [sys.executable, str(run_alphafold)]
    cmd.append(f'--fasta_paths={fasta_path}')
    cmd.append(f'--output_dir={output_base}')
    cmd.append(f'--data_dir={data_dir}')

    for name, p in _database_paths(data_dir, db_preset, model_preset):
        cmd.append(f'--{name}={p}')

    cmd.append(f'--max_template_date={max_template_date}')
    cmd.append(f'--db_preset={db_preset}')
    cmd.append(f'--model_preset={model_preset}')
    cmd.append(f'--use_precomputed_msas={str(use_precomputed_msas).lower()}')
    cmd.append(f'--use_gpu_relax={str(use_gpu_relax).lower()}')
    if random_seed is not None:
        cmd.append(f'--random_seed={random_seed}')
    cmd.extend(passthrough)
    return cmd


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_csv = (
        Path('/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh')
        / 'data' / 'benchmark' / 'summary.csv'
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--benchmark_csv',
        type=Path,
        default=default_csv if default_csv.is_file() else None,
        help='CSV with columns id, sequence (and optional length).',
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        required=True,
        help='AlphaFold data root (params + genetic DBs + pdb_mmcif), same as Docker.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Base output directory; each protein is written to <output_dir>/<id>/.',
    )
    parser.add_argument(
        '--fasta_dir',
        type=Path,
        default=None,
        help='Where to write per-protein FASTAs (default: <output_dir>/fastas).',
    )
    parser.add_argument(
        '--db_preset',
        choices=('full_dbs', 'reduced_dbs'),
        default='full_dbs',
    )
    parser.add_argument(
        '--model_preset',
        choices=(
            'monomer',
            'monomer_casp14',
            'monomer_ptm',
            'multimer',
        ),
        default='monomer_ptm',
    )
    parser.add_argument(
        '--max_template_date',
        default='2022-01-01',
        help='ISO date; use a recent date if you want more templates.',
    )
    parser.add_argument('--use_precomputed_msas', action='store_true')
    parser.add_argument('--no_gpu_relax', action='store_true')
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument(
        '--protein_ids',
        default=None,
        help='Comma-separated subset of IDs to run (optional).',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip if <output_dir>/<id>/ranking_debug.json already exists.',
    )
    args, passthrough = parser.parse_known_args()

    if args.benchmark_csv is None:
        print('Provide --benchmark_csv or place summary.csv at the default path.', file=sys.stderr)
        return 1

    csv_path = args.benchmark_csv.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    fasta_dir = (args.fasta_dir or output_dir / 'fastas').resolve()
    run_alphafold = repo_root / 'run_alphafold.py'

    if not run_alphafold.is_file():
        print(f'Missing {run_alphafold}', file=sys.stderr)
        return 1

    rows = _parse_benchmark_csv(csv_path)
    if args.protein_ids:
        wanted = set(args.protein_ids.split(','))
        rows = [r for r in rows if r['id'] in wanted]

    end = args.end_idx if args.end_idx is not None else len(rows)
    subset = rows[args.start_idx : end]

    use_gpu_relax = not args.no_gpu_relax
    failures = []

    for i, row in enumerate(subset):
        pid = row['id'].strip()
        seq = row['sequence'].strip()
        out_sub = output_dir / pid
        if args.skip_existing and (out_sub / 'ranking_debug.json').is_file():
            print(f'[{args.start_idx + i + 1}] {pid} skip (exists)')
            continue

        fasta_path = fasta_dir / f'{pid}.fasta'
        _write_fasta(fasta_path, pid, seq)

        cmd = _build_command(
            run_alphafold=run_alphafold,
            fasta_path=fasta_path,
            output_base=output_dir,
            data_dir=data_dir,
            db_preset=args.db_preset,
            model_preset=args.model_preset,
            max_template_date=args.max_template_date,
            use_precomputed_msas=args.use_precomputed_msas,
            use_gpu_relax=use_gpu_relax,
            random_seed=args.random_seed,
            passthrough=passthrough,
        )

        global_idx = args.start_idx + i + 1
        print(f'[{global_idx}] {pid} ({len(seq)} aa) ...', flush=True)
        env = os.environ.copy()
        env.setdefault('TF_FORCE_UNIFIED_MEMORY', '1')
        env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '4.0')

        rc = subprocess.call(cmd, cwd=str(repo_root), env=env)
        if rc != 0:
            print(f'FAILED {pid} rc={rc}', file=sys.stderr)
            failures.append(pid)

    if failures:
        print(f'Failed ({len(failures)}): {failures}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
