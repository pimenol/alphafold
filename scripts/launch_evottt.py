#!/usr/bin/env python3
"""Launch EvoTTT experiments from YAML config files.

Usage:
  # Submit as SLURM job (default):
  python scripts/launch_evottt.py configs/evottt/default.yaml

  # Submit multiple configs at once:
  python scripts/launch_evottt.py configs/evottt/*.yaml

  # Override any hparam from CLI:
  python scripts/launch_evottt.py configs/evottt/default.yaml --ttt_lr 1e-3 --ttt_steps 100

  # Dry-run (print the sbatch script without submitting):
  python scripts/launch_evottt.py configs/evottt/default.yaml --dry_run

  # Run locally (no SLURM):
  python scripts/launch_evottt.py configs/evottt/default.yaml --local
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_SH = REPO_ROOT / 'jobs' / 'evottt' / 'run_evottt.sh'

# Keys that map to env vars consumed by run_evottt.sh
HPARAM_ENV = {
    'ttt_steps':      'TTT_STEPS',
    'ttt_lr':         'TTT_LR',
    'lora_rank':      'LORA_RANK',
    'last_n_blocks':  'LAST_N_BLOCKS',
    'lora_alpha':     'LORA_ALPHA',
    'grad_clip':      'GRAD_CLIP',
    'mask_fraction':  'MASK_FRACTION',
    'start_idx':      'START_IDX',
    'end_idx':        'END_IDX',
    'protein_ids':    'PROTEIN_IDS',
    'data_dir':       'DATA_DIR',
    'msa_dir':        'MSA_DIR',
    'benchmark_csv':  'BENCHMARK_CSV',
    'output_dir':     'OUTPUT_DIR',
}

# Keys forwarded as extra CLI args to the python script
EXTRA_CLI = {
    'model_name':     '--model_name',
    'seed':           '--seed',
    'mask_fraction':  '--mask_fraction',
    'grad_clip':      '--grad_clip',
}

SLURM_DEFAULTS = {
    'time': '48:00:00',
    'gpus': 1,
    'cpus_per_task': 8,
    'partition': 'qgpu',
    'account': 'OPEN-35-8',
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Parse --key value pairs from leftover CLI args into the config dict."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip('-')
        if i + 1 < len(overrides) and not overrides[i + 1].startswith('--'):
            val = overrides[i + 1]
            # Try numeric conversion
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            cfg[key] = val
            i += 2
        else:
            cfg[key] = True
            i += 1
    return cfg


def build_sbatch_script(cfg: dict, config_name: str) -> str:
    slurm = {**SLURM_DEFAULTS, **(cfg.get('slurm') or {})}
    job_name = f'evottt_{config_name}'
    log_dir = REPO_ROOT / 'jobs' / 'evottt'

    # Build output subdir that encodes key hparams
    output_dir = cfg.get('output_dir')
    if not output_dir:
        tag = (f"s{cfg.get('ttt_steps', 50)}"
               f"_lr{cfg.get('ttt_lr', 3e-4)}"
               f"_r{cfg.get('lora_rank', 4)}"
               f"_b{cfg.get('last_n_blocks', 8)}"
               f"_a{cfg.get('lora_alpha', 1.0)}")
        output_dir = str(REPO_ROOT / 'evottt_outputs' / tag)

    # Environment variable exports
    env_lines = []
    for key, env_var in HPARAM_ENV.items():
        if key in cfg and key != 'output_dir':
            env_lines.append(f'export {env_var}="{cfg[key]}"')
    env_lines.append(f'export OUTPUT_DIR="{output_dir}"')

    # Extra CLI flags passed through "$@" in run_evottt.sh
    extra_args = []
    for key, flag in EXTRA_CLI.items():
        if key in cfg:
            extra_args.extend([flag, str(cfg[key])])
    if cfg.get('skip_existing'):
        extra_args.append('--skip_existing')
    if cfg.get('skip_baseline'):
        extra_args.append('--skip_baseline')

    extra_str = ' '.join(extra_args)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --account={slurm['account']}
        #SBATCH --partition={slurm['partition']}
        #SBATCH --nodes=1
        #SBATCH --gpus={slurm['gpus']}
        #SBATCH --cpus-per-task={slurm['cpus_per_task']}
        #SBATCH --time={slurm['time']}
        #SBATCH --output={log_dir}/{job_name}_%A.out
        #SBATCH --error={log_dir}/{job_name}_%A.err

        # Auto-generated from config: {config_name}
        {chr(10).join(env_lines)}

        exec bash {TEMPLATE_SH} {extra_str}
    """)
    return script


def submit_slurm(script: str, dry_run: bool = False) -> None:
    if dry_run:
        print('--- sbatch script (dry run) ---')
        print(script)
        print('-------------------------------')
        return

    result = subprocess.run(
        ['sbatch'],
        input=script,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_local(cfg: dict) -> None:
    """Run the benchmark script directly (no SLURM)."""
    cmd = [sys.executable, '-u', str(REPO_ROOT / 'scripts' / 'run_evottt_benchmark.py')]

    output_dir = cfg.get('output_dir')
    if not output_dir:
        tag = (f"s{cfg.get('ttt_steps', 50)}"
               f"_lr{cfg.get('ttt_lr', 3e-4)}"
               f"_r{cfg.get('lora_rank', 4)}"
               f"_b{cfg.get('last_n_blocks', 8)}"
               f"_a{cfg.get('lora_alpha', 1.0)}")
        output_dir = str(REPO_ROOT / 'evottt_outputs' / tag)

    # Map config keys to CLI args
    arg_map = {
        'benchmark_csv': '--benchmark_csv',
        'msa_dir': '--msa_dir',
        'data_dir': '--data_dir',
        'model_name': '--model_name',
        'ttt_steps': '--ttt_steps',
        'ttt_lr': '--ttt_lr',
        'lora_rank': '--lora_rank',
        'last_n_blocks': '--last_n_blocks',
        'lora_alpha': '--lora_alpha',
        'grad_clip': '--grad_clip',
        'mask_fraction': '--mask_fraction',
        'eval_interval': '--eval_interval',
        'seed': '--seed',
        'start_idx': '--start_idx',
        'end_idx': '--end_idx',
        'protein_ids': '--protein_ids',
    }

    cmd.extend(['--output_dir', output_dir])
    for key, flag in arg_map.items():
        if key in cfg:
            cmd.extend([flag, str(cfg[key])])
    if cfg.get('skip_existing'):
        cmd.append('--skip_existing')
    if cfg.get('skip_baseline'):
        cmd.append('--skip_baseline')

    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Launch EvoTTT experiments from YAML configs.',
        epilog='Any extra --key value args are merged into the config.',
    )
    parser.add_argument('configs', nargs='+', type=Path,
                        help='YAML config file(s)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print sbatch script without submitting')
    parser.add_argument('--local', action='store_true',
                        help='Run locally instead of submitting to SLURM')
    args, extra = parser.parse_known_args()

    for config_path in args.configs:
        cfg = load_config(config_path)
        cfg = merge_cli_overrides(cfg, extra)
        config_name = config_path.stem

        print(f'Config: {config_path.name}')
        for k in ['ttt_steps', 'ttt_lr', 'lora_rank', 'last_n_blocks',
                   'lora_alpha', 'mask_fraction']:
            if k in cfg:
                print(f'  {k}: {cfg[k]}')

        if args.local:
            run_local(cfg)
        else:
            script = build_sbatch_script(cfg, config_name)
            submit_slurm(script, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
