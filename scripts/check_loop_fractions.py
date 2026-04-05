#!/usr/bin/env python3
"""Print loop fraction for each protein kept by the AF2 filter."""

import csv
import numpy as np
from pathlib import Path

from alphafold.common import protein
from alphafold.common.structure_detects import (
    _compute_phi_psi,
    _classify_secondary_structure,
)

PDB_DIR = Path('data/bfvd/AF2')
SUMMARY = Path('data/bfvd/summary.csv')

with open(SUMMARY) as f:
    rows = list(csv.DictReader(f))

print(f"{'ID':<18} {'len':>4} {'pLDDT':>7} {'%helix':>7} {'%sheet':>7} {'%loop':>7}  dynamic?")
print('-' * 80)

for row in rows:
    pid = row['id']
    pdb_path = PDB_DIR / f'{pid}.pdb'
    prot_obj = protein.from_pdb_string(pdb_path.read_text())

    plddt = np.mean(prot_obj.b_factors[:, 1])
    phi, psi = _compute_phi_psi(
        prot_obj.atom_positions, prot_obj.atom_mask, prot_obj.chain_index
    )
    ss = _classify_secondary_structure(phi, psi)
    n = len(ss)
    helix_pct = 100.0 * np.sum(ss == 0) / n
    sheet_pct = 100.0 * np.sum(ss == 1) / n
    loop_pct = 100.0 * np.sum(ss == 2) / n
    is_dyn = plddt < 50 and loop_pct > 50

    print(f'{pid:<18} {int(row["length"]):>4} {plddt:>7.2f} {helix_pct:>6.1f}% {sheet_pct:>6.1f}% {loop_pct:>6.1f}%  {"YES" if is_dyn else "no"}')
