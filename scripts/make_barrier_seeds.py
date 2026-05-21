"""
Convert my_barrier_train.csv to li_barrier_seeds.extxyz.
Filters structures with 14-30 atoms (avoid OOD for MatterGen).
Usage: python scripts/make_barrier_seeds.py
"""
import os
import pandas as pd
import ase.io
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

CSV_PATH = os.environ.get('BARRIER_CSV', 'my_barrier_train.csv')
OUT_PATH = 'li_barrier_seeds.extxyz'
MIN_ATOMS = 14
MAX_ATOMS = 30

df = pd.read_csv(CSV_PATH)
print(f"Total entries: {len(df)}")

atoms_list = []
skipped = 0
for _, row in df.iterrows():
    try:
        struc = Structure.from_str(row['cif'], fmt='cif')
        n = len(struc)
        if n < MIN_ATOMS or n > MAX_ATOMS:
            skipped += 1
            continue
        atoms = AseAtomsAdaptor.get_atoms(struc)
        atoms.info['material_id'] = row['material_id']
        if 'li_migration_barrier_min_bulk' in row:
            atoms.info['li_barrier'] = row['li_migration_barrier_min_bulk']
        atoms_list.append(atoms)
    except Exception as e:
        print(f"  skip {row['material_id']}: {e}")
        skipped += 1

ase.io.write(OUT_PATH, atoms_list, format='extxyz')
print(f"Written {len(atoms_list)} structures to {OUT_PATH} ({skipped} skipped)")
print(f"Atom count range: {min(len(a) for a in atoms_list)}-{max(len(a) for a in atoms_list)}")
