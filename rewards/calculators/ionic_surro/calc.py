import sys
import os
import tempfile
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

sys.path.insert(0, os.path.abspath('D:/ionic_surro'))
from scorer import score_structures

from rewards.calculators.base import Calculator


class IonicSurroCalculator(Calculator):
    def __init__(self, root_dir: str, task: str = 'ionic_conductivity') -> None:
        super().__init__(root_dir, task)

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp',
    ) -> np.ndarray:
        struc_list = samples[0]
        scores = []
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_paths = []
            for i, struc in enumerate(struc_list):
                cif_path = os.path.join(tmpdir, f'{label}_{i}.cif')
                CifWriter(struc).write_file(cif_path)
                cif_paths.append(cif_path)

            results = score_structures(cif_paths)
            score_map = dict(zip(results['cif_path'], results['preference_score']))
            for p in cif_paths:
                scores.append(score_map.get(p, np.nan))

        out = np.array(scores, dtype=float)
        np.savetxt(os.path.join(self.root_dir, f'{label}.txt'), out, fmt='%.8f')
        return out
