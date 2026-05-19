import logging
from typing import List, Optional

from pymatgen.core.structure import Structure


class CompositionFilter:
    """Filter crystal structures by required and excluded elements.

    Args:
        required_elements: Elements that MUST be present in the structure.
            e.g., ["Li"] means every structure must contain Li.
        excluded_elements: Elements that must NOT be present in the structure.
            e.g., ["Sb"] means structures containing Sb are removed.
    """

    def __init__(
        self,
        required_elements: Optional[List[str]] = None,
        excluded_elements: Optional[List[str]] = None,
    ) -> None:
        self.required = set(required_elements or [])
        self.excluded = set(excluded_elements or [])

    def __call__(self, data_list: list, structures: List[Structure]):
        mask = []
        for s in structures:
            elements = set(str(e) for e in s.composition.elements)
            has_required = self.required.issubset(elements)
            no_excluded = len(elements & self.excluded) == 0
            mask.append(has_required and no_excluded)

        filtered_data = [x for x, m in zip(data_list, mask) if m]
        filtered_struc = [x for x, m in zip(structures, mask) if m]

        n_before = len(structures)
        n_after = len(filtered_struc)
        logging.info(
            f'CompositionFilter: {n_before} -> {n_after} '
            f'(required={self.required}, excluded={self.excluded})'
        )
        return filtered_data, filtered_struc
