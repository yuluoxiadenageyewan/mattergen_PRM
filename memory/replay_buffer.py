"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from pymatgen.core.structure import Structure


class ReplayBuffer:
    """
    Replay buffer with prioritized sampling for sparse reward settings.
    Stores the top K highest reward crystals generated so far.
        1. Crystals (data, pymatgen.Structure, compositions)
        2. Reward
    """

    def __init__(
        self,
        buffer_size: int = 100,
        sample_size: int = 8,
        reward_cutoff: float = 0.0,
        priority_beta: float = 0.6,
    ) -> None:
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.reward_cutoff = reward_cutoff
        self.priority_beta = priority_beta
        # Stores the top N highest reward crystal generated so far
        self.buffer = pd.DataFrame(
            columns=["data", "struc", "comp", "ele_comb", "reward"]
        )

    def extend(
        self,
        data: list,
        strucs: List[Structure],
        rewards: np.ndarray[float],
    ) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = []
        for s in strucs:
            elements = set(str(e) for e in s.species)
            comb = tuple(sorted(elements))
            ele_comb.append(comb)

        df_sam = pd.DataFrame.from_dict({
            "data": data,
            "struc": strucs,
            "comp": comps,
            "ele_comb": ele_comb,
            "reward": rewards
        })
        if len(self.buffer) > 0:
            df_all = pd.concat([self.buffer, df_sam])
        else:
            df_all = df_sam
        unique_df = self.deduplicate(df_all)
        sorted_df = unique_df.sort_values("reward", ascending=False)
        self.buffer = sorted_df.head(self.buffer_size)
        # reward cutoff
        self.buffer = self.buffer.loc[self.buffer["reward"] > self.reward_cutoff]

    def deduplicate(self, df: pd.DataFrame, method="composition") -> pd.DataFrame:
        """
        Removes duplicate crystals based on different methods like composition,
        StructureMatcher, symmetry (crystal system, space group, etc.)
        Keep only non-zero rewards crystals.
        """
        _df = df.sort_values("reward", ascending=False)
        if method == "composition":
            unique_df = _df.drop_duplicates(subset=["comp"])
        elif method == "element_comb":
            unique_df = _df.drop_duplicates(subset=["ele_comb"])

        return unique_df

    def sample(
        self, baseline: float = 0.0,
    ) -> Tuple[List[Data], np.ndarray, np.ndarray]:
        """Prioritized experience replay sampling.

        Samples with probability proportional to |reward - baseline|, so
        high-reward experiences are replayed more frequently in sparse reward
        settings. Returns importance sampling (IS) weights to correct for the
        non-uniform sampling bias.

        Args:
            baseline: EMA baseline reward for priority computation.

        Returns:
            data: List of sampled ChemGraph data.
            rewards: Array of rewards for sampled data.
            is_weights: Importance sampling weights (normalized).
        """
        sample_size = min(len(self.buffer), self.sample_size)
        if sample_size == 0:
            return [], np.array([]), np.array([])

        # Priority = |reward - baseline| + epsilon (avoid zero priority)
        rewards_arr = self.buffer["reward"].values.astype(float)
        priorities = np.abs(rewards_arr - baseline) + 1e-4
        probs = priorities / priorities.sum()

        # Prioritized sampling without replacement
        indices = np.random.choice(
            len(self.buffer), size=sample_size, replace=False, p=probs,
        )
        sampled = self.buffer.iloc[indices]

        # Importance sampling weights for bias correction
        N = len(self.buffer)
        is_weights = (1.0 / (N * probs[indices])) ** self.priority_beta
        is_weights = is_weights / is_weights.max()  # normalize to [0, 1]

        data = sampled["data"].values.tolist()
        rewards = sampled["reward"].values.astype(float)
        return data, rewards, is_weights

    def memory_purge(self, strucs: List[Structure]) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        self.buffer = self.buffer[~self.buffer["comp"].isin(comps)]

    def __len__(self) -> int:
        return len(self.buffer)
