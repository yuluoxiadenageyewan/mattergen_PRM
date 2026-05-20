import os
import time
import logging
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig

from pipeline.mat_invent import MatInvent
from memory.replay_buffer import ReplayBuffer
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.eval_utils import load_structures


def _structure_to_chemgraph(struc):
    pos = torch.tensor(struc.frac_coords, dtype=torch.float)
    cell = torch.tensor(struc.lattice.matrix, dtype=torch.float).unsqueeze(0)
    atomic_numbers = torch.tensor([s.specie.Z for s in struc], dtype=torch.long)
    return ChemGraph(
        pos=pos, cell=cell, atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(struc)]),
    )


class DPOSolidElectrolyte(MatInvent):
    """Diffusion-DPO finetuning for solid electrolyte generation.

    Extends MatInvent with:
    - DPO loss replacing policy gradient
    - Dual replay buffer (winner + loser) for richer contrast signal
    - Optional warm-start seed injection from a known Li structure file
    """

    def __init__(
        self,
        dpo_beta: float = 0.1,
        loser_buffer_size: int = 50,
        seed_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dpo_beta = dpo_beta
        self.seed_path = seed_path
        self.loser_replay = ReplayBuffer(
            buffer_size=loser_buffer_size,
            sample_size=self.replay.sample_size if self.replay else 5,
            reward_cutoff=-999,
            priority_beta=0.6,
        )

    def run_rl(self):
        if self.seed_path:
            self._inject_seeds()
        super().run_rl()

    def _inject_seeds(self):
        strucs = list(load_structures(Path(self.seed_path)))
        if not strucs:
            logging.warning(f'No structures found in seed_path={self.seed_path}')
            return
        rewards, _, failed_mask = self.reward.scoring((strucs, self.seed_path), label='seed')
        valid = ~failed_mask
        strucs = [s for s, v in zip(strucs, valid) if v]
        rewards = rewards[valid].astype(float)
        if len(strucs) == 0:
            return
        seed_data = [_structure_to_chemgraph(s) for s in strucs]
        median = np.median(rewards)
        w = rewards >= median
        if w.any() and self.replay:
            self.replay.extend(
                [seed_data[i] for i, v in enumerate(w) if v],
                [strucs[i] for i, v in enumerate(w) if v],
                rewards[w],
            )
        if (~w).any():
            self.loser_replay.extend(
                [seed_data[i] for i, v in enumerate(w) if not v],
                [strucs[i] for i, v in enumerate(w) if not v],
                rewards[~w],
            )
        logging.info(f'Seed injected: {w.sum()} winners, {(~w).sum()} losers')

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')
        start_time = time.time()

        logging.info('SAMPLE:')
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        logging.info('SCORE:')
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list, sample_struc, xyz_path, f'step_{self.step:0>4d}',
        )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})
        log_dict.update(sample_metrics)

        if len(rewards) > 0:
            batch_mean = float(rewards.mean())
            batch_std = float(rewards.std()) + 1e-8
            self.ema_baseline = (
                self.ema_decay * self.ema_baseline + (1 - self.ema_decay) * batch_mean
            )
            self.ema_std = (
                self.ema_decay * self.ema_std + (1 - self.ema_decay) * batch_std
            )

        self.ltm.extend(sample_struc, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        logging.info(
            f'{len(self.ltm)} crystals generated so far, '
            f'{len(self.ltm.unique_comps)} unique components.'
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update({
            'crystal_num': len(self.ltm),
            'unique_comps': len(self.ltm.unique_comps),
            'burden': metrics[0],
            'div_ratio': metrics[1],
            'cost': self.cost,
            'ema_baseline': self.ema_baseline,
            'sigma': self.sigma,
            'residual_alpha': self._get_residual_alpha(),
        })
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        # collect losers into loser buffer (valid structures with low reward)
        if len(rewards) > 0:
            median_r = np.median(rewards)
            loser_mask = rewards < median_r
            loser_data = [d for d, m in zip(sample_list, loser_mask) if m]
            loser_strucs = [s for s, m in zip(sample_struc, loser_mask) if m]
            if loser_data:
                self.loser_replay.extend(loser_data, loser_strucs, rewards[loser_mask])
                logging.info(f'loser buffer size={len(self.loser_replay)}')

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_sample = [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f'Diversity filter: tol_n={tol_n}, buff_n={buff_n}')

        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        is_weights = None
        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay, is_weights = self.replay.sample(
                baseline=self.ema_baseline,
            )
            ft_data = sample_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            self.replay.extend(sample_topk, strucs_topk, reward_topk)
            logging.info(f'replay buffer size={len(self.replay)}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        logging.info('FINETUNE:')
        baseline = self.ltm.get_baseline(self.step)
        baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline, is_weights=is_weights)

        end_time = time.time()
        logging.info(f'*****   LOOP {self.step} FINISH   *****')
        logging.info(f'Total time taken: {(end_time - start_time) / 60:.2f} min.\n\n')

    def ft_step(self, data_list, rewards, baseline, is_weights=None):
        torch.set_float32_matmul_precision("high")
        cfg = self.finetune_cfg

        n = len(data_list)
        if n < 2:
            logging.warning("DPO ft_step: need at least 2 samples, skipping.")
            return

        sort_idx = np.argsort(rewards)
        half = n // 2
        winner_idx = sort_idx[half:][::-1]
        loser_idx  = sort_idx[:half]
        pair_len = min(len(winner_idx), len(loser_idx))
        winner_idx = winner_idx[:pair_len]
        loser_idx  = loser_idx[:pair_len]

        margin = rewards.std() * 0.1
        valid = np.abs(rewards[winner_idx] - rewards[loser_idx]) >= margin
        winner_idx = winner_idx[valid]
        loser_idx  = loser_idx[:len(winner_idx)]
        pair_len = len(winner_idx)
        if pair_len == 0:
            logging.warning("DPO ft_step: all pairs below margin, skipping.")
            return

        data_w = [data_list[i] for i in winner_idx]
        data_l = [data_list[i] for i in loser_idx]

        # augment with historical pairs from dual buffer
        if self.replay and len(self.replay) >= 2 and len(self.loser_replay) >= 2:
            hist_w, _, _ = self.replay.sample(baseline=self.ema_baseline)
            hist_l, _, _ = self.loser_replay.sample(baseline=0.0)
            hist_pair_n = min(len(hist_w), len(hist_l))
            if hist_pair_n > 0:
                w_perm = np.random.permutation(len(hist_w))[:hist_pair_n]
                l_perm = np.random.permutation(len(hist_l))[:hist_pair_n]
                data_w = data_w + [hist_w[i] for i in w_perm]
                data_l = data_l + [hist_l[i] for i in l_perm]
                pair_len = len(data_w)
                logging.info(f'DPO: {hist_pair_n} historical pairs added, total={pair_len}')

        loader_w = self.model_suite.get_dataloader(data_w, rewards=None, batch_size=pair_len, shuffle=False)
        loader_l = self.model_suite.get_dataloader(data_l, rewards=None, batch_size=pair_len, shuffle=False)

        optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr)
        accum_steps = cfg.accum_steps

        num_sampled_t = cfg.get('num_sampled_timesteps', cfg.timesteps)
        scale_factor = cfg.timesteps / num_sampled_t if num_sampled_t < cfg.timesteps else 1.0

        for epoch in range(cfg.epochs):
            self.agent.train()
            loss_all = 0.0

            for (batch_w, batch_l) in zip(loader_w, loader_l):
                batch_w = batch_w.to(self.device)
                batch_l = batch_l.to(self.device)
                optimizer.zero_grad()

                sampled_ts = torch.randperm(cfg.timesteps)[:num_sampled_t].sort().values

                for step_idx, t in enumerate(sampled_ts):
                    t = t.item()
                    noised_w = self.agent.add_noise(batch_w, t)
                    noised_l = self.agent.add_noise(batch_l, t)

                    loss = self.agent.calc_dpo_loss(
                        noised_w, noised_l, self.prior, beta=self.dpo_beta,
                    )
                    loss = loss * scale_factor / accum_steps
                    loss.backward()

                    if (step_idx + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    loss_all += loss.item() * accum_steps / scale_factor

                if (step_idx + 1) % accum_steps != 0:
                    optimizer.step()

            logging.info(f'DPO Epoch {epoch}: loss={loss_all / max(pair_len, 1):.4f}')
