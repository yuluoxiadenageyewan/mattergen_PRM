import logging
import torch
import numpy as np
from omegaconf import DictConfig

from pipeline.mat_invent import MatInvent


class DPOSolidElectrolyte(MatInvent):
    """Diffusion-DPO finetuning for solid electrolyte generation.

    Replaces the RL ft_step with pairwise DPO loss. No explicit reward values
    are used during finetuning — only winner/loser preference pairs derived
    from ionic conductivity scores.
    """

    def __init__(self, dpo_beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dpo_beta = dpo_beta

    def ft_step(self, data_list, rewards, baseline, is_weights=None):
        torch.set_float32_matmul_precision("high")
        cfg = self.finetune_cfg

        # Build winner/loser pairs by ranking on reward
        n = len(data_list)
        if n < 2:
            logging.warning("DPO ft_step: need at least 2 samples, skipping.")
            return

        sort_idx = np.argsort(rewards)
        # pair top half (winners) with bottom half (losers)
        half = n // 2
        winner_idx = sort_idx[half:][::-1]   # highest reward first
        loser_idx  = sort_idx[:half]          # lowest reward first
        pair_len = min(len(winner_idx), len(loser_idx))
        winner_idx = winner_idx[:pair_len]
        loser_idx  = loser_idx[:pair_len]

        data_w = [data_list[i] for i in winner_idx]
        data_l = [data_list[i] for i in loser_idx]

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
