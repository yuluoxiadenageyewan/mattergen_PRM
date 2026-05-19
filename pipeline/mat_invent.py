import os
import time
import logging
from typing import Dict
import numpy as np
import torch
from omegaconf import DictConfig

from pipeline.base import ReinL
from pipeline.filters import OptEval, invalid_filter, CompositionFilter
from pipeline.utils.save import save_structures
from pipeline.utils.logger import Logger
from rewards.reward import Reward
from models.suite.base import ModelSuite


class MatInvent(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        # diversity filter
        self.div_filter = div_filter
        self.df_args = df_args

        # composition filter (required/excluded elements)
        if self.sample_cfg.get('comp_filter'):
            self.comp_filter = CompositionFilter(**self.sample_cfg.comp_filter)
        else:
            self.comp_filter = None

        if 'filter' not in self.sample_cfg:
            self.opt_eval = OptEval()

        # --- EMA baseline for advantage normalization ---
        self.ema_baseline = 0.0
        self.ema_std = 1.0
        self.ema_decay = self.finetune_cfg.get('ema_decay', 0.95)

        # --- Adaptive KL coefficient ---
        self.sigma = self.finetune_cfg.get('sigma', 0.025)
        self.target_kl = self.finetune_cfg.get('target_kl', 0.1)
        self.sigma_lr = self.finetune_cfg.get('sigma_lr', 0.1)

        # --- Residual RL: alpha annealing ---
        self.residual_alpha_init = self.finetune_cfg.get('residual_alpha_init', 1.0)
        self.residual_warmup_steps = self.finetune_cfg.get('residual_warmup_steps', 60)

        self.load_model()

    def load_model(self):
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.requires_grad = True
        # Freeze the parameter of prior (pretrained) model
        for param in self.prior.parameters():
            param.requires_grad = False
        self.agent.to(self.device)
        self.prior.to(self.device)

    def _get_residual_alpha(self):
        """Compute current residual alpha via linear annealing."""
        if self.residual_warmup_steps <= 0:
            return 1.0
        alpha = self.residual_alpha_init + (
            (1.0 - self.residual_alpha_init) * self.step / self.residual_warmup_steps
        )
        return min(alpha, 1.0)

    def sample_step(self):
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent, **self.sample_cfg,
        )
        # Filter invalid samples
        sample_data, sample_struc = invalid_filter(sample_data, sample_struc)

        # Composition constraint filter (required/excluded elements)
        if self.comp_filter is not None:
            sample_data, sample_struc = self.comp_filter(sample_data, sample_struc)

        # save all generated valid structures
        valid_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_valid.extxyz',
        )

        # MLIP relaxation
        if self.sample_cfg.get('mlip_opt'):
            mlip_opt = self.sample_cfg.mlip_opt
            sample_struc, energies = mlip_opt(sample_struc, valid_xyz_path)
        else:
            energies = None

        # Filter bad samples by selected metrics
        if self.sample_cfg.get('filter'):
            filter = self.sample_cfg.filter
            sample_data, sample_struc, metrics = filter(
                sample_data, sample_struc, energies,
            )
            logging.info(f'Number of filtered samples: {len(sample_struc)}')
        else:
            # metrics, _ = self.opt_eval(sample_struc, energies)
            metrics = {}

        log_str = [f'{k}: {v:.6f}' for k, v in metrics.items()]
        logging.info(', '.join(log_str))

        # max sample size to score/reward
        if self.sample_cfg.get('max_num'):
            max_num = self.sample_cfg.max_num
            if len(sample_struc) > max_num:
                sample_data = sample_data[:max_num]
                sample_struc = sample_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_eval.extxyz',
        )

        return sample_data, sample_struc, eval_xyz_path, metrics

    def ft_step(self, data_list, rewards, baseline, is_weights=None):
        # Tensor Core acceleration for new GPUs (Ampere, Hopper, etc)
        torch.set_float32_matmul_precision("high")
        cfg = self.finetune_cfg
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=rewards,
            batch_size=len(data_list),
        )

        optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr)
        accum_steps = cfg.accum_steps

        # --- Timestep importance sampling ---
        num_sampled_t = cfg.get('num_sampled_timesteps', cfg.timesteps)
        use_timestep_sampling = num_sampled_t < cfg.timesteps
        if use_timestep_sampling:
            scale_factor = cfg.timesteps / num_sampled_t
        else:
            scale_factor = 1.0

        # --- Residual alpha for this RL step ---
        alpha = self._get_residual_alpha()
        use_residual = alpha < 1.0
        if use_residual:
            logging.info(f'Residual RL: alpha={alpha:.4f}')

        for epoch in range(cfg.epochs):
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0., 0., 0.
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss, loss_diff, loss_kl = 0., 0., 0.

                # Sample random timesteps or use all
                if use_timestep_sampling:
                    sampled_ts = torch.randperm(cfg.timesteps)[:num_sampled_t].sort().values
                else:
                    sampled_ts = range(cfg.timesteps)

                for step_idx, t in enumerate(sampled_ts):
                    if isinstance(t, torch.Tensor):
                        t = t.item()

                    noised_input = self.agent.add_noise(batch, t)

                    # --- Residual RL or standard forward ---
                    if use_residual:
                        sample_loss, agent_pred, prior_pred = (
                            self.agent.calc_residual_sample_loss(
                                noised_input, self.prior, alpha=alpha,
                            )
                        )
                    else:
                        sample_loss, agent_pred = self.agent.calc_sample_loss(
                            noised_input,
                        )
                        _, prior_pred = self.prior.calc_sample_loss(noised_input)

                    # --- Advantage normalization (EMA baseline) ---
                    adv = (batch.reward - self.ema_baseline) / (self.ema_std + 1e-8)
                    adv = torch.clamp(adv, -5.0, 5.0)

                    _loss_diff = adv * sample_loss

                    # --- KL regularization ---
                    kl_term = self.agent.calc_kl_reg(agent_pred, prior_pred, batch)
                    _loss_kl = kl_term * (1.1 - batch.reward)

                    # --- Combine with adaptive sigma and timestep scaling ---
                    _loss = (_loss_diff + _loss_kl * self.sigma).mean()
                    _loss = _loss * scale_factor / accum_steps
                    _loss.backward()

                    if (step_idx + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    loss += _loss.item() * accum_steps / scale_factor
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                num_ts = num_sampled_t if use_timestep_sampling else cfg.timesteps
                loss_diff = loss_diff / num_ts
                loss_kl = loss_kl / num_ts
                loss = loss / num_ts

                if (step_idx + 1) % accum_steps != 0:
                    optimizer.step()

                loss_all += loss * batch.num_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            # --- Adaptive KL coefficient ---
            mean_kl = loss_kl_all / max(len(data_list), 1)
            if mean_kl > self.target_kl * 1.5:
                self.sigma = min(self.sigma * (1 + self.sigma_lr), 1.0)
            elif mean_kl < self.target_kl * 0.5:
                self.sigma = max(self.sigma * (1 - self.sigma_lr), 0.001)

            loss_dict = {
                'loss': loss_all / len(data_list),
                'loss_diff': loss_diff_all / len(data_list),
                'loss_kl': loss_kl_all / len(data_list),
                'sigma': self.sigma,
                'alpha': alpha,
            }
            log_str = [f'{k}: {v:.4f}' for k, v in loss_dict.items()]
            logging.info(f'Epoch {epoch}: ' + ', '.join(log_str))

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')
        start_time = time.time()

        logging.info('SAMPLE:')
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info('SCORE:')
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list, sample_struc, xyz_path, f'step_{self.step:0>4d}',
        )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})
        log_dict.update(sample_metrics)

        # --- Update EMA baseline ---
        if len(rewards) > 0:
            batch_mean = float(rewards.mean())
            batch_std = float(rewards.std()) + 1e-8
            self.ema_baseline = (
                self.ema_decay * self.ema_baseline
                + (1 - self.ema_decay) * batch_mean
            )
            self.ema_std = (
                self.ema_decay * self.ema_std
                + (1 - self.ema_decay) * batch_std
            )

        # long-term memory
        self.ltm.extend(sample_struc, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        logging.info(
            f'{len(self.ltm)} crystals generated so far, ' +
            f'{len(self.ltm.unique_comps)} unique components.' +
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update(
            {
                'crystal_num': len(self.ltm),
                'unique_comps': len(self.ltm.unique_comps),
                'burden': metrics[0],
                'div_ratio': metrics[1],
                'cost': self.cost,
                'ema_baseline': self.ema_baseline,
                'sigma': self.sigma,
                'residual_alpha': self._get_residual_alpha(),
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_sample = [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f'Diversity filter: tol_n={tol_n}, buff_n={buff_n}')

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay (prioritized)
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
            logging.info(f'buffer reward mean={self.replay.buffer["reward"].values.mean()}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        # finetuning
        logging.info('FINETUNE:')
        baseline = self.ltm.get_baseline(self.step)
        baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline, is_weights=is_weights)

        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f'*****   LOOP {self.step} FINISH   *****')
        logging.info(f'Total time taken: {total_time:.2f} min.\n\n')

    def run_rl(self):
        logging.info('*****   RL START   *****')
        start_time = time.time()
        last_ckpt_dir = None

        try:
            for step in range(self.rl_epoch):
                self.step = step
                self.rl_step()
                if (step + 1) % 20 == 0:
                    ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                    self.model_suite.save_model(self.agent, ckpt_dir)
                    last_ckpt_dir = ckpt_dir
                    logging.info(f'Checkpoint saved: {ckpt_dir}')
                # legacy save_freq (keep for backward compat if save_freq != 20)
                elif (step + 1) % self.save_freq == 0:
                    ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                    self.model_suite.save_model(self.agent, ckpt_dir)
                    last_ckpt_dir = ckpt_dir
        except KeyboardInterrupt:
            logging.info('Early stop triggered.')
            if last_ckpt_dir is not None:
                logging.info(f'Last checkpoint: {last_ckpt_dir}')
            else:
                logging.info('No checkpoint saved yet; nothing to restore.')
            return

        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info('Total time taken: {} s.'.format(int(end_time - start_time)))
