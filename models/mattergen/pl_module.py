from typing import Optional, TypeVar
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch_scatter import scatter

from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.lightning_module import DiffusionLightningModule

from models.mattergen.loss import SampleLoss

T = TypeVar("T", bound=BatchedData)


class MatterGenModule(DiffusionLightningModule):
    """LightningModule for instantiating and training a MatterGen."""

    def __init__(
        self,
        diffusion_module,
        optimizer_partial=None,
        scheduler_partials=None,
    ):
        """_summary_

        Args:
            diffusion_module: The diffusion module to use.
            optimizer_partial: Used to instantiate optimizer.
            scheduler_partials: used to instantiate learning rate schedulers
        """
        super().__init__(
            diffusion_module=diffusion_module,
            optimizer_partial=optimizer_partial,
            scheduler_partials=scheduler_partials,
        )
        self.sample_loss_fn = SampleLoss()

    def ft_step(self, batch: T):
        batch = self.diffusion_module.pre_corruption_fn(batch)
        noisy_batch, t = self.diffusion_module._corrupt_batch(batch)
        score_model_output = self.diffusion_module.model(noisy_batch, t)

        loss, metrics = self.diffusion_module.loss_fn(
            multi_corruption=self.diffusion_module.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
        )
        assert loss.numel() == 1
        return loss, metrics

    def add_noise(self, batch: T, timestep: int):
        batch = self.diffusion_module.pre_corruption_fn(batch)
        max_t = self.diffusion_module.corruption.T
        device = self.diffusion_module._get_device(batch)
        N = 1000
        time_list = torch.linspace(
            max_t, 1 / N, N, device=device,
        )
        t = torch.full(
            (batch.get_batch_size(),),
            time_list[timestep],
            device=device,
        )
        noisy_batch = self.diffusion_module.corruption.sample_marginal(batch, t)
        return noisy_batch, batch , t

    def calc_sample_loss(self, noised_input):
        noisy_batch, batch , t = noised_input
        score_model_output = self.diffusion_module.model(noisy_batch, t)
        loss, metrics = self.sample_loss_fn(
            multi_corruption=self.diffusion_module.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
        )
        return loss, score_model_output

    def calc_residual_sample_loss(self, noised_input, prior_model, alpha=1.0):
        """Residual RL: compute denoising loss on the effective output
        ε_eff = ε_prior + α · (ε_agent - ε_prior).

        Based on Residual RL (Johannink et al., 2019): the agent learns a
        residual correction on top of the prior's prediction rather than the
        full denoising output. α anneals from small (safe, prior-dominated)
        to 1.0 (full agent control) during training.

        Args:
            noised_input: Tuple of (noisy_batch, batch, t) from add_noise().
            prior_model: The frozen prior MatterGenModule.
            alpha: Residual interpolation coefficient in [0, 1].

        Returns:
            loss: Per-sample denoising loss on the effective output.
            agent_pred: Raw agent model output (for KL computation).
            prior_pred: Raw prior model output (for KL computation).
        """
        noisy_batch, batch, t = noised_input

        # Agent forward pass (gradients flow through here)
        agent_pred = self.diffusion_module.model(noisy_batch, t)

        # Prior forward pass (no gradients)
        with torch.no_grad():
            prior_pred = prior_model.diffusion_module.model(noisy_batch, t)

        # Effective output: ε_prior + α · (ε_agent - ε_prior)
        effective_output = agent_pred.replace(
            pos=prior_pred["pos"] + alpha * (agent_pred["pos"] - prior_pred["pos"]),
            cell=prior_pred["cell"] + alpha * (agent_pred["cell"] - prior_pred["cell"]),
            atomic_numbers=prior_pred["atomic_numbers"] + alpha * (
                agent_pred["atomic_numbers"] - prior_pred["atomic_numbers"]
            ),
        )

        # Denoising loss on the effective (residual-blended) output
        loss, metrics = self.sample_loss_fn(
            multi_corruption=self.diffusion_module.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=effective_output,
            t=t,
        )
        return loss, agent_pred, prior_pred

    def calc_kl_reg(self, agent_pred, prior_pred, batch):
        (pred_x, pred_l, pred_t) = (
            agent_pred["pos"],
            agent_pred["cell"],
            agent_pred["atomic_numbers"],
        )
        (pred_x_p, pred_l_p, pred_t_p) = (
            prior_pred["pos"].detach(),
            prior_pred["cell"].detach(),
            prior_pred["atomic_numbers"].detach(),
        )
        batch_idx = batch.get_batch_idx("pos")

        kl_term0 = torch.pow((pred_l - pred_l_p), 2).mean(dim=(1,2))
        x_ap = torch.pow((pred_x - pred_x_p), 2).mean(dim=1)
        kl_term1 = scatter(x_ap, batch_idx, dim=0, reduce='mean')
        t_ap = torch.pow((pred_t - pred_t_p), 2).mean(dim=1)
        kl_term2 = scatter(t_ap, batch_idx, dim=0, reduce='mean')
        kl_term = kl_term0 + kl_term1 + kl_term2
        return kl_term

    def calc_dpo_loss(self, noised_input_w, noised_input_l, prior_model, beta=0.1):
        """Diffusion-DPO loss (Wallace et al., 2024).

        Uses SampleLoss (score-matching loss) as the implicit reward proxy, which
        correctly handles MatterGen's mixed prediction targets:
          - pos/cell: score_times_std  (not ε, not x_0)
          - atomic_numbers: logits     (D3PM categorical)

        L = -log σ( β · (loss_ref_w - loss_θ_w - loss_ref_l + loss_θ_l) )
        """
        noisy_w, batch_w, t_w = noised_input_w
        noisy_l, batch_l, t_l = noised_input_l

        agent_w = self.diffusion_module.model(noisy_w, t_w)
        agent_l = self.diffusion_module.model(noisy_l, t_l)
        with torch.no_grad():
            ref_w = prior_model.diffusion_module.model(noisy_w, t_w)
            ref_l = prior_model.diffusion_module.model(noisy_l, t_l)

        def _loss_per_sample(pred, noisy_batch, clean_batch, t):
            loss_vec, _ = self.sample_loss_fn(
                multi_corruption=self.diffusion_module.corruption,
                batch=clean_batch,
                noisy_batch=noisy_batch,
                score_model_output=pred,
                t=t,
            )
            return loss_vec  # shape: (batch_size,)

        loss_agent_w = _loss_per_sample(agent_w, noisy_w, batch_w, t_w)
        loss_agent_l = _loss_per_sample(agent_l, noisy_l, batch_l, t_l)
        with torch.no_grad():
            loss_ref_w = _loss_per_sample(ref_w, noisy_w, batch_w, t_w)
            loss_ref_l = _loss_per_sample(ref_l, noisy_l, batch_l, t_l)

        # implicit log-ratio: log(π_θ/π_ref) ∝ -(loss_agent - loss_ref)
        implicit_reward_w = loss_ref_w - loss_agent_w
        implicit_reward_l = loss_ref_l - loss_agent_l

        loss = -torch.nn.functional.logsigmoid(beta * (implicit_reward_w - implicit_reward_l))
        return loss.mean()

    @classmethod
    def load_from_checkpoint_and_config(
        cls,
        checkpoint_path: str,
        config: DictConfig,
        map_location: Optional[str] = None,
        strict: bool = True,
    ) -> tuple[DiffusionLightningModule, torch.nn.modules.module._IncompatibleKeys]:
        """Load model from checkpoint, but instead of using the config stored in the checkpoint,
        use the config passed in as an argument. This is useful when, e.g., an unused argument was
        removed in the code but is still present in the checkpoint config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config._target_ = "models.mattergen.pl_module.MatterGenModule"

        lightning_module = instantiate(config)
        lightning_module.config = config
        # assert isinstance(lightning_module, cls)

        # Restore state of the DiffusionLightningModule.
        result = lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)

        return lightning_module, result
