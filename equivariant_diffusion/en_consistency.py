import torch
import numpy as np

from equivariant_diffusion.en_diffusion import (
    EnHierarchicalVAE,
    EnLatentDiffusion,
    disabled_train,
)
from equivariant_diffusion import utils as diffusion_utils


class EnLatentConsistency(EnLatentDiffusion):
    """
    The E(n) Latent Consistency Module.
    """

    def __init__(
        self,
        teacher_model,
        sigma_data: float = 0.5,
        timestep_scaling: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma_data = sigma_data
        self.timestep_scaling = timestep_scaling

        if teacher_model:
            self.teacher_pred_noise = teacher_model.phi
        else:
            self.teacher_pred_noise = self.teacher_approx

    def teacher_approx(z_t, t, node_mask, edge_mask, context):
        pass

    def scalings_for_boundary_conditions(self, timestep):
        scaled_timestep = timestep * self.timestep_scaling

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = (timestep / 0.1) / (scaled_timestep**2 + self.sigma_data**2) ** 0.5

        return c_skip.unsqueeze(-1), c_out.unsqueeze(-1)

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample noise epsilon ~ Normal(0, I)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device
        ).float()
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1].
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Concatenate x, h[integer] and h[categorical].
        # There shouldn't be any categorical features here, so just keep the int
        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        xh = torch.cat([x, h["integer"]], dim=2)

        # Reparamaterize to sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        # Predict noise
        eps_pred = self.phi(z_t, t, node_mask, edge_mask, context)

        # Predict latent unnoised from noise
        xh_pred = (z_t - eps_pred * sigma_t) / alpha_t

        # Implement skip connection for consistency boundary
        # TODO: Should this be t or t_int?
        c_skip, c_out = self.scalings_for_boundary_conditions(t)
        model_pred = (c_skip * z_t) + (c_out * xh_pred)

    # def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
    #     """
    #     Get predictions
    #     """

    #     # VAE Part is the same as diffusion model:

    #     # Encode data to latent parameters ---------------
    #     z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(
    #         x, h, node_mask, edge_mask, context
    #     )

    #     # Infer latent z ---------------------------------
    #     z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)

    #     # Compute fixed sigma values.
    #     t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
    #     gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
    #     sigma_0 = self.sigma(gamma_0, x)
    #     diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
    #     z_xh_sigma = sigma_0

    #     # Sample from VAE
    #     z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
    #     z_xh = z_xh.detach()  # Always keep the encoder fixed.
    #     diffusion_utils.assert_correctly_masked(z_xh, node_mask)

    #     # Compute reconstruction loss ----------------------
    #     if self.trainable_ae:
    #         xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
    #         # Decoder output (reconstruction).
    #         x_recon, h_recon = self.vae.decoder._forward(
    #             z_xh, node_mask, edge_mask, context
    #         )
    #         xh_rec = torch.cat([x_recon, h_recon], dim=2)
    #         loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
    #     else:
    #         loss_recon = 0

    #     z_x = z_xh[:, :, : self.n_dims]
    #     z_h = z_xh[:, :, self.n_dims :]
    #     diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
    #     # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
    #     z_h = {"categorical": torch.zeros(0).to(z_h), "integer": z_h}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """

        # Encode data to latent space.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(
            x, h, node_mask, edge_mask, context
        )
        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
        sigma_0 = self.sigma(gamma_0, x)

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)

        # Compute reconstruction loss.
        if self.trainable_ae:
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            # Decoder output (reconstruction).
            x_recon, h_recon = self.vae.decoder._forward(
                z_xh, node_mask, edge_mask, context
            )
            xh_rec = torch.cat([x_recon, h_recon], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
        else:
            loss_recon = 0

        z_x = z_xh[:, :, : self.n_dims]
        z_h = z_xh[:, :, self.n_dims :]
        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {"categorical": torch.zeros(0).to(z_h), "integer": z_h}

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss_ld, loss_dict = self.compute_loss(
                z_x, z_h, node_mask, edge_mask, context, t0_always=False
            )
        else:
            # Less variance in the estimator, costs two forward passes.
            loss_ld, loss_dict = self.compute_loss(
                z_x, z_h, node_mask, edge_mask, context, t0_always=True
            )

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_h_given_z0(
            torch.cat([h["categorical"], h["integer"]], dim=2), node_mask
        )
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        neg_log_pxh = loss_ld + loss_recon + neg_log_constants

        return neg_log_pxh

    @torch.no_grad()
    def sample(
        self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False
    ):
        """
        Draw samples from the generative model.
        """
        z_x, z_h = super().sample(
            n_samples, n_nodes, node_mask, edge_mask, context, fix_noise
        )

        z_xh = torch.cat([z_x, z_h["categorical"], z_h["integer"]], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)
        x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)

        return x, h

    @torch.no_grad()
    def sample_chain(
        self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        chain_flat = super().sample_chain(
            n_samples, n_nodes, node_mask, edge_mask, context, keep_frames
        )

        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # chain[0] = xh  # Overwrite last frame with the resulting x and h.

        # chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        chain = chain_flat.view(keep_frames, n_samples, *chain_flat.size()[1:])
        chain_decoded = torch.zeros(
            size=(*chain.size()[:-1], self.vae.in_node_nf + self.vae.n_dims),
            device=chain.device,
        )

        for i in range(keep_frames):
            z_xh = chain[i]
            diffusion_utils.assert_mean_zero_with_mask(
                z_xh[:, :, : self.n_dims], node_mask
            )

            x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain_decoded[i] = xh

        chain_decoded_flat = chain_decoded.view(
            n_samples * keep_frames, *chain_decoded.size()[2:]
        )

        return chain_decoded_flat

    def instantiate_first_stage(self, vae: EnHierarchicalVAE):
        if not self.trainable_ae:
            self.vae = vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = vae.train()
            for param in self.vae.parameters():
                param.requires_grad = True
