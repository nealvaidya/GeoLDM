import torch

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion import utils as diffusion_utils



class EnLatentDiffusion_Distiller(torch.nn.Module):
    def __init__(
        self, student: EnVariationalDiffusion, teacher: EnVariationalDiffusion, **kwargs
    ):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher

    def denoise_step(self, model, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context):
        return z_t / alpha_t - model.phi(z_t, t, node_mask, edge_mask, context) * sigma_t / alpha_t


    def sample_time_steps(self, model, N, x):

        # Sample a timestep t.
        t_int = torch.randint(
            1, model.T + 1, size=(x.size(0), 1), device=x.device).float()

        t = t_int / model.T
        u = t - .5/N
        v = t - 1/N

        # Compute gamma_s and gamma_t via the network.
        gamma_t = model.inflate_batch_array(model.gamma(t), x)
        gamma_u = model.inflate_batch_array(model.gamma(u), x)
        gamma_v = model.inflate_batch_array(model.gamma(v), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t, sigma_t = model.alpha(gamma_t, x), model.sigma(gamma_t, x)
        alpha_u, sigma_u = model.alpha(gamma_u, x), model.sigma(gamma_u, x)
        alpha_v, sigma_v = model.alpha(gamma_v, x), model.sigma(gamma_v, x)

        return t, u, v, alpha_t, sigma_t, alpha_u, sigma_u, alpha_v, sigma_v

    # Not training the VAE here
    @torch.no_grad
    def encode_to_latent_space(self, model, x, h, node_mask, edge_mask, context):

        # Encode data to latent space.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = model.vae.encode(x, h, node_mask, edge_mask, context)
        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device = x.device)
        model.gamma.to('cuda')
        gamma_0 = model.inflate_batch_array(model.gamma(t_zeros), x)
        sigma_0 = model.sigma(gamma_0, x)

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = model.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)

        z_x = z_xh[:, :, :model.n_dims]
        z_h = z_xh[:, :, model.n_dims:]
        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}

        return z_x, z_h

    def forward(self, x, h, node_mask, edge_mask, context):

        # Send batch to latent space
        z_x, z_h =  self.encode_to_latent_space(self.teacher, x, h, node_mask, edge_mask, context)

        # sample timesteps
        timesteps = torch.randint(
            1, self.student.T, size=(x.size(0), 1), device=x.device
        ).float()

        t = timesteps/self.student.T
        u = (timesteps - 0.5)/self.student.T
        v = (timesteps - 1)/self.student.T

        # Compute gamma_s and gamma_t via the network.
        gamma_t = self.teacher.inflate_batch_array(self.teacher.gamma(t), x)
        gamma_u = self.teacher.inflate_batch_array(self.teacher.gamma(u), x)
        gamma_v = self.teacher.inflate_batch_array(self.teacher.gamma(v), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t, sigma_t = self.teacher.alpha(gamma_t, x), self.teacher.sigma(gamma_t, x)
        alpha_u, sigma_u = self.teacher.alpha(gamma_u, x), self.teacher.sigma(gamma_u, x)
        alpha_v, sigma_v = self.teacher.alpha(gamma_v, x), self.teacher.sigma(gamma_v, x)

        # Concatenate x, h[integer] and h[categorical].
        z_0 = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)

        # forward diffusion
        eps = self.teacher.sample_combined_position_feature_noise(n_samples=x.size(0),
                                                                 n_nodes=x.size(1), node_mask=node_mask)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * z_0 + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.teacher.n_dims], node_mask)

        # compute teacher target (no_grad so the teacher stays fixed)
        with torch.no_grad():

            # Compute double denoising steps
            z0_hat_zt = self.denoise_step(self.teacher, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context)
            z_u = alpha_u * z0_hat_zt + (sigma_u/sigma_t) * (z_t - alpha_t * z0_hat_zt)

            z0_hat_zu = self.denoise_step(self.teacher, z_u, alpha_u, sigma_u, u, node_mask, edge_mask, context)
            z_v = alpha_v * z0_hat_zu + (sigma_v/sigma_u) * (z_u - alpha_u * z0_hat_zu)

            teacher_target = (z_v - (sigma_v/sigma_t)*z_t)/(alpha_v - (sigma_v/sigma_t)*alpha_t)

            lambda_t = torch.log(alpha_t**2/sigma_t**2)

        # Detach target and inputs for *extra* caution
        teacher_target.detach()
        z_t.detach()
        alpha_t.detach()
        sigma_t.detach()
        t.detach()

        student_gamma_t = self.student.inflate_batch_array(self.student.gamma(t), x)
        alpha_t, sigma_t = self.student.alpha(student_gamma_t, x), self.student.sigma(student_gamma_t, x)
        student_pred = self.denoise_step(self.student, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context)

        return lambda_t, teacher_target, student_pred



