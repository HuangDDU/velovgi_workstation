import numpy as np
import torch

def add_velovi_outputs_to_adata(adata, vae):
    """Add velocity/rate/t from model to adata

    Args:
        adata (_type_): adata object
        vae (_type_): vae model
    """
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")
    velocities_u = vae.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")  # TODO: 加入unsplcied的速率
    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["velocity_u"] = velocities_u / scaling # TODO: 加入unsplcied的速率
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0
