import torch
import pytorch_lightning as pl
import os
from ldm.models.diffusion.ddpm import LatentDiffusion, default
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
from utils import exist, disabled_train

class StyleLatentDiffusion(LatentDiffusion):
    def __init__(self, style_gan_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instantiate_style_gan(style_gan_config)
    
    def instantiate_style_gan(self, config):
        model = instantiate_from_config(config)
        self.style_gan_model = model.eval().to(self.device)
        self.style_gan_model.train = disabled_train
        for param in self.style_gan_model.parameters():
            param.requires_grad = False

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(self.cond_stage_model.preprocess(c))
            else:
                c = self.cond_stage_model(self.cond_stage_model.preprocess(c))
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2))
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    @torch.no_grad()
    def get_input(self, batch, *args, **kwargs):
        image, w = self.style_gan_model.gnerate_render_w(batch)
        cls = self.get_learned_conditioning(image).detach()
        return [w, cls]

    @torch.no_grad()
    def generate_w(self, example, batch_size, unconditional_guidance_scale=1., seed=None, reuse_seed=False, steps=20, eta=0):
        # set global seed for generation
        if reuse_seed:
            seed = os.environ.get("PL_GLOBAL_SEED")
        elif seed == -1:
            seed = None
            if exist(os.environ.get("PL_GLOBAL_SEED")):
                del os.environ["PL_GLOBAL_SEED"]
        pl.seed_everything(seed)
        v = self.get_learned_conditioning(example)
        c = v

        sampler = DPMSolverSampler(self, self.device)
        samples, inters = sampler.sample(steps, conditioning=c, batch_size=batch_size,
                                    shape=(self.channels, self.image_size),
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=None, eta=eta, x_T=None)
        
        return samples, inters

    @torch.no_grad()
    def log_images(self, batch, steps, **kwargs):
        batch_size = batch.shape[0]
        image_gt, _ = self.style_gan_model.gnerate_render_w(batch)
        cls = self.get_learned_conditioning(image_gt).detach()
        sampler = DPMSolverSampler(self, self.device)
        samples, _ = sampler.sample(steps, conditioning=cls, batch_size=batch_size,
                                    shape=(self.channels, self.image_size),
                                    unconditional_guidance_scale=1.,
                                    unconditional_conditioning=cls, eta=0, x_T=None)
        w = self.decode_first_stage(samples)

        img_pred = self.style_gan_model.generate_maps(w)
        log = dict()
        log["GT"] = image_gt
        log["Pred"] = img_pred
        return log, self.first_stage_key
