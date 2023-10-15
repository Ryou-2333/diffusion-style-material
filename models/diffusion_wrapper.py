import torch
import pytorch_lightning as pl
import os
from ldm.models.diffusion.ddpm import LatentDiffusion, default
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
from utils import exist, disabled_train
from torch_utils.fid_score import InceptionFID

class StyleLatentDiffusion(LatentDiffusion):
    def __init__(self, condition_drop_rate, fid_eval_count, fid_eval_batch, style_gan_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_drop_rate = condition_drop_rate
        self.instantiate_style_gan(style_gan_config)
        self.fid_eval_count = fid_eval_count
        self.fid_eval_batch = fid_eval_batch
    
    def instantiate_style_gan(self, config):
        model = instantiate_from_config(config)
        self.style_gan_model = model.eval().to(self.device)
        self.style_gan_model.train = disabled_train
        for param in self.style_gan_model.parameters():
            param.requires_grad = False

    def get_learned_conditioning(self, c):
        c = self.cond_stage_model.encode(self.cond_stage_model.preprocess(c))
        if self.training and self.condition_drop_rate:
            c = torch.bernoulli((1 - self.condition_drop_rate) * torch.ones(c.shape[0], device=c.device)[:, None, None]) * c

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

    def fid_evaluation(self):
        self.eval()
        batch_size = self.fid_eval_batch
        count = self.fid_eval_count
        data_size = batch_size * count
        net = InceptionFID(data_size=data_size, device=self.device)

        print(f'Evaluation data size {data_size}, begin to evaluate training via FID distance...')
        w_loss = 0
        for _ in range(count):
            batch = torch.randn((batch_size, 512)).cuda()
            img, w = self.style_gan_model.gnerate_render_w(batch, False)
            pred_w, _ = self.generate_w(img, batch_size)
            pred = self.style_gan_model.generate_render_from_w(pred_w, False)
            net.accumulate_statistics_of_imgs(img, target='real')
            net.accumulate_statistics_of_imgs(pred, target='fake')
            net.forward_idx(batch_size)
            l = self.get_loss(pred_w, w, mean=False).mean()
            w_loss += l

        w_loss = w_loss / count
        fid = net.fid_distance()
        self.train()
        return (f', \tfid_score: {fid:.6f}, \tw_loss: {w_loss:.6f}')

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
        c = self.get_learned_conditioning(example).detach()
        sampler = DPMSolverSampler(self, self.device)
        samples, inters = sampler.sample(steps, conditioning=c, batch_size=batch_size,
                                    shape=(self.channels, self.image_size),
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=torch.zeros_like(c), eta=eta, x_T=None)
        w = self.decode_first_stage(samples)
        return w, inters

    @torch.no_grad()
    def log_images(self, batch, steps, **kwargs):
        batch_size = batch.shape[0]
        image_gt, _ = self.style_gan_model.gnerate_render_w(batch)
        c = self.get_learned_conditioning(image_gt).detach()
        sampler = DPMSolverSampler(self, self.device)
        samples, _ = sampler.sample(steps, conditioning=c, batch_size=batch_size,
                                    shape=(self.channels, self.image_size),
                                    unconditional_guidance_scale=1.,
                                    unconditional_conditioning=torch.zeros_like(c), eta=0, x_T=None)
        w = self.decode_first_stage(samples)

        img_pred = self.style_gan_model.generate_render_from_w(w)
        log = dict()
        log["GT"] = image_gt
        log["Pred"] = img_pred
        return log, self.first_stage_key
