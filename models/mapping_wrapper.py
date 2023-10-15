from typing import Any
import torch
import pytorch_lightning as pl
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import default, count_params
from utils import disabled_train, exist
import os

class MappingWrapper(pl.LightningModule):
    def __init__(self, mapping_config, style_gan_config, cond_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = instantiate_from_config(mapping_config)
        count_params(self.model, verbose=True)
        self.instantiate_style_gan(style_gan_config)
        self.instantiate_cond_stage(cond_stage_config)

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval().to(self.device)
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
            
    def fid_evaluation(self):
        return ""

    def instantiate_style_gan(self, config):
        model = instantiate_from_config(config)
        self.style_gan_model = model.eval().to(self.device)
        self.style_gan_model.train = disabled_train
        for param in self.style_gan_model.parameters():
            param.requires_grad = False
    
    def get_learned_conditioning(self, img):
        c = self.cond_stage_model.encode(self.cond_stage_model.preprocess(img))
        return c
    
    def forward(self, x, c, *args: Any, **kwargs: Any):
        return self.losses(x, c, *args, **kwargs)
    
    def losses(self, x_start, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        model_output = self.model(noise, cond)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        target = x_start
        loss = torch.nn.functional.mse_loss(target, model_output)
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        x, c = self.get_input(batch)
        loss, loss_dict = self(x, c)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    @torch.no_grad()
    def get_input(self, batch, *args, **kwargs):
        image, w = self.style_gan_model.gnerate_render_w(batch)
        cls = self.get_learned_conditioning(image).detach()
        return w, cls
    
    @torch.no_grad()
    def log_images(self, batch, steps, **kwargs):
        batch_size = batch.shape[0]
        image_gt, w = self.style_gan_model.gnerate_render_w(batch)
        cls = self.get_learned_conditioning(image_gt).detach()
        x = torch.randn_like(w).to(self.device)
        w_pred = self.model(x, cls)
        img_pred = self.style_gan_model.generate_render_from_w(w_pred)
        log = dict()
        log["GT"] = image_gt
        log["Pred"] = img_pred
        return log, "image"
    
    @torch.no_grad()
    def generate_w(self, example, batch_size, unconditional_guidance_scale=1., seed=None, reuse_seed=False, **kwargs):
        if reuse_seed:
            seed = os.environ.get("PL_GLOBAL_SEED")
        elif seed == -1:
            seed = None
            if exist(os.environ.get("PL_GLOBAL_SEED")):
                del os.environ["PL_GLOBAL_SEED"]
        pl.seed_everything(seed)
        c = self.get_learned_conditioning(example).detach()
        x = torch.randn((batch_size, self.model.z_dim)).to(self.device)
        w_pred = self.model(x, c)
        return w_pred, []