from typing import Any
import torch
import pytorch_lightning as pl
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import default, count_params
from utils import disabled_train

class MappingWrapper(pl.LightningModule):
    def __init__(self, mapping_config, style_gan_config, cond_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = instantiate_from_config(mapping_config)
        count_params(self.model, verbose=True)
        self.instantiate_style_gan(style_gan_config)
        self.instantiate_cond_stage(cond_stage_config)

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval().to(self.device)
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
            

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
        x = torch.randn_like(w)
        w_pred = self.model(x, cls)
        img_pred = self.style_gan_model.generate_maps_from_w(w_pred)
        log = dict()
        log["GT"] = image_gt
        log["Pred"] = img_pred
        return log, "image"