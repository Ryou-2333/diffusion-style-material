from typing import Any
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from ldm.util import instantiate_from_config

def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MappingNet(nn.Module):
    def __init__(
        self,
        noise_size : int,
        context_size : int,
        context_embed_dim: int,
        depth : int,
        use_checkpoint=False,
        use_fp16=False,
    ):
        super().__init__()
        self.noise_size = noise_size
        self.context_size = context_size
        self.context_embed_dim = context_embed_dim
        self.hidden_size = noise_size + context_embed_dim
        self.dtype = th.float16 if use_fp16 else th.float32
        self.use_checkpoint = use_checkpoint

        self.contex_embed = nn.Sequential(
            nn.Linear(context_size, context_embed_dim),
            nn.SiLU(),
            nn.Linear(context_embed_dim, context_embed_dim)
        )

        layers = []
        for _ in range(depth):
            layers += [  
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                nn.GELU(),
                ]

        self.hidden_layers = nn.Sequential(*layers)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, noise_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.noise_size, noise_size, bias=True),
            nn.Linear(self.noise_size, noise_size, bias=True),
        )

    def forward(self, x, context):
        c_emb = self.contex_embed(context)
        x_t = th.concat((x, c_emb), dim=-1)
        x_t = self.hidden_layers(x_t)
        return self.out(x_t)

class MappingWrapper(pl.LightningModule):
    def __init__(self, mapping_config, style_gan_config, cond_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instantiate_style_gan(style_gan_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_mapping(mapping_config)
    
    def instantiate_mapping(self, config):
        model = instantiate_from_config(config)
        self.mapping_model = model.eval().to(self.device)
        self.mapping_model.train = disabled_train
        for param in self.mapping_model.parameters():
            param.requires_grad = False

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
        return self.mapping_model(x, c, *args, **kwargs)