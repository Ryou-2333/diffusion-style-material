from utils import disabled_train
from ldm.models.autoencoder import AutoencoderKL
from utils import instantiate_from_config

class Material_VAE(AutoencoderKL):
    def __init__(self, style_gan_config, *args, **kwargs):
        super(self).__init__(*args, **kwargs)
        self.instantiate_style_gan(style_gan_config)
    
    def instantiate_style_gan(self, config):
        model = instantiate_from_config(config)
        self.style_gan_model = model.eval().to(self.device)
        self.style_gan_model.train = disabled_train
        for param in self.style_gan_model.parameters():
            param.requires_grad = False

    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        return super().log_images(batch, only_inputs, log_ema, **kwargs)