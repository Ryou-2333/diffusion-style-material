import torch
import pytorch_lightning as pl
from stylegan_interface import load_generator_decoder, gnerate_random_render_from_w, gnerate_random_render_from_batch, generate_render, set_param, get_rand_light_pos

class StyleGANWrapper(pl.LightningModule):
    def __init__(self,
                 generator_pth: str,
                 decoder_pth: str,
                 num_ws=16,
                 use_fp16=False,
                 ):
        super().__init__()
        #Freeze all layers
        self.gen, self.dec, self.res = load_generator_decoder(generator_pth, decoder_pth, use_fp16, self.device)
        self.num_ws = num_ws
    
    def get_render_loss(self, w_pred, w_gt):
        w_s_pred = w_pred.repeat([1, self.num_ws, 1])
        w_s_gt = w_gt.repeat([1, self.num_ws, 1])
        light_color, _, scale = set_param(self.device)
        l_pos = get_rand_light_pos(scale) 
        pred = generate_render(self.gen, self.dec, w_s_pred, light_color, l_pos, scale, self.res, self.device, False)
        gt = generate_render(self.gen, self.dec, w_s_gt, light_color, l_pos, scale, self.res, self.device, False)
        loss = torch.nn.functional.mse_loss(gt, pred)
        return loss

    def gnerate_render_w(self, batch):
        return gnerate_random_render_from_batch(self.gen, self.dec, batch, self.res, device=self.device)
            
    def generate_maps(self, w):
        w = w.to(self.device)
        w_s = w.repeat([1, self.num_ws, 1])
        return gnerate_random_render_from_w(self.gen, self.dec, w_s, self.res, self.device)

