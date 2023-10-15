import torch
import pytorch_lightning as pl
import stylegan_interface as si
import numpy as np

class StyleGANWrapper(pl.LightningModule):
    def __init__(self,
                 generator_pth: str,
                 decoder_pth: str,
                 num_ws=16,
                 use_fp16=False,
                 use_dir_li=False,
                 ):
        super().__init__()
        #Freeze all layers
        self.gen, self.dec, self.res = si.load_generator_decoder(generator_pth, decoder_pth, use_fp16, self.device)
        self.num_ws = num_ws
        self.dir_flag = use_dir_li
    
    def get_render_loss(self, w_pred, w_gt):
        w_s_pred = w_pred.repeat([1, self.num_ws, 1])
        w_s_gt = w_gt.repeat([1, self.num_ws, 1])
        light_color, _, scale = si.set_param(self.device)
        l_pos = si.get_rand_light_pos(scale) 
        pred = si.generate_render(self.gen, self.dec, w_s_pred, light_color, l_pos, scale, self.res, self.device, False)
        gt = si.generate_render(self.gen, self.dec, w_s_gt, light_color, l_pos, scale, self.res, self.device, False)
        loss = torch.nn.functional.mse_loss(gt, pred)
        return loss

    def gnerate_render_w(self, batch, random=True):
        if random:
            return si.gnerate_random_render_from_batch(self.gen, self.dec, batch, self.res, device=self.device, dir_flag=self.dir_flag)
        else:
            light_color, _, scale = si.set_param(self.device)
            light_pos = np.array([[0, 0, 4]])
            return si.gnerate_render_from_batch(self.gen, self.dec, batch, light_color, light_pos, scale, self.res, device=self.device, dir_flag=False)
        
    def generate_render_from_w(self, w, random=True):
        w = w.to(self.device)
        w_s = w.repeat([1, self.num_ws, 1])
        if random:
            return si.gnerate_random_render_from_w(self.gen, self.dec, w_s, self.res, self.device)
        else:
            light_color, _, scale = si.set_param(self.device)
            light_pos = np.array([[0, 0, 4]])
            return si.generate_render(self.gen, self.dec, w_s, light_color, light_pos, scale, self.res, device=self.device, dir_flag=False)

    
    def gnerate_render_fea(self, batch):
        return si.gnerate_random_render_and_fea_from_batch(self.gen, self.dec, batch, self.res, device=self.device)
            
    def generate_maps_from_fea(self, fea):
        return si.gnerate_random_render_from_fea(self.gen, self.dec, fea, self.res, self.device)

