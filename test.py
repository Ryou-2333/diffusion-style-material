import torch
from utils import instantiate_from_config
from stylegan_interface import load_generator_decoder, generate_carpaint, generate_render, set_param
import yaml
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import os

MODEL_DICTS = {
    "cls-mlp": "configs/mlp.yaml",
    "local-attn": "configs/local.yaml",
    "full-dp15": "configs/local.yaml",
    "style-mapping": "configs/mapping.yaml",
    "unet-dp15": "configs/u_net.yaml",
}

CKPT_DICTS = {
    "cls-mlp": "checkpoints/cls-mlp/latest.ckpt",
    "local-attn": "checkpoints/local-dir/epoch-10.ckpt",
    "full-dp15": "checkpoints/full-dir-dp15/epoch-25.ckpt",
    "style-mapping": "checkpoints/style-mapping/latest.ckpt",
    "unet-dp15": "checkpoints/unet-dp15/latest.ckpt",
}

gen_path = 'weights/photomat/G_512.pkl'
dec_path='weights/photomat/MatUnet_512.pt'

transform = transforms.Compose([
    transforms.ToTensor()
])

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

def load_models(model_name):
    config_file = MODEL_DICTS[model_name]
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f.read())
    model = instantiate_from_config(configs["model"]).eval().cuda()
    model.init_from_ckpt(CKPT_DICTS[model_name])
    gen, dec, res = load_generator_decoder(gen_path, dec_path, device=model.device)
    return model, gen, dec, res

def test_model(model_name, input_pth, output_dir, output_inter, count=1, gs = 1.0, random_seed = False, name=0, save_material = True):
    img = Image.open(input_pth)
    model, gen, dec, res = load_models(model_name)
    img_t = transform(img).unsqueeze(0).to(model.device)
    imgs = []
    for i in range(count):
        if random_seed: 
            seed=random.randint(min_seed_value, max_seed_value)
        else:
            seed=i * 23572
        w, inters = model.generate_w(img_t, 1, unconditional_guidance_scale=gs, seed=seed)
        w_s = w.repeat([1, 16, 1]).to(model.device)
        if save_material:
            save_carpaints(model, gen, dec, w_s, inters, res, output_dir, name, gs, seed, output_inter)
        else:
            if output_inter:
                outs = generate_render_img(gen, dec, w_s, inters, res, model.device, output_inter)
                out = Image.fromarray(np.hstack(outs), 'RGB')
                out.save(f"{output_dir}/result_{name}_gs{gs}_{seed}.png")
            else:
                out = generate_render_img(gen, dec, w_s, inters, res, model.device, output_inter)
                imgs.append(out)
    
    if not save_material and not output_inter:
        out = Image.fromarray(np.hstack(imgs), 'RGB')
        out.save(f"{output_dir}/result_{name}_gs{gs}.png")

def conditional_latent_walk(model_name, from_pth, to_pth, num_frames, seed, gs, outdir='../diffusion-style-material-outputs/latent_walk', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    from_img = Image.open(from_pth)
    to_img = Image.open(to_pth)
    model, gen, dec, res = load_models(model_name)
    f_img_t = transform(from_img).unsqueeze(0).to(model.device)
    t_img_t = transform(to_img).unsqueeze(0).to(model.device)
    step = 1.0 / num_frames  
    light_color, _, scale = stylegan_interface.set_param()
    light_pos = np.array([[0, 0, 4]])
    imgs = []
    for i in range(num_frames+1):
        c_new = w_f * (1 - step * i) + w_t * step * i
        w, inters = model.generate_w(img_t, 1, unconditional_guidance_scale=gs, seed=seed)
        w_s = w.repeat([1, 16, 1]).to(model.device)
        out = generate_render_img(gen, dec, w_s, inters, res, model.device, output_inter)
        imgs.append(out)
        img = stylegan_interface.generate_render(gen, dec, w_new, light_color, light_pos, scale, res, dir_flag=False)
        img = (img*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        imgs.append(img)
        if i == 0:
            Image.fromarray(img, 'RGB').save(os.path.join(outdir, f"start_latent_walk.png"))
        elif i == num_frames:
            Image.fromarray(img, 'RGB').save(os.path.join(outdir, f"end_latent_walk.png"))

    stacked_image = np.hstack(imgs)
    Image.fromarray(stacked_image, 'RGB').save(os.path.join(outdir, f"{i}_latent_walk.png"))

def generate_render_img(gen, dec, w_s, inters, res, device, output_inter=False):
    light_color, _, scale = set_param(device)
    light_pos = np.array([[0, 0, 4]])
    if not output_inter:
        out = generate_render(gen, dec, w_s, light_color, light_pos, scale, res, device, dir_flag=False)
        out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        out = out.squeeze(0).detach()
        out = out.cpu().numpy()
        return out
    else:
        out = generate_render(gen, dec, w_s, light_color, light_pos, scale, res, device, dir_flag=False)
        out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        out = out.squeeze(0).detach()
        out = out.cpu().numpy()
        inter_outs = []
        for inter in inters:
            i_s = inter.repeat([1, 16, 1]).to(device)
            out = generate_render(gen, dec, i_s, light_color, light_pos, scale, res, device, dir_flag=False)
            out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            out = out.squeeze(0).detach()
            out = out.cpu().numpy()
            inter_outs.append(out)
        inter_outs.append[out]
        return inter_outs

def save_carpaints(model, gen, dec, w_s, inters, res, output_dir, name, gs, seed, output_inter=False):
    out = generate_carpaint(gen, dec, w_s, res, device=model.device)
    out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    out = out.squeeze(0).detach()
    out = Image.fromarray(out.cpu().numpy(), 'RGB')
    out.save(f"{output_dir}/result_{name}_gs{gs}_{seed}.png")
    if(output_inter):
        j = 0
        for inter in inters:
            i_s = inter.repeat([1, 16, 1]).to(model.device)
            out = generate_carpaint(gen, dec, i_s, res, device=model.device)
            out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            out = out.squeeze(0).detach()
            out = Image.fromarray(out.cpu().numpy(), 'RGB')
            out.save(f"{output_dir}/inter_{name}_gs{gs}_{j}_{seed}.png")
            j += 1

#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.1)
#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.2)
#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.3)
#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.4)
#test_model("cls-mlp", "checkpoints/full-dir-dp15/marble03.jpg", "checkpoints/cls-mlp", False, 20, 2.5, random_seed=False, name=0, save_material=True)
test_model("unet-dp15", "checkpoints/full-dir-dp15/marble03.jpg", "checkpoints/unet-dp15", False, 20, 2.5, random_seed=False, name=0, save_material=True)
test_model("full-dp15", "checkpoints/full-dir-dp15/marble03.jpg", "checkpoints/full-dir-dp15", False, 20, 2.5, random_seed=False, name=0, save_material=True)

