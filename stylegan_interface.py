import os
import torch
import dnnlib
import numpy as np
import legacy
from torch_utils.render import set_param, getTexPos, render, height_to_normal
from training.networks import Generator, MatUnet, weights_init
from torch_utils import misc
import PIL.Image

def load_generator_decoder(generator_pth, matunet_pth, use_fp16=False, device=torch.device('cuda')):
    G_tmp = legacy.load_network_pkl(dnnlib.util.open_url(generator_pth))['G_ema'].to(device)
    init_kwargs_tmp = G_tmp.init_kwargs
    res = init_kwargs_tmp['img_resolution']
    try:
        if 'mlp_fea' not in init_kwargs_tmp:
            init_kwargs_tmp['mlp_fea']=32
        if 'mlp_hidden' not in init_kwargs_tmp:
            init_kwargs_tmp['mlp_hidden']=64
        init_kwargs_tmp["synthesis_kwargs"].pop('high_res', None)
    except KeyError:
        print("Key Error")
        pass

    gen = Generator(*G_tmp.init_args, **init_kwargs_tmp).eval().requires_grad_(False)
    misc.copy_params_and_buffers(G_tmp, gen, require_all=True)
    dec = MatUnet(out_c = 8, batch_norm=False, layer_n=5).eval()
    gen = gen.to(device)
    dec = dec.to(device)
    dec.apply(weights_init)
    dec.load_state_dict(torch.load(matunet_pth)['MatUnet'])
    return gen, dec, res

def gnerate_random_render(gen, dec, bs, res, device=torch.device('cuda')):
    z = get_random_noise(bs, gen.z_dim)
    w_s = gen.mapping(z, None, truncation_psi=1, truncation_cutoff=14)
    light_color, _, scale = set_param(device)
    l_pos = get_rand_light_pos(scale)
    return generate_render(gen, dec, w_s, light_color, l_pos, scale, res, device)**(2.2), w_s[:,0:1,:]

def gnerate_random_render_from_batch(gen, dec, batch, res, device=torch.device('cuda')):
    w_s = gen.mapping(batch, None, truncation_psi=1, truncation_cutoff=14)
    light_color, _, scale = set_param(device)
    l_pos = get_rand_light_pos(scale)
    # full w_s is calculated from primary latent w(1, 512).
    return generate_render(gen, dec, w_s, light_color, l_pos, scale, res, device)**(2.2), w_s[:,0:1,:]

def generate_render(gen, dec, w_s, light_color, l_pos, scale, res, device=torch.device('cuda')):
    N, D, R, S = generate_material(gen, dec, w_s, device)
    rens = render_material(N, D, R, S, light_color, l_pos, scale, res, device)
    return rens

def generate_material(gen, dec, w_s, device=torch.device('cuda')):
    _, _, scale = set_param(device)
    fea = gen.synthesis(w_s, None, out_fea=True, noise_mode='const', test_mode=True, no_shift=True)
    maps = dec(fea) * 0.5 + 0.5 # (-1,1) --> (0,1)
    N = height_to_normal(maps[:,0:1,:,:], scale)
    D = maps[:,1:4,:,:].clamp(min=0, max=1)
    R = maps[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
    S = maps[:,5:8,:,:].clamp(min=0, max=1)
    return N, D, R, S

def render_material(N, D, R, S, light_color, l_pos, scale, res, device=torch.device('cuda')):
    tex_pos = getTexPos(res, scale, device).unsqueeze(0)
    light_pos = torch.tensor([l_pos]).to(device=device)
    ren_fea = torch.cat((N, D, R, S), dim=1)
    rens = render(ren_fea, tex_pos, light_color, light_pos, isMetallic=False, amb_li=True, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
    return rens

def get_random_noise(bs, z_dim, seed=None, device=torch.device('cuda')):
    #if seed is not None:
        #np.random.RandomState(seed)
    
    z = torch.from_numpy(np.random.normal(0, 1, (bs, z_dim))).to(device=device)
    return z

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def get_rand_light_pos(scale):
    z = np.random.uniform(3.5, scale + 0.5)
    clp = 1.8 / 4 * z
    x = np.random.normal(0, clp / 8.0)
    y = np.random.normal(0, clp / 8.0)
    return [clamp(x, -clp, clp), clamp(y, -clp, clp), z]

def generate_carpaint(gen, dec, w_s, res, l_pos = None, device=torch.device('cuda')):
    N, D, R, S = generate_material(gen, dec, w_s, device)
    light_color, _, scale = set_param(device)
    if l_pos is None:
        l_pos = get_rand_light_pos(scale)
    rens = render_material(N, D, R, S, light_color, l_pos, scale, res, device)
    D = D**(2.2)
    rens = rens**(2.2)
    # save seperate maps
    stacked_image = torch.concat((N, D, R, S, rens), dim=2)
    return stacked_image

def generate_random_carpaints(generator_pth, matunet_pth, outdir, num, device=torch.device('cuda')):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    gen, dec, res = load_generator_decoder(generator_pth, matunet_pth, device=device)
    for i in range(num):
        print(f"Generating carpaint {i}/{num}")
        z = get_random_noise(gen.z_dim)
        w = gen.mapping(z, None, truncation_psi=1, truncation_cutoff=14) 
        l = [0, 0, 4.0000]
        stacked_image = generate_carpaint(gen, dec, w, l, res, device)
        PIL.Image.fromarray(stacked_image, 'RGB').save(os.path.join(outdir, f"{i}_maps.png"))

def generate_lanten_w_walk(generator_pth, matunet_pth, outdir, seed_from, seed_to, num, device=torch.device('cuda')):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    gen, dec, res = load_generator_decoder(generator_pth, matunet_pth, device=device)
    z = get_random_noise(gen.z_dim, seed=seed_from)
    w = gen.mapping(z, None, truncation_psi=1, truncation_cutoff=14)   
    z_t = get_random_noise(gen.z_dim, seed=seed_to)
    w_t = gen.mapping(z_t, None, truncation_psi=1, truncation_cutoff=14) 
    step = 1.0 / num  
    for i in range(num):
        print(f"Generating latent walk {i}/{num}")
        w_new = w * (1 - step * i) + w_t * step * i
        l = [0, 0, 4.0000]
        stacked_image = generate_carpaint(gen, dec, w_new, l, res, device)
        PIL.Image.fromarray(stacked_image, 'RGB').save(os.path.join(outdir, f"{i}_maps.png"))
