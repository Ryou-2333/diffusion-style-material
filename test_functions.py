import stylegan_interface
import torch
from PIL import Image
from cldm.model import create_model
import os
import numpy as np
import torch
from torchvision import transforms
import cv2

def test_laten_walk(num_frames, bias, outdir='output_carpaint', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    stylegan_interface.generate_lanten_w_walk(gen_path, dec_path, outdir, bias, num_frames)

def test_random_render_generate(num, row, col, outdir='output_rendered', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    gen, dec, res = stylegan_interface.load_generator_decoder(gen_path, dec_path)
    for k in range(num):
        combined_imgs = []
        for i in range(row):
            print(f"Generating random render {i}/{row}...")
            rendered, w = stylegan_interface.gnerate_random_render(gen, dec, col, res, dir_flag=True)
            rendered = (rendered.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            imgs = []
            for j in range(col):
                imgs.append(rendered[j, :, :, :])
            combined_imgs.append(np.hstack(imgs))
        final_image = np.vstack(combined_imgs)
        Image.fromarray(final_image, 'RGB').save(f"{outdir}/render_{k}.png")

def test_material_generate(num, outdir='output_carpaint', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    stylegan_interface.generate_random_carpaints(gen_path, dec_path, outdir, num)

def test_create_model(path):
    model_name = 'diffusion_style'
    model = create_model(f'./configs/{model_name}.yaml')
    write_to_file(path, model)

def test_inference():
    model_name = 'diffusion_style'
    model = create_model(f'./configs/{model_name}.yaml').cuda().eval()
    img = torch.rand(1, 3, 512, 512)
    result = model.generate_image(img, 1)
    print(result.shape)
    img = Image.fromarray(result, 'RGB')
    img.save("result.png")

def test_random_generation(num, outdir='output_random_w', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    gen, dec, res = stylegan_interface.load_generator_decoder(gen_path, dec_path)
    bs = 1
    device = torch.device('cuda')
    for i in range(num):
        w = torch.rand(bs, 1, 512).to(device)
        w_m = stylegan_interface.get_meaningful_w(gen, bs)
        loss = torch.nn.functional.mse_loss(w_m, w)
        print(f"loss_{i}:{loss}")
        w_s = w.repeat([1, 16, 1])
        rendered = stylegan_interface.gnerate_random_render_from_w(gen, dec, w_s, res)
        rendered = (rendered.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imgs = []
        for j in range(bs):
            imgs.append(rendered[j, :, :, :])
        combined_img = np.hstack(imgs)
        Image.fromarray(combined_img, 'RGB').save(f"{outdir}/render_{i}.png")

def test_mean_variation_w(num, gen_path='weights/photomat/G_512.pkl'):
    gen, _ = stylegan_interface.load_generator(gen_path)
    means = []
    vars = []
    mins = []
    maxs = []
    for _ in range(num):
        w = stylegan_interface.get_meaningful_w(gen)
        means.append(torch.mean(w))
        vars.append(torch.var(w))
        mins.append(torch.min(w))
        maxs.append(torch.max(w))
    
    print(f"mean({sum(means) / len(means)}), var({sum(vars) / len(vars)}), min({min(mins)}), max({max(maxs)})")

def write_to_file(path, o):
    with open(path, 'w') as f:
        f.write(str(o))

def render_test_data(in_dir='../test/in', outdir='../test/out', res = 512):
    to_tensor = transforms.ToTensor()

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for mat in os.listdir(in_dir):
        mat_dir = os.path.join(in_dir, mat)
        if os.path.isdir(mat_dir):
            M = torch.zeros(1, 3, 512, 512)
            D = torch.zeros(1, 3, 512, 512)
            N = torch.ones(1, 3, 512, 512)
            N[:, 2, :, :] = 1
            R = torch.zeros(1, 3, 512, 512)
            for tex in os.listdir(mat_dir):
                tex_path = os.path.join(mat_dir, tex)
                if(tex.endswith('diff.jpg')):
                    img = cv2.imread(tex_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (res, res))
                    D = to_tensor(img_resized).unsqueeze(0)
                elif(tex.endswith('rou.jpg')):
                    img = cv2.imread(tex_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (res, res))
                    R = to_tensor(img_resized).clamp(min=0.2, max=0.9).unsqueeze(0)
                elif(tex.endswith('metal.jpg')):
                    img = cv2.imread(tex_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (res, res))
                    M = to_tensor(img_resized).unsqueeze(0)
                elif(tex.endswith('nor.jpg')):
                    img = cv2.imread(tex_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (res, res))
                    N = (to_tensor(img_resized) * 2 - 1).unsqueeze(0)
            light_color, _, scale = stylegan_interface.set_param(device=D.device)
            l_pos = stylegan_interface.get_rand_light_pos(scale)
            dir_dir = stylegan_interface.get_rand_light_pos(scale * 4)
            dir_color=np.random.normal(0.1, 0.5) * light_color
            rendered = stylegan_interface.render_material(N, D, R, M, light_color, l_pos, scale, 512, D.device, dir_flag=False, isMetal=True)
            rendered = (rendered.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            rendered = rendered.squeeze(0)
            Image.fromarray(rendered, 'RGB').save(f"{outdir}/{mat}_rendered.png")
            
if __name__ == '__main__':
    #test_random_render_generate(1000, 1, 1)
    #test_material_generate(100)
    #test_create_model('diffusion_style.model')
    #test_inference()
    #for i in range(20):
    #test_laten_walk(10, 1, f"latent_walk_{i}")
    render_test_data("../../TexMat", "../../TexMat_rendered")
