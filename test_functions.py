import stylegan_interface
import torch
from PIL import Image
from cldm.model import create_model
import os
import numpy as np

def test_laten_walk(num_frames, bias, outdir='output_carpaint', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    stylegan_interface.generate_lanten_w_walk(gen_path, dec_path, outdir, bias, num_frames)

def test_random_render_generate(num, outdir='output_rendered', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    gen, dec, res = stylegan_interface.load_generator_decoder(gen_path, dec_path)
    bs = 8
    for i in range(num):
        print(f"Generating random render {i}/{num}...")
        rendered, w = stylegan_interface.gnerate_random_render(gen, dec, bs, res)
        rendered = (rendered.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imgs = []
        for j in range(bs):
            imgs.append(rendered[j, :, :, :])
        combined_img = np.hstack(imgs)
        Image.fromarray(combined_img, 'RGB').save(f"{outdir}/render_{i}.png")

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

if __name__ == '__main__':
    #test_random_render_generate(10)
    #test_material_generate(100)
    #test_create_model('diffusion_style.model')
    #test_inference()
    #for i in range(20):
    test_laten_walk(10, 1, f"latent_walk_{i}")
