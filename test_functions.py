import stylegan_interface
import torch
from PIL import Image
from cldm.model import create_model
import os
import numpy as np

def test_laten_walk(strat_seed, end_seed, num_frames, outdir='output_carpaint', gen_path='weights/photomat/G_512.pkl', dec_path='weights/photomat/MatUnet_512.pt'):
    stylegan_interface.generate_lanten_w_walk(gen_path, dec_path, outdir, strat_seed, end_seed, num_frames)

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

def write_to_file(path, o):
    with open(path, 'w') as f:
        f.write(str(o))

if __name__ == '__main__':
    #test_random_render_generate(10)
    #test_material_generate(100)
    #test_create_model('diffusion_style.model')
    test_inference()